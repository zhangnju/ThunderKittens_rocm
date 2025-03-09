#define KITTENS_TIMINGS

#include "vm.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

template<typename _config=default_config> struct TestOp_layout {
    using config = _config;
    using a_tile = st_bf<64, 64>;
    using b_tile = st_bf<64, 256>;
    using c_tile = st_bf<64, 256>;
    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using c_gl = gl<bf16, 1, 1, -1, -1, c_tile>;
    struct globals {
        instruction_layout<config> instructions;
        timing_layout<config> timings;
        semaphore_layout<config> semaphore;
        a_gl a;
        b_gl b;
        c_gl c;

        int dynamic_shared_memory() { return 226000; }
        dim3 grid()  { return dim3(132); }
        dim3 block() { return dim3(config::NUM_WARPS*WARP_THREADS); }
    };
    struct controller_state : base_controller_state {
        int num_iters, iter;
        int row, col;
    };
    struct producer_state {};
    struct consumer_state {
        rt_fl<16, 256> c;
    };
    struct input_block {
        st_bf<64, 64> a[2];
        st_bf<64, 256> b;
    };
    struct output_block {
        st_bf<64, 256> c[2];
    };
};
struct TestOp {
    static constexpr int opcode = 1; // Whatever opcode you want.
    using layout = TestOp_layout<>;
    using config = layout::config;
    using globals = layout::globals;
    using kvms_t = kittens_virtual_machine_state<config>;
    struct controller {
        __device__ static inline void setup(const globals &g, kvms_t &kvms, typename layout::controller_state &control) {
            if(threadIdx.x == 0) kvms.record(0);
            control.num_iters = kvms.instruction[1];
            control.row = kvms.instruction[2];
            control.col = kvms.instruction[3];
            control.iter = 0; // this is gross and needs to be fixed at some point.
            control.mini_pages = {0, 0}; // Not using these.
        }
        __device__ static inline bool run(const globals &g, kvms_t &kvms, typename layout::controller_state &control) {
            if(control.iter < control.num_iters) {
                control.pages = kvms.get_pages(sizeof(typename layout::input_block));
                return true;
            }
            return false;
        }
        __device__ static inline void advance(const globals &g, kvms_t &kvms, typename layout::controller_state &control) {
            control.iter++;
        }
        __device__ static inline void finish(const globals &g, kvms_t &kvms, typename layout::controller_state &control) {
            control.pages = kvms.get_pages(sizeof(typename layout::output_block));
        }
    };
    struct producer {
        __device__ static inline void setup(const globals &g, kvms_t &kvms, typename layout::controller_state &control, typename layout::producer_state &state) {}
        __device__ static inline void run(const globals &g, kvms_t &kvms, typename layout::controller_state &control, typename layout::producer_state &state) {
            vm::wait_finished(kvms, control);
            warpgroup::sync(warpgroup::groupid());
            if(threadIdx.x == 256 && control.iter < (config::TIMING_EVENTS-16) / 2) kvms.record(config::TIMING_EVENTS/2 + control.iter);

            typename layout::input_block *input = (typename layout::input_block*)kvms.page_addr(control.pages.start);
            if(warpgroup::warpid() == 0) {
                // semaphore &sem = vm::expect(kvms, control, 32, *input);
                semaphore &sem = kvms.pages.semaphore[control.pages.start];
                tma::expect(sem, *input);
                if(laneid() == 0) arrive(sem, 31);
                tma::load_async<axis::ROW, cache_policy::EVICT_LAST>(input->a[0], g.a, {control.row*2 + 0, control.iter}, sem);
                tma::load_async<axis::ROW, cache_policy::EVICT_LAST>(input->a[1], g.a, {control.row*2 + 1, control.iter}, sem);
                tma::load_async<axis::ROW, cache_policy::EVICT_LAST>(input->b, g.b, {control.iter, control.col}, sem);
            }
        }
        __device__ static inline void finish(const globals &g, kvms_t &kvms, typename layout::controller_state &control, typename layout::producer_state &state) {
            vm::wait_finished(kvms, control);
            warpgroup::sync(warpgroup::groupid());

            // Memory is now released for the consumer.

            if(warpgroup::warpid() == 0) vm::arrive(kvms, control, 32);
            if(laneid() == 0) arrive(*kvms.global_semaphore_ready);
        }
    };
    struct consumer {
        __device__ static inline void setup(const globals &g, kvms_t &kvms, typename layout::controller_state &control, typename layout::consumer_state &state) {
            if(threadIdx.x == 0) kvms.record(1);

            // Setup load here.
            zero(state.c);

            if(threadIdx.x == 0) kvms.record(2);
        }
        __device__ static inline void run(const globals &g, kvms_t &kvms, typename layout::controller_state &control, typename layout::consumer_state &state) {
            auto *input = kvms.page_addr<typename layout::input_block>(control.pages.start);
            // vm::wait_arrived(kvms, control);
            kittens::wait(kvms.pages.semaphore[control.pages.start], 0);
            if(threadIdx.x == 0 && control.iter < (config::TIMING_EVENTS-16) / 2) kvms.record(8 + control.iter);
            
            // Do work here.
            warpgroup::mma_AB(state.c, input->a[warpgroup::groupid()], input->b);
            warpgroup::mma_async_wait();

            warpgroup::sync(warpgroup::groupid());
            if(warpgroup::warpid() == 0) {
                if(laneid() == 0) kittens::arrive(kvms.pages.semaphore[control.pages.start], 16);
                // vm::arrive(kvms, control, 32 / 2); // 2 consumer warp groups
            }
        }
        __device__ static inline void finish(const globals &g, kvms_t &kvms, typename layout::controller_state &control, typename layout::consumer_state &state) {
            if(threadIdx.x == 0) kvms.record(126);

            vm::wait_arrived(kvms, control);

            auto *output = kvms.page_addr<typename layout::output_block>(control.pages.start);
            warpgroup::store(output->c[warpgroup::groupid()], state.c);
            warpgroup::sync(warpgroup::groupid());
            if(warpgroup::warpid() == 0) {
                tma::store_async<axis::ROW, cache_policy::EVICT_FIRST>(g.c, output->c[warpgroup::groupid()], {control.row*2 + warpgroup::groupid(), control.col});
                tma::store_async_read_wait();
            }

            warpgroup::sync(warpgroup::groupid());
            if(warpgroup::warpid() == 0) {
                vm::arrive(kvms, control, 32 / 2); // 2 consumer warp groups
            }
            if(laneid() == 0) arrive(*kvms.global_semaphore_ready);
            
            if(threadIdx.x == 0) kvms.record(127);
        }
    };
};


PYBIND11_MODULE(test, m) {
    m.doc() = "test vm test python module";
    kittens::py::bind_kernel<vm::kernel<default_config, typename TestOp::globals, TestOp>>(m, "matmul",
        &TestOp::globals::instructions,
#ifdef KITTENS_TIMINGS
        &TestOp::globals::timings,
#endif
        &TestOp::globals::semaphore,
        &TestOp::globals::a,
        &TestOp::globals::b,
        &TestOp::globals::c
    );
}