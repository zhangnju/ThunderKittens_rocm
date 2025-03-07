#define KITTENS_TIMINGS

#include "vm.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

template<typename _config=default_config> struct Matmul_layout {
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
        int num_iters, row, col, iter;
    };
    struct producer_state {};
    struct consumer_state {
        page_alloc prev_pages;
        rt_fl<16, 256> c;
    };
    struct input_block {
        a_tile a[2];
        b_tile b;
    };
    struct output_block {
        c_tile c[2];
    };
};
struct Matmul {
    static constexpr int opcode = 1;
    using layout = Matmul_layout<>;
    using config = layout::config;
    using globals = layout::globals;
    using kvms_t = kittens_virtual_machine_state<config>;
    struct controller {
        __device__ static inline void setup(const globals &g, kvms_t &kvms, typename layout::controller_state &control) {
            if(threadIdx.x == 0) kvms.record(0);
            control.num_iters = kvms.instruction[1];
            control.row = kvms.instruction[2];
            control.col = kvms.instruction[3];
            control.iter = -1; // this is gross and needs to be fixed at some point.
            control.mini_pages = {0, 0}; // Not using these.
        }
        __device__ static inline bool run(const globals &g, kvms_t &kvms, typename layout::controller_state &control) {
            if(control.iter+1 >= control.num_iters) return false;
            control.pages = kvms.get_pages(sizeof(typename layout::input_block));
            control.iter++;
            return true;
        }
        __device__ static inline void finish(const globals &g, kvms_t &kvms, typename layout::controller_state &control) {
            control.pages = kvms.get_pages(sizeof(typename layout::output_block));
        }
    };
    struct producer {
        __device__ static inline void setup(const globals &g, kvms_t &kvms, typename layout::controller_state &control, typename layout::producer_state &state) {}
        __device__ static inline void run(const globals &g, kvms_t &kvms, typename layout::controller_state &control, typename layout::producer_state &state) {
            typename layout::input_block *input = (typename layout::input_block*)kvms.page_addr(control.pages.start);
            wait_finished(kvms, control);
            if(warpgroup::laneid() == 0) {
                semaphore &sem = vm::expect(kvms, control, 32, *input);
                if(control.iter < 56) kvms.record(64+control.iter);
                tma::load_async<axis::ROW, cache_policy::EVICT_LAST>(input->a[0], g.a, {control.row*2 + 0, control.iter}, sem);
                tma::load_async<axis::ROW, cache_policy::EVICT_LAST>(input->a[1], g.a, {control.row*2 + 1, control.iter}, sem);
                tma::load_async<axis::ROW, cache_policy::EVICT_LAST>(input->b, g.b, {control.iter, control.col}, sem);
            }
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void finish(const globals &g, kvms_t &kvms, typename layout::controller_state &control, typename layout::producer_state &state) {
            if(laneid() == 0) arrive(*kvms.global_semaphore_ready);
        }
    };
    struct consumer {
        __device__ static inline void setup(const globals &g, kvms_t &kvms, typename layout::controller_state &control, typename layout::consumer_state &state) {
            if(threadIdx.x == 0) kvms.record(1);
            zero(state.c);
            if(threadIdx.x == 0) kvms.record(2);
        }
        __device__ static inline void run(const globals &g, kvms_t &kvms, typename layout::controller_state &control, typename layout::consumer_state &state) {
            auto *input = kvms.page_addr<typename layout::input_block>(control.pages.start);
            wait_arrived(kvms, control);
            warpgroup::mma_AB(state.c, input->a[warpgroup::groupid()], input->b);
            if(threadIdx.x == 0 && control.iter < 56) kvms.record(8+control.iter);
            warpgroup::mma_async_wait<1>();
            if(warpgroup::laneid() == 0 && control.iter > 0) {
                #pragma unroll
                for(int i = 0; i < state.prev_pages.num; i++) {
                    kittens::arrive(kvms.pages.semaphore[state.prev_pages.start+i], 16);
                }
            }
            state.prev_pages = control.pages;
        }
        __device__ static inline void finish(const globals &g, kvms_t &kvms, typename layout::controller_state &control, typename layout::consumer_state &state) {
            warpgroup::mma_async_wait();
            if(threadIdx.x == 0) kvms.record(126);
            auto *output = kvms.page_addr<typename layout::output_block>(control.pages.start);
            warpgroup::store(output->c[warpgroup::groupid()], state.c);
            warpgroup::sync(warpgroup::groupid());
            if(warpgroup::laneid() == 0) {
                tma::store_async<axis::ROW, cache_policy::EVICT_FIRST>(g.c, output->c[warpgroup::groupid()], {control.row*2 + warpgroup::groupid(), control.col});
                // Free up that last chunk of memory
                #pragma unroll
                for(int i = 0; i < state.prev_pages.num; i++) {
                    kittens::arrive(kvms.pages.semaphore[state.prev_pages.start+i], 16);
                }
                tma::store_async_read_wait();
                if(warpgroup::laneid() == 0) arrive(*kvms.global_semaphore_ready, 4);
            }
            if(threadIdx.x == 0) kvms.record(127);
        }
    };
};


PYBIND11_MODULE(matmul, m) {
    m.doc() = "matmul vm test python module";
    kittens::py::bind_kernel<vm::kernel<default_config, typename Matmul::globals, Matmul>>(m, "matmul",
        &Matmul::globals::instructions,
#ifdef KITTENS_TIMINGS
        &Matmul::globals::timings,
#endif
        &Matmul::globals::semaphore,
        &Matmul::globals::a,
        &Matmul::globals::b,
        &Matmul::globals::c
    );
}