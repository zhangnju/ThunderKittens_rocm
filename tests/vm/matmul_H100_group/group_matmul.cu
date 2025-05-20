#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

constexpr int M_BLOCK = 128;
constexpr int N_BLOCK = 256;
constexpr int K_BLOCK = 128;

using a_tile = st_fp8e4m3<M_BLOCK / 2, K_BLOCK>; // 2 consumer warpgroups
using b_tile = st_fp8e4m3<N_BLOCK, K_BLOCK>;
using c_tile = st_fp8e4m3<M_BLOCK / 2, N_BLOCK>;

static constexpr int SM_COUNT = 132;
static constexpr int NUM_GROUPS = 16; // must be decided at compile-time

template<ducks::gl::all _GL, size_t _N>
struct gl_array {
    using GL = _GL;
    static constexpr int N = _N;

    GL gls[N];

    __host__ inline gl_array(pybind11::list l) : 
        gl_array(std::make_index_sequence<N>{}, l) { }

    template<size_t... I> __host__ inline gl_array(std::index_sequence<I...>, pybind11::list l) : 
        gls{py::from_object<GL>::make(l[I])...} { }

    __host__ __device__ GL& operator[](size_t i) { return gls[i]; }
    __host__ __device__ const GL& operator[](size_t i) const { return gls[i]; }
};

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using fp8_matrix = gl<fp8e4m3, 1, -1, -1, -1, a_tile, b_tile, c_tile>;
    instruction_layout instructions;
    timing_layout timings;
    gl_array<fp8_matrix, NUM_GROUPS> A, B, C;
    dim3 grid() { return dim3(SM_COUNT); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct GroupMatmulOp {
    static constexpr int opcode = 1;
    static constexpr int PIPELINE_STAGES = 4;

    static_assert(config::NUM_PAGES == 13);
    static_assert(config::PAGE_SIZE == 16384);
    static_assert(PIPELINE_STAGES >= 2);
    static_assert(config::NUM_CONSUMER_WARPS == 8);
    static_assert(sizeof(a_tile) == config::PAGE_SIZE / 2);
    static_assert(sizeof(b_tile) == config::PAGE_SIZE * 2);
    static_assert(sizeof(c_tile) == config::PAGE_SIZE);
    
    struct parsed_instruction {
        int group_id, row, col, num_iters;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            /*
                Instruction format:
                [0] = opcode
                [1] = Group ID
                [2] = Row offset of C, in units of M_BLOCK
                [3] = Col offset of C, in units of N_BLOCK
                [4] = K reduction dimension, in units of K_BLOCK
            */
            group_id = instruction[1];
            row = instruction[2];
            col = instruction[3];
            num_iters = instruction[4];
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &inputs_arrived(state<config> &s, int stage) { return s.semaphores()[stage]; }
    __device__ static inline semaphore &inputs_finished(state<config> &s, int stage) { return s.semaphores()[stage+PIPELINE_STAGES]; }
    __device__ static inline semaphore &outputs_arrived(state<config> &s, int consumer_id) { return s.semaphores()[consumer_id+PIPELINE_STAGES*2]; }
    __device__ static inline int get_a_page(state<config> &s, int stage) { return stage*3; }
    __device__ static inline int get_b_page(state<config> &s, int stage) { return stage*3 + 1; }
    __device__ static inline int get_store_page(state<config> &s, int consumer_id) {
        return (PIPELINE_STAGES-1)*3 + consumer_id; // use 2 pages from the last stage
    }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            return query;
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for(int i = 0; i < PIPELINE_STAGES; i++) {
                init_semaphore(inputs_arrived(s, i), 1);
                init_semaphore(inputs_finished(s, i), 2);
            }
            for(int i = 0; i < 2; i++) {
                init_semaphore(outputs_arrived(s, i), 4);
            }
            return 2*PIPELINE_STAGES + 2;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warp::laneid();

            if (laneid >= PIPELINE_STAGES*3 && laneid < config::NUM_PAGES) {
                s.wait_page_ready(laneid);
                s.finish_page(laneid, config::NUM_CONSUMER_WARPS); // release unused pages immediately
            }

            int pipeline_stage = 0;
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            for (int i = 0; i < inst.num_iters; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), sizeof(a_tile)*2 + sizeof(b_tile));
                __syncwarp(); // none-0 lanes must wait for the tma::expect_bytes to complete
                if (laneid < 2) {
                    int a_page = get_a_page(s, pipeline_stage);
                    if (i < PIPELINE_STAGES) {
                        s.wait_page_ready(a_page);
                    }
                    a_tile &a = *reinterpret_cast<a_tile *>((uint8_t *)s.pages[a_page].data + sizeof(a_tile) * laneid);
                    tma::load_async(a, g.A[inst.group_id], {inst.row*2 + laneid, i}, inputs_arrived(s, pipeline_stage));
                } else if (laneid == 2) {
                    int b_page = get_b_page(s, pipeline_stage);
                    if (i < PIPELINE_STAGES) {
                        s.wait_page_ready(b_page);
                        s.wait_page_ready(b_page + 1); // because b_page is a megapage
                    }
                    b_tile &b = *reinterpret_cast<b_tile *>(s.pages[b_page].data);
                    tma::load_async(b, g.B[inst.group_id], {inst.col, i}, inputs_arrived(s, pipeline_stage));
                }
                update_phasebit<1>(semaphore_bitfield, pipeline_stage);
            }
            warp::sync();

            for (int i = 0; i < PIPELINE_STAGES; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                if ((laneid < 3 && pipeline_stage < PIPELINE_STAGES - 1) || (laneid == 2 && pipeline_stage == PIPELINE_STAGES - 1)) {
                    int release_pid = pipeline_stage*3 + laneid;
                    s.finish_page(release_pid, config::NUM_CONSUMER_WARPS);
                }
            }
        }
    };
    struct launcher { // no warpgroup-level tensor cores in H100s
        static __device__ void run(const globals &g, state<config> &s) { }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warpgroup::laneid();
            int warpid = warpgroup::warpid();
            int groupid = warpgroup::groupid();

            rt_fl<M_BLOCK / config::NUM_CONSUMER_WARPS, N_BLOCK> acc_fl;
            warp::zero(acc_fl);

            int pipeline_stage = 0;
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            for (int i = 0; i < inst.num_iters; i++, update_phasebit<0>(semaphore_bitfield, pipeline_stage), pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                int a_page = get_a_page(s, pipeline_stage);
                int b_page = get_b_page(s, pipeline_stage);
                a_tile &a = *reinterpret_cast<a_tile *>((uint8_t *)s.pages[a_page].data + sizeof(a_tile) * groupid);
                b_tile &b = *reinterpret_cast<b_tile *>(s.pages[b_page].data);
                warpgroup::mma_ABt(acc_fl, a, b);
                warpgroup::mma_async_wait();
                warpgroup::arrive(inputs_finished(s, pipeline_stage));
            }

            rt_fp8e4m3<M_BLOCK / config::NUM_CONSUMER_WARPS, N_BLOCK> acc_fp8;
            warp::copy(acc_fp8, acc_fl);

            int store_page = get_store_page(s, groupid);
            c_tile &store_buffer = *reinterpret_cast<c_tile *>(s.pages[store_page].data);
            group<config::NUM_CONSUMER_WARPS>::sync(1); // must sync as we reuse the pages from the last stage
            warpgroup::store(store_buffer, acc_fp8);
            __syncwarp();
            warp::arrive(outputs_arrived(s, groupid));
        }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int laneid = warp::laneid();

            if (laneid < 2) {
                wait(outputs_arrived(s, laneid), 0);
                int store_page = get_store_page(s, laneid);
                c_tile &store_buffer = *reinterpret_cast<c_tile *>(s.pages[store_page].data);
                tma::store_async(g.C[inst.group_id], store_buffer, {inst.row*2 + laneid, inst.col});
            }
            __syncwarp();
            if (laneid == 0) tma::store_async_read_wait();
            __syncwarp();
            if (laneid < 2) {
                int store_page = get_store_page(s, laneid);
                s.finish_page(store_page, config::NUM_CONSUMER_WARPS);
            }
        }
    };
};

PYBIND11_MODULE(group_matmul, m) {
    m.doc() = "group_matmul python module";
    m.def("group_matmul", [](
        pybind11::object instructions,
        pybind11::object timings,
        pybind11::object A,
        pybind11::object B,
        pybind11::object C,
        pybind11::kwargs kwargs
    ) {
        globals __g__{
            py::from_object<typename globals::instruction_layout>::make(instructions),
            py::from_object<typename globals::timing_layout>::make(timings),
            gl_array<typename globals::fp8_matrix, NUM_GROUPS>(pybind11::cast<pybind11::list>(A)),
            gl_array<typename globals::fp8_matrix, NUM_GROUPS>(pybind11::cast<pybind11::list>(B)),
            gl_array<typename globals::fp8_matrix, NUM_GROUPS>(pybind11::cast<pybind11::list>(C))
        };
        cudaStream_t raw_stream = nullptr;
        if (kwargs.contains("stream")) {
            uintptr_t stream_ptr = kwargs["stream"].attr("cuda_stream").cast<uintptr_t>();
            raw_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        }
        int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
        cudaFuncSetAttribute(kvm<config, globals, GroupMatmulOp<config>>, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
        kvm<config, globals, GroupMatmulOp<config>><<<__g__.grid(), __g__.block(), __dynamic_shared_memory__, raw_stream>>>(__g__);
    });
}
