#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

constexpr int M_BLOCK = 128;
constexpr int N_BLOCK = 128;
constexpr int K_BLOCK = 128;
constexpr int SCALE_BLOCK = 128;

using a_tile = st_fp8e4m3<M_BLOCK / 2, K_BLOCK>; // 2 consumer warpgroups
using b_tile = st_fp8e4m3<N_BLOCK, K_BLOCK>;
using c_tile = st_fl<M_BLOCK / 2, N_BLOCK>;
using a_scale_vec = col_vec<st_fl<M_BLOCK / 2, K_BLOCK>>;
using b_scale_vec = row_vec<st_fl<K_BLOCK, N_BLOCK>/* Transposed */>;

static constexpr int SM_COUNT = 132;

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

struct config
{
    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;

    // num bits required to represent num pipeline stages
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction.
    using instruction_t = int[INSTRUCTION_WIDTH];

    // Timing info
    static constexpr int TIMING_WIDTH = 128;
    using timing_t = int[TIMING_WIDTH];

    // How many semaphores are available for dynamic use?
    static constexpr int DYNAMIC_SEMAPHORES = 32;

    // One controller warp, one load warp, one store warp, and one mma warp.
    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY;

    // Shared memory declared statically
    static constexpr int SCRATCH_BYTES = 4096;
    static constexpr int STATIC_SHARED_MEMORY = 512 + INSTRUCTION_PIPELINE_STAGES * (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 + DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    // Shared memory declared dynamically
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 13, "NUM_PAGES must be 13");

    static constexpr bool TIMING_RECORD_ENABLED = false;

    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

    static constexpr int CONSUMER_REGISTERS = 88;
    static constexpr int NON_CONSUMER_REGISTERS = 136;
};

struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using fp8_matrix = gl<fp8e4m3, 1, 1, -1, -1, a_tile, b_tile>;
    using fp8_scale = gl<float, 1, 1, -1, -1, a_scale_vec, b_scale_vec>;
    using fl_matrix = gl<float, 1, -1, -1, -1, c_tile>;
    instruction_layout instructions;
    fp8_matrix A;
    fp8_scale A_scale;
    fp8_matrix B;
    fp8_scale B_scale;
    fl_matrix C;
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
    static_assert(sizeof(b_tile) == config::PAGE_SIZE);
    static_assert(sizeof(c_tile) == config::PAGE_SIZE * 2);
    static_assert(PIPELINE_STAGES * (sizeof(a_scale_vec) + sizeof(b_scale_vec)) <= config::SCRATCH_BYTES);
    
    struct parsed_instruction {
        int group_id, row, col, red_start, red_end;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            /*
                Instruction format:
                [0] = opcode
                [1] = Group ID
                [2] = Row offset of C, in units of M_BLOCK
                [3] = Col offset of C, in units of N_BLOCK
                [4] = K reduction dimension start index, in units of K_BLOCK (inclusive)
                [5] = K reduction dimension end index, in units of K_BLOCK (exclusive)
            */
            group_id = instruction[1];
            row = instruction[2];
            col = instruction[3];
            red_start = instruction[4];
            red_end = instruction[5];
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &inputs_arrived(state<config> &s, int stage) { return s.semaphores()[stage]; }
    __device__ static inline semaphore &inputs_finished(state<config> &s, int stage) { return s.semaphores()[stage+PIPELINE_STAGES]; }
    __device__ static inline semaphore &outputs_arrived(state<config> &s, int consumer_id) { return s.semaphores()[consumer_id+PIPELINE_STAGES*2]; }
    __device__ static inline int get_a_page(state<config> &s, int stage) { return stage*2 + 0; }
    __device__ static inline int get_b_page(state<config> &s, int stage) { return stage*2 + 1; }
    __device__ static inline int get_store_page(state<config> &s, int consumer_id) { 
        if (consumer_id == 0)
            return 8; // 8, 9
        else
            return 10; // 10, 11
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

            int pipeline_stage = 0;
            uint32_t semaphore_bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s
            for (int i = 0; i < inst.red_end - inst.red_start; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                warp::tma::expect_bytes(inputs_arrived(s, pipeline_stage), sizeof(a_tile)*2 + sizeof(b_tile) + sizeof(a_scale_vec)*2 + sizeof(b_scale_vec));
                __syncwarp(); // none-0 lanes must wait for the tma::expect_bytes to complete
                if (laneid < 2) {
                    int a_page = get_a_page(s, pipeline_stage);
                    if (i < PIPELINE_STAGES) {
                        s.wait_page_ready(a_page);
                    }
                    a_tile &a = *reinterpret_cast<a_tile *>((uint8_t *)s.pages[a_page].data + sizeof(a_tile) * laneid);
                    tma::load_async(a, g.A, {inst.row*2 + laneid, i + inst.red_start}, inputs_arrived(s, pipeline_stage));
                    a_scale_vec &a_scale = *reinterpret_cast<a_scale_vec *>(
                        (char *)s.scratch() + pipeline_stage*(sizeof(a_scale_vec)*2 + sizeof(b_scale_vec)) + laneid*sizeof(a_scale_vec));
                    tma::load_async(a_scale, g.A_scale, {inst.row*2 + laneid}, inputs_arrived(s, pipeline_stage));
                } else if (laneid == 2) {
                    int b_page = get_b_page(s, pipeline_stage);
                    if (i < PIPELINE_STAGES) {
                        s.wait_page_ready(b_page);
                    }
                    b_tile &b = *reinterpret_cast<b_tile *>(s.pages[b_page].data);
                    tma::load_async(b, g.B, {inst.col, i + inst.red_start}, inputs_arrived(s, pipeline_stage));
                    b_scale_vec &b_scale = *reinterpret_cast<b_scale_vec *>(
                        (char *)s.scratch() + pipeline_stage*(sizeof(a_scale_vec)*2 + sizeof(b_scale_vec)) + 2*sizeof(a_scale_vec));
                    tma::load_async(b_scale, g.B_scale, {inst.col}, inputs_arrived(s, pipeline_stage));
                }
                update_phasebit<1>(semaphore_bitfield, pipeline_stage);
            }
            warp::sync();

            for (int i = 0; i < PIPELINE_STAGES; i++, pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_finished(s, pipeline_stage), get_phasebit<1>(semaphore_bitfield, pipeline_stage));
                if (laneid < 2) {
                    int release_pid = pipeline_stage*2 + laneid;
                    s.finish_page(release_pid, config::NUM_CONSUMER_WARPS);
                }
            }

            if (laneid == 0) { // Release unused page (no need to do this first as we are relying on physical page IDs)
                s.wait_page_ready(12);
                s.finish_page(12, config::NUM_CONSUMER_WARPS);
            }
        }
    };
    struct launcher { // no B200-like tensor cores in H100s
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
            for (int i = 0; i < inst.red_end - inst.red_start; i++, update_phasebit<0>(semaphore_bitfield, pipeline_stage), pipeline_stage=ring_advance<PIPELINE_STAGES>(pipeline_stage)) {
                wait(inputs_arrived(s, pipeline_stage), get_phasebit<0>(semaphore_bitfield, pipeline_stage));
                int a_page = get_a_page(s, pipeline_stage);
                int b_page = get_b_page(s, pipeline_stage);
                a_tile &a = *reinterpret_cast<a_tile *>((uint8_t *)s.pages[a_page].data + sizeof(a_tile) * groupid);
                a_scale_vec &a_scale = *reinterpret_cast<a_scale_vec *>(
                    (char *)s.scratch() + pipeline_stage*(sizeof(a_scale_vec)*2 + sizeof(b_scale_vec)) + groupid*sizeof(a_scale_vec));
                b_tile &b = *reinterpret_cast<b_tile *>(s.pages[b_page].data);
                b_scale_vec &b_scale = *reinterpret_cast<b_scale_vec *>(
                    (char *)s.scratch() + pipeline_stage*(sizeof(a_scale_vec)*2 + sizeof(b_scale_vec)) + 2*sizeof(a_scale_vec));
                warpgroup::mma_ABt(acc_fl, a, b);
                col_vec<rt_fl<M_BLOCK / 8, K_BLOCK>> a_scale_reg;
                row_vec<rt_fl<K_BLOCK, N_BLOCK>> b_scale_reg;
                warpgroup::load(a_scale_reg, a_scale);
                warp::load(b_scale_reg, b_scale);
                warpgroup::mma_async_wait();
                warpgroup::sync(groupid + 3);
                warpgroup::arrive(inputs_finished(s, pipeline_stage)); // A, A_scale, B, B_scale all done
                warp::mul_row(acc_fl, acc_fl, a_scale_reg); // dequantize
                warp::mul_col(acc_fl, acc_fl, b_scale_reg); // dequantize
            }

            int store_page = get_store_page(s, groupid);
            c_tile &store_buffer = *reinterpret_cast<c_tile *>(s.pages[store_page].data);
            group<config::NUM_CONSUMER_WARPS>::sync(1); // must sync as we reuse the pages from the last stage
            warpgroup::store(store_buffer, acc_fl);
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
                tma::store_async(g.C, store_buffer, {inst.group_id, inst.row*2 + laneid, inst.col});
            }
            __syncwarp();
            if (laneid == 0) tma::store_async_read_wait();
            __syncwarp();
            if (laneid < 4) {
                int store_page = get_store_page(s, laneid / 2);
                s.finish_page(store_page + (laneid % 2), config::NUM_CONSUMER_WARPS);
            }
        }
    };
};

PYBIND11_MODULE(group_matmul, m) {
    m.doc() = "group_matmul python module";
    m.def("group_matmul", [](
        pybind11::object A,
        pybind11::object A_scale,
        pybind11::object B,
        pybind11::object B_scale,
        pybind11::object C,
        pybind11::object tokens_per_ep,
        pybind11::kwargs kwargs
    ) {
        auto py_shape = A.attr("shape").cast<pybind11::tuple>();
        int M = pybind11::cast<int>(py_shape[0]);
        int K = pybind11::cast<int>(py_shape[1]);

        py_shape = A_scale.attr("shape").cast<pybind11::tuple>();
        if (pybind11::cast<int>(py_shape[0]) != K / SCALE_BLOCK) throw std::runtime_error("Reduction dimension mismatch on A_scale (must be divisible by " + std::to_string(SCALE_BLOCK) + ")");
        if (pybind11::cast<int>(py_shape[1]) != M) throw std::runtime_error("M dimension mismatch on A_scale");

        py_shape = B.attr("shape").cast<pybind11::tuple>();
        int N = pybind11::cast<int>(py_shape[0]);
        if (pybind11::cast<int>(py_shape[1]) != K) throw std::runtime_error("Reduction dimension mismatch on B");

        py_shape = B_scale.attr("shape").cast<pybind11::tuple>();
        if (pybind11::cast<int>(py_shape[0]) != K / SCALE_BLOCK) throw std::runtime_error("Reduction dimension mismatch on B_scale (must be divisible by " + std::to_string(SCALE_BLOCK) + ")");
        if (pybind11::cast<int>(py_shape[1]) != N) throw std::runtime_error("N dimension mismatch on B_scale");

        py_shape = C.attr("shape").cast<pybind11::tuple>();
        int num_ep = pybind11::cast<int>(py_shape[0]);
        if (pybind11::cast<int>(py_shape[1]) != M) throw std::runtime_error("Row dimension mismatch on C");
        if (pybind11::cast<int>(py_shape[2]) != N) throw std::runtime_error("Column dimension mismatch on C");

        py_shape = tokens_per_ep.attr("shape").cast<pybind11::tuple>();
        if (pybind11::cast<int>(py_shape[0]) != num_ep) throw std::runtime_error("Expert dimension mismatch on tokens_per_ep");

        if (M%M_BLOCK != 0) throw std::runtime_error(std::string("M must be divisible by ") + std::to_string(M_BLOCK));
        if (K%K_BLOCK != 0) throw std::runtime_error(std::string("K must be divisible by ") + std::to_string(K_BLOCK));
        if (N%N_BLOCK != 0) throw std::runtime_error(std::string("N must be divisible by ") + std::to_string(N_BLOCK));

        int *tokens_per_ep_host = new int[num_ep];
        int *tokens_per_ep_dev = reinterpret_cast<int *>(tokens_per_ep.attr("data_ptr")().cast<uint64_t>());
        cudaMemcpy(tokens_per_ep_host, tokens_per_ep_dev, num_ep * sizeof(int), cudaMemcpyDeviceToHost);    

        int num_row_blocks = M / M_BLOCK;
        int num_col_blocks = N / N_BLOCK;

        int num_instructions = num_ep * num_row_blocks * num_col_blocks;
        int num_instructions_per_sm = (num_instructions + SM_COUNT - 1) / SM_COUNT;

        int token_index = 0;
        int instruction_index = 0;
        int *instructions_host = new int[SM_COUNT * num_instructions_per_sm * config::INSTRUCTION_WIDTH];
        for (int group_id = 0; group_id < num_ep; group_id++) {
            if (token_index >= K) throw std::runtime_error("token_index (" + std::to_string(token_index) + ") >= K (" + std::to_string(K) + ")");
            for (int row = 0; row < num_row_blocks; row++) {
                for (int col = 0; col < num_col_blocks; col++) {
                    int sm = instruction_index % SM_COUNT;
                    int local_instruction_index = instruction_index / SM_COUNT;
                    instructions_host[sm * num_instructions_per_sm * config::INSTRUCTION_WIDTH + local_instruction_index * config::INSTRUCTION_WIDTH + 0] = GroupMatmulOp<config>::opcode;
                    instructions_host[sm * num_instructions_per_sm * config::INSTRUCTION_WIDTH + local_instruction_index * config::INSTRUCTION_WIDTH + 1] = group_id;
                    instructions_host[sm * num_instructions_per_sm * config::INSTRUCTION_WIDTH + local_instruction_index * config::INSTRUCTION_WIDTH + 2] = row;
                    instructions_host[sm * num_instructions_per_sm * config::INSTRUCTION_WIDTH + local_instruction_index * config::INSTRUCTION_WIDTH + 3] = col;
                    instructions_host[sm * num_instructions_per_sm * config::INSTRUCTION_WIDTH + local_instruction_index * config::INSTRUCTION_WIDTH + 4] = token_index / K_BLOCK;
                    instructions_host[sm * num_instructions_per_sm * config::INSTRUCTION_WIDTH + local_instruction_index * config::INSTRUCTION_WIDTH + 5] = (token_index + tokens_per_ep_host[group_id]) / K_BLOCK;
                    instruction_index++;
                }
            }
            token_index += tokens_per_ep_host[group_id];
        }
        while (instruction_index % SM_COUNT != 0) {
            int sm = instruction_index % SM_COUNT;
            int local_instruction_index = instruction_index / SM_COUNT;
            instructions_host[sm * num_instructions_per_sm * config::INSTRUCTION_WIDTH + local_instruction_index * config::INSTRUCTION_WIDTH] = 0; // no-op
            instruction_index++;
        }

        int *instructions_dev;
        cudaMalloc(&instructions_dev, SM_COUNT * num_instructions_per_sm * config::INSTRUCTION_WIDTH * sizeof(int));
        cudaMemcpy(instructions_dev, instructions_host, SM_COUNT * num_instructions_per_sm * config::INSTRUCTION_WIDTH * sizeof(int), cudaMemcpyHostToDevice);
        typename globals::instruction_layout instructions_g{instructions_dev, nullptr, (unsigned long)SM_COUNT, (unsigned long)num_instructions_per_sm, nullptr};

        globals __g__{
            instructions_g,
            py::from_object<typename globals::fp8_matrix>::make(A),
            py::from_object<typename globals::fp8_scale>::make(A_scale),
            py::from_object<typename globals::fp8_matrix>::make(B),
            py::from_object<typename globals::fp8_scale>::make(B_scale),
            py::from_object<typename globals::fl_matrix>::make(C)
        };

        cudaStream_t raw_stream = nullptr;
        if (kwargs.contains("stream")) {
            uintptr_t stream_ptr = kwargs["stream"].attr("cuda_stream").cast<uintptr_t>();
            raw_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        }

        int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
        cudaFuncSetAttribute(kvm<config, globals, GroupMatmulOp<config>>, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
        kvm<config, globals, GroupMatmulOp<config>><<<__g__.grid(), __g__.block(), __dynamic_shared_memory__, raw_stream>>>(__g__);

        delete[] instructions_host;
    });
}
