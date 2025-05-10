#define RED_TEXT "\033[31m"
#define GREEN_TEXT "\033[32m"
#define YELLOW_TEXT "\033[33m"
#define BLUE_TEXT "\033[34m"
#define MAGENTA_TEXT "\033[35m"
#define CYAN_TEXT "\033[36m"
#define WHITE_TEXT "\033[37m"
#define RESET_TEXT "\033[0m"

#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

/*
Instruction format:
[0] = opcode
[1] = Row offset of C, in units of 128
[2] = Col offset of C, in units of 128
[3] = K reduction dimension, in units of 128
*/

struct config {
    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;

    // num bits required to represent num pipeline stages
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 2;

    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction.
    using instruction_t = int[INSTRUCTION_WIDTH];

    // Timing info
    static constexpr int TIMING_WIDTH = 64;
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
    static constexpr int SCRATCH_BYTES = 1024;
    static constexpr int STATIC_SHARED_MEMORY = 512 + INSTRUCTION_PIPELINE_STAGES * (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 + DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    // Shared memory declared dynamically
    static constexpr int PAGE_SIZE = 128 * 576;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    

    static constexpr bool TIMING_RECORD_ENABLED = true;

    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

    static constexpr int CONSUMER_REGISTERS = 104;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};

static constexpr int QKRot_D = 64, QVO_D = 512, QVO_Dd2 = QVO_D/2, NUM_ROWS = 128, PAGE_SIZE = 256, Q_HEADS = 16, Q_STEP = 8;

template<int rows, int cols> using st_fp8 = st_fp8e4m3<rows, cols>;
template<int rows, int cols> using rt_fp8 = rt_fp8e4m3<rows, cols>;
using fp8 = fp8e4m3;

struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;

    instruction_layout instructions;

    gl<fp8, -1, -1, -1, QKRot_D, typename PipelineOp::qrot_st_type> Q; // batch, new_tokens, q_heads, QKRot_D
    gl<fp8, -1, -1, -1, QVO_D, typename PipelineOp::qvo_st_type> QV; // batch, new_tokens, q_heads, QVO_D

    gl<fp8, 1, -1, PAGE_SIZE, QKRot_D, typename PipelineOp::kcache_st_type> K_cache; // 1, num_pages, PAGE_SIZE, QKRot_D
    gl<fp8, 1, -1, PAGE_SIZE, QVO_D, typename PipelineOp::vcache_st_type> V_cache; // 1, num_pages, PAGE_SIZE, QVO_D

    gl<int, 1, 1, -1, -1> Table; // B, num_pages
    gl<fp8, -1, -1, -1, QVO_D, st_fp8<16, QVO_Dd2>, st_fp8<16, QVO_D/config::NUM_CONSUMER_WARPS>> O; // batch_size, new_tokens, q_heads, QVO_D
    
    gl<float, -1, -1, Q_HEADS, QVO_D, st_fl<16, QVO_D/config::NUM_CONSUMER_WARPS>, st_fl<16,256>> O_scratch; // num_instructions, new_tokens, q_heads, QVO_D
    gl<float,  1, -1, -1, Q_HEADS, sv_fl<16>> Lvec_scratch; // num_instructions, new_tokens, q_heads

    gl<int,    1,  1,  -1, -1> Bar; // 1, 1, num_instructions, new_tokens

    float softmax_scale;

    int tic;

    timing_layout timings;

    dim3 grid() { return dim3(148); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

struct location {
    int batch_idx; // batch_idx >=0, otherwise it's the negative index, minus one, into scratch
    int seq_idx;
};

template <typename config = config, int _OP_IDX = 0>
struct PartialOp
{
    static constexpr int opcode = 1;
    static constexpr int OP_IDX = _OP_IDX; // Op index within the layer -- controls which barrier to listen to.
    struct parsed_instruction
    {
        int uid;
        location output_location;
        int q_batch_idx, q_seq_idx;
        int start_pos, end_pos, length;
        int num_iters;

        __device__ static inline int get_num_iters(int start_pos, int end_pos) {
            return (end_pos - start_pos + NUM_ROWS - 1) / NUM_ROWS;
        }

        __device__ static inline int get_length(int length, int q_seq_idx, int q_depth) {
            return length - (q_depth - (q_seq_idx + warpgroup::warpid()) - 1);
        }

        __device__ inline parsed_instruction(typename config::instruction_t &instruction, int q_depth)
        {
            uid = instruction[1];
            output_location = {instruction[2], instruction[3]};
            q_batch_idx = instruction[4];
            q_seq_idx = instruction[5];
            start_pos = instruction[6];
            end_pos = instruction[7];
            length = get_length(length, q_seq_idx, q_depth);
            num_iters = get_num_iters(start_pos, end_pos);
        }
        __device__ inline parsed_instruction(state<config> &s) : parsed_instruction(s.instruction()) {}
    };
    static __device__ inline parsed_instruction parse_instruction(const globals &g, state<config> &s)
    {
        return parsed_instruction{
            s.instruction()[1], 
            s.instruction()[2], 
            {s.instruction()[3], s.instruction()[4]}, 
            s.instruction()[5], 
            s.instruction()[6], 
            s.instruction()[7], 
            parsed_instruction::get_length(s.instruction()[8], s.instruction()[5], g.Q.depth()),
            parsed_instruction::get_num_iters(s.instruction()[6], s.instruction()[7]),
        };
    }

    static constexpr int NUM_PIPELINE_STAGES = 2;
    static constexpr int NUM_CONSUMER_WARPS = config::NUM_CONSUMER_WARPS;
    static constexpr int NUM_PAGES = config::NUM_PAGES;
    static constexpr int NUM_ATTN_ARITH_WARPS = NUM_CONSUMER_WARPS / 2;
    static constexpr int NUM_ATTN_ARITH_WARPGROUPS = NUM_ATTN_ARITH_WARPS / 4;
    static constexpr int NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS / 4;

    // Pipelined objects
    using kcache_st_type         = st_fp8<NUM_ROWS, QKRot_D>;
    static constexpr int KCACHE_TILE_SIZE = sizeof(kcache_st_type::dtype) * kcache_st_type::rows * kcache_st_type::cols;
    static constexpr int KCACHE_SIZE = NUM_PIPELINE_STAGES * KCACHE_TILE_SIZE;

    using vcache_st_type         = st_fp8<NUM_ROWS, QVO_D>;
    static constexpr int VCACHE_SIZE = NUM_PIPELINE_STAGES * sizeof(vcache_st_type::dtype) * vcache_st_type::rows * vcache_st_type::cols;

    using vcache_half_st_type         = st_fp8<NUM_ROWS, QVO_D/2>;
    static constexpr int VCACHE_HALF_TILE_SIZE = sizeof(vcache_half_st_type::dtype) * vcache_half_st_type::rows * vcache_half_st_type::cols;

    static_assert(KCACHE_SIZE + VCACHE_SIZE == NUM_PIPELINE_STAGES * config::PAGE_SIZE, "KCACHE_SIZE + VCACHE_SIZE must fit in NUM_PIPELINE_STAGES * config::PAGE_SIZE");
    static_assert(NUM_PAGES >= NUM_PIPELINE_STAGES + 1, "NUM_PAGES must be greater than NUM_PIPELINE_STAGES");

    // Non-pipelined objects
    using qrot_st_type           = st_fp8<Q_STEP * Q_HEADS, QKRot_D>;
    static constexpr int QROT_SIZE = sizeof(qrot_st_type::dtype) * qrot_st_type::rows * qrot_st_type::cols;

    using qvo_st_type            = st_fp8<Q_STEP * Q_HEADS, QVO_D>;
    static constexpr int QVO_SIZE = sizeof(qvo_st_type::dtype) * qvo_st_type::rows * qvo_st_type::cols;

    static_assert(QROT_SIZE + QVO_SIZE == config::PAGE_SIZE, "QROT_SIZE + QVO_SIZE must fit in one page");

    using o_st_type = st_fl<Q_STEP * Q_HEADS, QVO_D/NUM_CONSUMER_WARPS>;
    static constexpr int O_SIZE = sizeof(o_st_type::dtype) * o_st_type::rows * o_st_type::cols;
    static constexpr int NUM_O_SHARED_PAGES = (O_SIZE * NUM_CONSUMER_WARPS/2 + config::PAGE_SIZE - 1) / config::PAGE_SIZE;
    static_assert(NUM_O_SHARED_PAGES <= NUM_PAGES, "NUM_O_SHARED_PAGES must be less than NUM_PAGES");
    static constexpr int NUM_O_SHARED_BLOCKS_PER_PAGE = (NUM_CONSUMER_WARPS/2) / NUM_O_SHARED_PAGES;

    using o_fp8_st_type = st_fp8<Q_STEP * Q_HEADS, QVO_D/(4*NUM_CONSUMER_WARPS)>;
    static constexpr int O_FP8_SIZE = sizeof(o_fp8_st_type::dtype) * o_fp8_st_type::rows * o_fp8_st_type::cols;
    static_assert(4 * O_FP8_SIZE <= config::PAGE_SIZE, "O_FP8_SIZE must be less than PAGE_SIZE/4");

    using l_sv_type = sv_fl<Q_STEP * Q_HEADS / NUM_ATTN_ARITH_WARPS>;
    static constexpr int L_SIZE = NUM_ATTN_ARITH_WARPS * sizeof(l_sv_type::dtype) * l_sv_type::length;
    static_assert(L_SIZE <= config::SCRATCH_BYTES, "L_SIZE must be less than SCRATCH_BYTES");

    // Tensor objects
    using attn_tt_type = tt<float, Q_STEP*Q_HEADS, NUM_ROWS>;
    static constexpr int ATTN_TT_SIZE = sizeof(attn_tt_type::dtype) * attn_tt_type::rows * attn_tt_type::cols;

    using o_tt_quarter_type = tt<float, Q_STEP*Q_HEADS, QVO_D/4>;
    static constexpr int O_TT_QUARTER_SIZE = sizeof(o_tt_quarter_type::dtype) * o_tt_quarter_type::rows * o_tt_quarter_type::cols;

    using o_tt_half_type = tt<float, Q_STEP*Q_HEADS, QVO_D/2>;
    static constexpr int O_TT_HALF_SIZE = sizeof(o_tt_half_type::dtype) * o_tt_half_type::rows * o_tt_half_type::cols;

    //  semaphores 
    __device__ static inline semaphore &inputs_arrived(state<config> &s, int i) { return s.semaphores()[i]; }
    __device__ static inline semaphore &inputs_finished(state<config> &s, int i) { return s.semaphores()[NUM_PIPELINE_STAGES + i]; }
    __device__ static inline semaphore &matmul_finished(state<config> &s) { return s.semaphores()[2 * NUM_PIPELINE_STAGES]; }
    __device__ static inline semaphore &out_arrived(state<config> &s) { return s.semaphores()[2 * NUM_PIPELINE_STAGES + 1]; }
    __device__ static inline semaphore &out_written(state<config> &s, int i) { return s.semaphores()[2 * NUM_PIPELINE_STAGES + 2 + i]; }

    // getters
    __device__ static inline kcache_st_type &get_kcache_tile(state<config> &s, int stage) {
        return reinterpret_cast<kcache_st_type*>(s.pages[s.pid(stage)].ptr());
    }

    __device__ static inline vcache_st_type &get_vcache_tile(state<config> &s, int stage) {
        return reinterpret_cast<vcache_st_type*>(
            (uint8_t*) s.pages[s.pid(stage)].ptr()
            + KCACHE_TILE_SIZE
        );
    }

    __device__ static inline vcache_half_st_type &get_vcache_half_tile(state<config> &s, int stage, int half) {
        return reinterpret_cast<vcache_half_st_type(&)[2]>(
            (uint8_t*) s.pages[s.pid(stage)].ptr()
            + KCACHE_TILE_SIZE
        )[half];
    }

    __device__ static inline qrot_st_type &get_qrot_tile(state<config> &s) {
        return reinterpret_cast<qrot_st_type*>(s.pages[s.pid(NUM_PIPELINE_STAGES)].ptr());
    }

    __device__ static inline qvo_st_type &get_qvo_tile(state<config> &s) {
        return reinterpret_cast<qvo_st_type*>(
            (uint8_t*) s.pages[s.pid(NUM_PIPELINE_STAGES)].ptr()
            + QROT_SIZE
        );
    }

    __device__ static inline o_st_type &get_o_tile(state<config> &s, int page, int warp) {
        return reinterpret_cast<o_st_type*>(
            s.pages[s.pid(page)].ptr()
        )[warp];
    }

    __device__ static inline o_fp8_st_type &get_o_fp8_tile(state<config> &s, int quarter, int warp) {
        return reinterpret_cast<o_fp8_st_type*>(s.pages[s.pid(0)].ptr())[quarter * NUM_CONSUMER_WARPS + warp];
    }

    __device__ static inline l_sv_type &get_l_vec(state<config> &s, int i) {
        return reinterpret_cast<l_sv_type*>(s.scratch.ptr())[i];
    }

    __device__ static inline attn_tt_type &get_attn_tt(state<config> &s) {
        return s.tensor_alloc.template allocate<attn_tt_type>(0, 0);
    }

    __device__ static inline o_tt_quarter_type &get_o_tt_quarter(state<config> &s, int i) {
        return s.tensor_alloc.template allocate<o_tt_quarter_type>(0, attn_tt_type::cols + i * o_tt_quarter_type::cols);
    }

    __device__ static inline o_tt_half_type &get_o_tt_half(state<config> &s, int i) {
        return s.tensor_alloc.template allocate<o_tt_half_type>(0, attn_tt_type::cols + i * o_tt_quarter_type::cols);
    }

    struct controller
    {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query)
        {
            int ret_order[] = {0, 1, 2};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s)
        {
            for (int i = 0; i < NUM_PIPELINE_STAGES; i++) {
                init_semaphore(inputs_arrived(s, i), 1);
                init_semaphore(inputs_finished(s, i), 1);
            }

            init_semaphore(matmul_finished(s), 2);
            init_semaphore(out_arrived(s), 1);

            for (int i = 0; i < NUM_PAGES; i++) {
                init_semaphore(out_written(s, i), 1);
            }

            return 2 * NUM_PIPELINE_STAGES + 2 + NUM_PAGES;
        }
    };

    struct loader
    {
        static __device__ void run(const globals &g, state<config> &s)
        {
            parsed_instruction inst{s};
            uint32_t bitfield = 0xFFFF0000; // 1 for arrived, 0 for finished

            for (int i = 0; i < NUM_PIPELINE_STAGES; i++) {
                s.wait_page_ready(i);
            }
            __syncwarp();

            for (int iter = 0; iter < inst.num_iters; iter++) {
                int pipeline_stage = iter % NUM_PIPELINE_STAGES;

                int pos = inst.start_pos + iter * NUM_ROWS;
                int within_page_idx = (pos % PAGE_SIZE) / NUM_ROWS;
                int next_page_id = s.globals.Table[coord<>{inst.q_batch_idx, pos/PAGE_SIZE}];
                
                wait(inputs_arrived(s, pipeline_stage), prototype::get_phasebit<1>(bitfield, pipeline_stage));
                prototype::update_phasebit<1>(bitfield, pipeline_stage);

                if (warp::laneid() == 0) {
                    kcache_st_type &k_cache = get_kcache_tile(s, pipeline_stage);
                    vcache_st_type &v_cache = get_vcache_tile(s, pipeline_stage);

                    semaphore &inputs_arrived = inputs_arrived(s, pipeline_stage);

                    tma::expect(inputs_arrived, k_cache, v_cache);
                    tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                        k_cache, 
                        s.globals.K_cache, 
                        {0, next_page_id, within_page_idx, 0}, 
                        inputs_arrived
                    );
                    tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                        v_cache, 
                        s.globals.V_cache, 
                        {0, next_page_id, within_page_idx, 0}, 
                        inputs_arrived
                    );
                }

                __syncwarp();
            }
        }
    };

    struct launcher
    { // launches mma's
        static __device__ void run(const globals &g, state<config> &s) {}
    };

    struct consumer
    {
        static __device__ void run(const globals &g, state<config> &s)
        {
            parsed_instruction inst{s};
            uint32_t bitfield = 0xFFFF0000; // 1 for arrived, 0 for finished

            qrot_st_type &qrot = get_qrot_tile(s);
            qvo_st_type &qvo = get_qvo_tile(s);

            auto qrot_subtile = qrot.template subtile<qrot_st_type::rows/NUM_CONSUMER_WARPS, qrot_st_type::cols>({group<NUM_CONSUMER_WARPGROUPS>::warpid(), 0});
            auto qvo_subtile = qvo.template subtile<qvo_st_type::rows/NUM_CONSUMER_WARPS, qvo_st_type::cols>({group<NUM_CONSUMER_WARPGROUPS>::warpid(), 0});

            s.wait_page_ready(NUM_PIPELINE_STAGES);
            s.wait_tensor_ready();
            group<NUM_CONSUMER_WARPGROUPS>::sync(9);

            using o_rt_quarter_type = rt_fl<o_tt_quarter_type::rows, o_tt_quarter_type::cols>;
            o_rt_quarter_type o_quarter;
            warp::zero(o_quarter);
            group<NUM_CONSUMER_WARPGROUPS>::store(get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 0), o_quarter);
            group<NUM_CONSUMER_WARPGROUPS>::store(get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 1), o_quarter);
            group<NUM_CONSUMER_WARPGROUPS>::store(get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 2), o_quarter);

            tensor_store_wait();
            group<NUM_CONSUMER_WARPGROUPS>::sync(8);

            warp::load_async(qrot_subtile, g.Q, {inst.q_batch_idx, inst.q_seq_idx + group<NUM_CONSUMER_WARPGROUPS>::warpid(), 0, 0});
            warp::load_async(qvo_subtile, g.QV, {inst.q_batch_idx, inst.q_seq_idx + group<NUM_CONSUMER_WARPGROUPS>::warpid(), 0, 0});

            load_async_wait();

            using attn_rt_type = rt_fl<attn_tt_type::rows/NUM_CONSUMER_WARPS, attn_tt_type::cols>;
            using rv_type = col_vec<attn_rt_type>;

            rv_type accumulated_max_vec, accumulated_norm_vec,
                    current_tile_max_vec, current_tile_norm_vec,
                    scaled_accumulated_max_vec, scaled_current_tile_max_vec;

            if(inst.num_iters > 0) warp::neg_infty(accumulated_max_vec);
            else { warp::one(accumulated_max_vec); warp::mul(accumulated_max_vec, accumulated_max_vec, -999999.f); }

            warp::zero(accumulated_norm_vec);

            attn_rt_type attn_rt;
            warp::zero(attn_rt);

            warp::store(accumulated_max_vec, accumulated_norm_vec, attn_rt);

            const float SOFTMAX_TEMPERATURE = g.softmax_scale * 1.44269504089f;

            group<NUM_CONSUMER_WARPS>::sync(10);

            // DEBUGGING MODE
            if (group<NUM_CONSUMER_WARPS>::warpid() == 0) {
                warp::one(qrot_subtile);
                warp::one(qvo_subtile);
            }
            group<NUM_CONSUMER_WARPS>::sync(10);
            // END DEBUGGING MODE

            for (int iter = 0; iter < inst.num_iters; iter++) {
                int pipeline_stage = iter % NUM_PIPELINE_STAGES;


                wait(inputs_arrived(s, pipeline_stage), prototype::get_phasebit<0>(bitfield, pipeline_stage));
                prototype::update_phasebit<0>(bitfield, pipeline_stage);

                attn_tt_type &attn_tt = get_attn_tt(s);
                kcache_st_type &kcache_st = get_kcache_tile(s, pipeline_stage);
                vcache_st_type &vcache_st = get_vcache_tile(s, pipeline_stage);
                group<NUM_CONSUMER_WARPS>::mm_ABt(attn_tt, qrot_subtile, kcache_st, matmul_finished(s));
                group<NUM_CONSUMER_WARPS>::mma_ABt(attn_tt, qvo_subtile, vcache_st, matmul_finished(s));

                // computation not needing matmul_finished
                // warp::copy(current_tile_max_vec, accumulated_max_vec);
                // warp::copy(current_tile_norm_vec, accumulated_norm_vec);

                // wait for matmul to finish and trigger inputs_finished
                wait(matmul_finished(s), 0);
                if (group<NUM_CONSUMER_WARPS>::laneid() == 1) {
                    arrive(inputs_finished(s, pipeline_stage), 1);
                }
                attn_rt_type attn_rt;
                group<NUM_ATTN_ARITH_WARPS>::load_async(attn_rt, attn_tt);
                group<NUM_ATTN_ARITH_WARPS>::load_async_wait();

                // if (inst.start_pos + NUM_ROWS * (iter + 1) >= inst.length) {
                //     // warp::apply(
                //     //     att_block_fp32_rt,
                //     //     att_block_fp32_rt,
                //     //     [length] __device__ (
                //     //         int r,
                //     //         int c,
                //     //         const float &x
                //     //     ) -> float {
                //     //         return (c >= length ? -9999999999.f : x);
                //     //     }
                //     // );
                // }

                warp::one(scaled_accumulated_max_vec);
                warp::mul(attn_rt, attn_rt, 2.f);

                group<NUM_CONSUMER_WARPGROUPS>::store_async(attn_tt, attn_rt);
                tensor_store_wait();

                group<NUM_CONSUMER_WARPGROUPS>::sync(11);

                // shuffle o around, normalize, and mma
                // constant is that before/after shuffling, o_quarter_rt = o[2], o_quarter_tt[0] = o[0], o_quarter_tt[1] = o[1], o_quarter_tt[2] = o[3]
                o_rt_quarter_type o_quarter_scratch;

                // normalize o[2], swap with o[1]
                warp::mul(o_quarter, o_quarter, 3);
                group<NUM_CONSUMER_WARPGROUPS>::load_async(o_quarter_scratch, get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 1));
                group<NUM_CONSUMER_WARPGROUPS>::store_async(get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 1), o_quarter);
                warp::copy(o_quarter, o_quarter_scratch);
                tensor_store_wait();

                // normalize o[1], swap with o[3]
                warp::mul(o_quarter, o_quarter, 2);
                group<NUM_CONSUMER_WARPGROUPS>::load_async(o_quarter_scratch, get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 2));
                group<NUM_CONSUMER_WARPGROUPS>::store_async(get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 2), o_quarter);
                warp::copy(o_quarter, o_quarter_scratch);
                tensor_store_wait();

                // normalize o[3], swap with o[1]
                warp::mul(o_quarter, o_quarter, 4);
                group<NUM_CONSUMER_WARPGROUPS>::load_async(o_quarter_scratch, get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 2));
                group<NUM_CONSUMER_WARPGROUPS>::store_async(get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 2), o_quarter);
                warp::copy(o_quarter, o_quarter_scratch);
                tensor_store_wait();

                // second half of mma
                group<NUM_CONSUMER_WARPGROUPS>::mma_AB(get_o_tt_half(s, 1), get_attn_tt(s), get_vcache_half_tile(s, pipeline_stage, 1), matmul_finished(s));

                // swap o[1] and o[2]
                group<NUM_CONSUMER_WARPGROUPS>::load_async(o_quarter_scratch, get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 1));
                group<NUM_CONSUMER_WARPGROUPS>::store_async(get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 1), o_quarter);
                warp::copy(o_quarter, o_quarter_scratch);
                tensor_store_wait();

                // write o[2] to o_tile[2]
                if (inst.batch_idx >= 0) {
                    warp::store(get_o_tile(s, 2, group<NUM_CONSUMER_WARPGROUPS>::warpid()), o_quarter);
                } else {
                    rt_fp8<o_rt_quarter_type::rows, o_rt_quarter_type::cols> o_quarter_fp8;
                    warp::copy(o_quarter_fp8, o_quarter);
                    warp::store(get_o_fp8_tile(s, 2, group<NUM_CONSUMER_WARPGROUPS>::warpid()), o_quarter_fp8);
                }

                // swap o[2] and o[0]
                group<NUM_CONSUMER_WARPGROUPS>::load_async(o_quarter_scratch, get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 0));
                group<NUM_CONSUMER_WARPGROUPS>::store_async(get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 0), o_quarter);
                warp::copy(o_quarter, o_quarter_scratch);
                tensor_store_wait();

                // normalize o[0] and swap with o[2]
                warp::mul(o_quarter, o_quarter, 1);
                group<NUM_CONSUMER_WARPGROUPS>::load_async(o_quarter_scratch, get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 0));
                group<NUM_CONSUMER_WARPGROUPS>::store_async(get_o_tt_warp_split_quarter(s, group<NUM_CONSUMER_WARPGROUPS>::warpid(), 0), o_quarter);
                warp::copy(o_quarter, o_quarter_scratch);
                tensor_store_wait();

                // first half of mma
                group<NUM_CONSUMER_WARPGROUPS>::mma_AB(get_o_tt_half(s, 0), get_attn_tt(s), get_vcache_half_tile(s, pipeline_stage, 0), matmul_finished(s));
                
                // end of shuffle

                if (inst.batch_idx < 0) {
                    warp::store(get_l_shared_scratch(s, group<NUM_CONSUMER_WARPGROUPS>::warpid()), accumulated_norm_vec);
                }
        
                wait(matmul_finished(s), 1);

                group<NUM_CONSUMER_WARPS>::sync(12);

                if (group<NUM_CONSUMER_WARPS>::laneid() == 1) {
                    arrive(out_arrived(s), 1);
                }
            }
        }
    };


    struct storer
    {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            uint32_t bitfield = 0xFFFF0000; // 1 for arrived, 0 for finished

            if (inst.batch_idx < 0) {
                if (warp::laneid() == 0) {
                    tma::store_async<dim::ROW, cache_policy::EVICT_LAST>(
                        get_o_shared_scratch(s, 2),
                        s.globals.O_shared,
                        {0, 0, 0, 0}
                    );


                }
                warp::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    get_o_shared_scratch(s, 2),
                    s.globals.O_shared,
                    {0, 0, 0, 0},
                    out_written(s, 2)
                );
            } else {
                
            }
            
            wait(out_written(s, 0), 0);
            wait(out_written(s, 1), 0);
            wait(out_written(s, 2), 1);

            if (laneid() == 0) {
                for (int i = 0; i < NUM_PAGES; i++) {
                    s.finish_page(i, NUM_CONSUMER_WARPS);
                }
            }

            asm volatile("fence.sc.gpu;\n"); // Can't reorder across this boundary
            __syncwarp();
            if(inst.batch_idx < 0) {
                if(warp::laneid() < Q_STEP && inst.seq_idx + warp::laneid() < g.O_scratch.depth()) {
                    // Todo: this can probably replaced by a st.async, which may prevent an expensive wait on the final finish barrier.
                    g.Bar[{-inst.batch_idx-1, inst.seq_idx + warp::laneid()}] = g.tic;
                    // For blackwell
                    // asm volatile(
                    //     "st.async.global.b32 [%0], %1;\n"
                    // ::  "l"(&args.globals.semaphore[{-args.common.dst.batch_idx-1, args.common.dst.seq_idx + group<8>::laneid()}]), "r"(args.globals.tic)
                    // :   "memory"
                    // );
                }
            }
        }
    };
};


























































































template <typename config = config, int _OP_IDX = 0>
struct ReductionOp
{
    static constexpr int opcode = 2;
    static constexpr int OP_IDX = _OP_IDX; // Op index within the layer -- controls which barrier to listen to.
    static constexpr int LOAD_UIDS_OFFSET = 6;
    struct parsed_instruction
    {
        int uid, num_iters;
        location output_location;
        int src_uid;

        __device__ inline parsed_instruction(typename config::instruction_t &instruction)
        {
            uid = instruction[1];
            num_iters = instruction[2];
            output_location = {instruction[3], instruction[4]};
            src_uid = instruction[5];
        }
        __device__ inline parsed_instruction(state<config> &s) : parsed_instruction(s.instruction()) {}
    };
    static __device__ inline parsed_instruction parse_instruction(const globals &g, state<config> &s)
    {
        return parsed_instruction{
            s.instruction()[1], 
            s.instruction()[2], 
            {s.instruction()[3], s.instruction()[4]}, 
            s.instruction()[5], 
        };
    }

    static constexpr int NUM_PIPELINE_STAGES = 4;
    static constexpr int NUM_CONSUMER_WARPS = config::NUM_CONSUMER_WARPS;
    static constexpr int NUM_PAGES = config::NUM_PAGES;

    using o_st_type = st_fl<16, QVO_D/NUM_CONSUMER_WARPS>;
    static constexpr int O_TILE_SIZE = sizeof(o_st_type::dtype) * o_st_type::rows * o_st_type::cols;
    static constexpr int O_SHARED_SIZE = NUM_PIPELINE_STAGES * NUM_CONSUMER_WARPS * O_TILE_SIZE;
    static constexpr int NUM_O_SHARED_PAGES = (O_SHARED_SIZE + config::PAGE_SIZE - 1) / config::PAGE_SIZE;
    static_assert(NUM_O_SHARED_PAGES <= NUM_PAGES, "NUM_O_SHARED_PAGES must be less than NUM_PAGES");
    static constexpr int NUM_O_SHARED_STAGES_PER_PAGE = NUM_PIPELINE_STAGES / NUM_O_SHARED_PAGES;

    using l_sv_type = sv_fl<16>;
    static constexpr int L_SHARED_SIZE = 2 * NUM_PIPELINE_STAGES * sizeof(l_sv_type::dtype) * l_sv_type::length;
    static constexpr int O_SHARED_SCRATCH_SIZE = O_TILE_SIZE * NUM_CONSUMER_WARPS;
    static constexpr int L_SHARED_SCRATCH_SIZE = 2 * sizeof(l_sv_type::dtype) * l_sv_type::length;

    using o_st_output_type = st_fp8<16, QVO_D/NUM_CONSUMER_WARPS>;
    static constexpr int O_SHARED_OUTPUT_SIZE = NUM_CONSUMER_WARPS * sizeof(o_st_output_type::dtype) * o_st_output_type::rows * o_st_output_type::cols;
    static_assert(L_SHARED_SIZE + O_SHARED_SCRATCH_SIZE + L_SHARED_SCRATCH_SIZE + O_SHARED_OUTPUT_SIZE <= config::PAGE_SIZE, "L_SHARED_SIZE + O_SHARED_SCRATCH_SIZE + L_SHARED_SCRATCH_SIZE + O_SHARED_OUTPUT_SIZE must fit in one page");

    //  semaphores 
    __device__ static inline semaphore &inputs_arrived(state<config> &s, int i) { return s.semaphores()[i]; }
    __device__ static inline semaphore &inputs_finished(state<config> &s, int i) { return s.semaphores()[NUM_PIPELINE_STAGES + i]; }
    __device__ static inline semaphore &out_arrived(state<config> &s) { return s.semaphores()[2 * NUM_PIPELINE_STAGES]; }

    // getters
    __device__ static inline o_st_type &get_o_shared_tile(state<config> &s, int stage, int w) {
        return reinterpret_cast<o_st_type*>(s.pages[s.pid(stage / NUM_O_SHARED_STAGES_PER_PAGE)].ptr())[(stage % NUM_O_SHARED_STAGES_PER_PAGE) * NUM_CONSUMER_WARPS + w];
    }

    __device__ static inline l_sv_type &get_l_shared_vec(state<config> &s, int stage) {
        return reinterpret_cast<l_sv_type*>(s.pages[s.pid(NUM_O_SHARED_PAGES)].ptr())[stage];
    }

    __device__ static inline o_st_type &get_o_shared_scratch(state<config> &s, int w) {
        return reinterpret_cast<o_st_type*>(
            (uint8_t*)s.pages[s.pid(NUM_O_SHARED_PAGES)].ptr()
            + L_SHARED_SIZE
        )[w];
    }

    __device__ static inline o_st_output_type &get_o_shared_output(state<config> &s, int w) {
        return reinterpret_cast<o_st_output_type*>(
            (uint8_t*)s.pages[s.pid(NUM_O_SHARED_PAGES)].ptr()
            + L_SHARED_SIZE + O_SHARED_SCRATCH_SIZE
        )[w];
    }

    __device__ static inline l_sv_type &get_l_shared_scratch(state<config> &s) {
        return *reinterpret_cast<l_sv_type*>(
            (uint8_t*)s.pages[s.pid(NUM_O_SHARED_PAGES)].ptr()
            + L_SHARED_SIZE + O_SHARED_SCRATCH_SIZE + O_SHARED_OUTPUT_SIZE
        );
    }

    __device__ static inline void wait_barrier(const globals &g, int barrier_x, int barrier_y, const char *barrier_name) {

        long long clock = clock64();
        if(laneid() == 0) {
            while(*(volatile int*)& g.Bar[{barrier_x, barrier_y}] != g.tic) {
                __nanosleep(config::GMEM_SPIN_LOOP_SLEEP_NANOS);
            }
        }

        asm volatile("fence.sc.gpu;\n");
        __syncwarp();

        // if (barrier_name[0] == 'c') {
        //     if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {
        //         printf("barrier %s[%d, %d] reached\n", barrier_name, barrier_x, barrier_y);
        //     }
        // } else {
        //     if (warp::laneid() == 0) {
        //         printf("barrier %s[%d, %d] reached\n", barrier_name, barrier_x, barrier_y);
        //     }
        // }
    }

    struct controller
    {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query)
        {
            int ret_order[] = {0, 1, 2};
            return ret_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s)
        {
            for (int i = 0; i < NUM_PIPELINE_STAGES; i++) {
                init_semaphore(inputs_arrived(s, i), 1);
                init_semaphore(inputs_finished(s, i), 1);
            }
            // output must wait for all consumer warps
            init_semaphore(out_arrived(s),  1);
            
            return 2 * NUM_PIPELINE_STAGES + 1;
        }
    };


    struct loader
    {
        static __device__ void run(const globals &g, state<config> &s)
        {
            parsed_instruction inst{s};
            uint32_t bitfield = 0xFFFF0000; // 1 for arrived, 0 for finished

            // f("loader waiting for page ready\n");}
            
            if (warp::laneid() == 0) {
                // printf("loader waiting for page ready across %d pages\n", NUM_PAGES);
                for (int i = 0; i < NUM_PAGES; i++) {
                    // printf("waiting for page ready: %d\n", i);
                    s.wait_page_ready(i);
                    // printf("page %d made it past wait_page_ready\n", i);
                    if (i > NUM_O_SHARED_PAGES) {
                        // printf("finishing page: %d\n", i);
                        s.finish_page(i, NUM_CONSUMER_WARPS);
                    }
                }
            }
            __syncwarp();
            // if (warp::laneid() == 0) {printf("loader made it past page ready\n");}

            for (int i = 0; i < inst.num_iters; i++) {
                // if (warp::laneid() == 0) {printf("loader iter %d of %d\n", i, inst.num_iters);}
                int load_uid = s.instruction()[LOAD_UIDS_OFFSET + i];

                // if (warp::laneid() == 0) {printf("loader waiting for barrier (iter %d, load_uid=%d)\n", i, load_uid);}
                wait_barrier(g, load_uid, inst.output_location.seq_idx, "load");
                // if (warp::laneid() == 0) {printf("loader made it past wait_barrier (iter %d, load_uid=%d)\n", i, load_uid);}

                int pipeline_stage = i % NUM_PIPELINE_STAGES;

                // if (warp::laneid() == 0) {printf("loader waiting for inputs_finished to equal %d (iter %d, load_uid=%d)\n", prototype::get_phasebit<1>(bitfield, pipeline_stage), i, load_uid);}
                wait(inputs_finished(s, pipeline_stage), prototype::get_phasebit<1>(bitfield, pipeline_stage));
                // if (warp::laneid() == 0) {printf("loader made it past inputs_finished (iter %d, load_uid=%d)\n", i, load_uid);}
                prototype::update_phasebit<1>(bitfield, pipeline_stage);
                
                if (warp::laneid() == 0) {
                    tma::expect_bytes(inputs_arrived(s, pipeline_stage), NUM_CONSUMER_WARPS * sizeof(o_st_type::dtype) * o_st_type::cols * o_st_type::rows + sizeof(l_sv_type::dtype) * l_sv_type::length);

                    for (int j = 0; j < NUM_CONSUMER_WARPS; j++) {
                        // if (warp::laneid() == 0) {printf("loader loading o_shared_tile %d of %d\n", j, NUM_CONSUMER_WARPS);}
                        tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                            get_o_shared_tile(s, pipeline_stage, j), 
                            g.O_scratch,
                            {load_uid, inst.output_location.seq_idx, 0, j},
                            inputs_arrived(s, pipeline_stage)
                        );
                        // if (warp::laneid() == 0) {printf("loader made it past load_async %d\n", j);}
                    }
                    // if (warp::laneid() == 0) {printf("loader loading l_shared_vec\n");}
                    tma::load_async<cache_policy::EVICT_FIRST>(
                        get_l_shared_vec(s, pipeline_stage),
                        g.Lvec_scratch,
                        {load_uid, inst.output_location.seq_idx, 0},
                        inputs_arrived(s, pipeline_stage)
                    );
                    // if (warp::laneid() == 0) {printf("loader made it past load_async l_shared_vec\n");}
                }
                // if (warp::laneid() == 0) {printf("loader made it past load_async\n");}
                __syncwarp();
                // if (warp::laneid() == 0) {printf("loader made it past syncwarp\n");}
            }
            // printf("loader finished\n");
            __syncwarp();
            // if (warp::laneid() == 0) {printf("LOADER DONE\n");}
        }
    };

    struct launcher
    { // launches mma's
        // launcher does nothing here, since this doesn't use tensor cores.
        static __device__ void run(const globals &g, state<config> &s)
        {
            // if (warp::laneid() == 0) {printf("launcher waiting for tensor ready\n");}
            s.wait_tensor_ready();
            if (laneid() == 0)
                arrive(s.tensor_finished, NUM_CONSUMER_WARPS);
            // if (warp::laneid() == 0) {printf("launcher finished\n");}
            __syncwarp();
            // if (warp::laneid() == 0) {printf("LAUNCHER DONE\n");}
        }
    };

    struct consumer
    {
        static __device__ void run(const globals &g, state<config> &s)
        {
            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer starting\n");}
            parsed_instruction inst{s};
            uint32_t bitfield = 0xFFFF0000; // 1 for arrived, 0 for finished

            o_st_type & o_shared_scratch = get_o_shared_scratch(s, group<NUM_CONSUMER_WARPS>::warpid());
            l_sv_type & l_shared_scratch = get_l_shared_scratch(s);

            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer waiting for barrier\n");}
            wait_barrier(g, inst.src_uid, inst.output_location.seq_idx, "consumer");
            group<NUM_CONSUMER_WARPS>::sync(12); // all warps must sync here.
            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer made it past barrier\n");}

            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer loading o_shared_scratch\n");}
            warp::load_async(
                o_shared_scratch, 
                g.O_scratch, 
                {inst.src_uid, inst.output_location.seq_idx, 0, group<NUM_CONSUMER_WARPS>::warpid()}
            );
            if(group<NUM_CONSUMER_WARPS>::warpid() == 0) { // only one warp needs to load the lvec, it is the same address for all warps.
                // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer loading l_shared_scratch\n");}
                warp::load_async(
                    l_shared_scratch,
                    g.Lvec_scratch, 
                    {inst.src_uid, inst.output_location.seq_idx, 0}
                );
            }
            warp::load_async_wait();
            __syncwarp();
            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer made it past load_async\n");}


            using o_rt_type = rt_fl<16, QVO_D/NUM_CONSUMER_WARPS>;
            using l_rv_type = col_vec<rt_fl<16, NUM_ROWS>>;
            o_rt_type o_state;
            l_rv_type l_state;

            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer loading o_state\n");}
            warp::load(o_state, o_shared_scratch);
            
            group<NUM_CONSUMER_WARPS>::sync(8); // needed for l_shared_scratch to be loaded
            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer made it past sync\n");}
            warp::load(l_state, l_shared_scratch);
            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer made it past load l_state\n");}
            l_rv_type lvec, max_lvec, sum_lvec;
            o_rt_type o_in;

            for (int i = 0; i < inst.num_iters; i++) {
                // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer iter %d of %d\n", i, inst.num_iters);}
                int pipeline_stage = i % NUM_PIPELINE_STAGES;

                // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer waiting for inputs_arrived\n");}
                wait(inputs_arrived(s, pipeline_stage), prototype::get_phasebit<0>(bitfield, pipeline_stage));
                // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer made it past inputs_arrived\n");}
                prototype::update_phasebit<0>(bitfield, pipeline_stage);

                // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer loading o_in\n");}
                warp::load(o_in, get_o_shared_tile(s, pipeline_stage, group<NUM_CONSUMER_WARPS>::warpid()));
                // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer made it past load o_in\n");}
                warp::load(lvec, get_l_shared_vec(s, pipeline_stage));
                // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer made it past load lvec\n");}
                
                __syncwarp();
                // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("pipeline_stage: %d\n", pipeline_stage);}
                if (group<NUM_CONSUMER_WARPS>::laneid() == 0) arrive(inputs_finished(s, pipeline_stage)); // done!

                // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer performing calculations\n");}

                // subtract off new max
                warp::max(max_lvec, l_state, lvec);
                warp::sub(l_state, l_state, max_lvec);
                warp::sub(lvec, lvec, max_lvec);

                // exp2
                warp::exp2(l_state, l_state);
                warp::exp2(lvec, lvec);
                // normalize the two lvecs
                warp::add(sum_lvec, l_state, lvec);
                warp::div(l_state, l_state, sum_lvec);
                warp::div(lvec, lvec, sum_lvec);

                // mul row
                warp::mul_row(o_state, o_state, l_state);
                warp::mul_row(o_in, o_in, lvec);

                // add o_in to o_state
                warp::add(o_state, o_state, o_in);

                // log2 and add max
                warp::log2(sum_lvec, sum_lvec);
                warp::add(l_state, sum_lvec, max_lvec);

                // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer made it past calculations\n");}
            }

            // store o_state and l_state

            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer storing o_state and l_state\n");}
            
            if (inst.output_location.batch_idx >= 0) {
                using o_rt_output_type = rt_fp8<16, QVO_D/NUM_CONSUMER_WARPS>;
                o_rt_output_type o_out_rt;

                warp::copy(o_out_rt, o_state);

                o_st_output_type & o_out = get_o_shared_output(s, group<NUM_CONSUMER_WARPS>::warpid());

                warp::store(o_out, o_out_rt);
            }
            else {
                warp::store(o_shared_scratch, o_state);

                if(group<NUM_CONSUMER_WARPS>::warpid() == 0) {
                    warp::store(l_shared_scratch, l_state);
                }
            }

            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer made it past store\n");}
            __syncwarp();
            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer made it past syncwarp\n");}
            if (group<NUM_CONSUMER_WARPS>::laneid() == 0) arrive(out_arrived(s));

            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer finishing pages\n");}
            for (int i = 0; i < NUM_O_SHARED_PAGES; i++) {
                // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) printf("consumer finishing page %d\n", i);
                if (group<NUM_CONSUMER_WARPS>::laneid() == 0 ) {
                    s.finish_page(i, NUM_CONSUMER_WARPS);
                }
            }
            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("consumer finished pages\n");}

            group<NUM_CONSUMER_WARPS>::sync(12);
            // if (group<NUM_CONSUMER_WARPS>::laneid() == 0) {printf("CONSUMER DONE\n");}
        }
    };


    struct storer
    {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};

            //if (warp::laneid() == 0) {printf("storer waiting for out_arrived\n");}
            wait(out_arrived(s), 0);
            //if (warp::laneid() == 0) {printf("storer made it past out_arrived\n");}

            if (inst.output_location.batch_idx >= 0) {
                if (warp::laneid() < NUM_CONSUMER_WARPS) {
                    //if (warp::laneid() == 0) {printf("storer storing o_shared_output\n");}
                    tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                        g.O, 
                        get_o_shared_output(s, laneid()), 
                        {
                            inst.output_location.batch_idx, 
                            inst.output_location.seq_idx, 
                            0,
                            laneid(),
                        }
                    );
                }
            }
            else {
                if (warp::laneid() < NUM_CONSUMER_WARPS) {
                    //if (warp::laneid() == 0) {printf("storer storing o_shared_scratch\n");}
                    tma::store_async<dim::ROW, cache_policy::EVICT_LAST>(
                        g.O_scratch,
                        get_o_shared_scratch(s, laneid()),
                        {
                            -inst.output_location.batch_idx-1, 
                            inst.output_location.seq_idx, 
                            0, 
                            laneid()
                        }
                    );
                } else if (warp::laneid() == NUM_CONSUMER_WARPS) {
                    // if (warp::laneid() == 0) {printf("storer storing l_shared_scratch\n");}
                    tma::store_async<cache_policy::EVICT_LAST>(
                        g.Lvec_scratch,
                        get_l_shared_scratch(s),
                        {
                            -inst.output_location.batch_idx-1, 
                            inst.output_location.seq_idx, 
                            0,
                        }
                    );
                }
            }
            // if (warp::laneid() == 0) {printf("storer waiting for store_async_wait\n");}
            tma::store_async_wait();
            // if (warp::laneid() == 0) {printf("storer made it past store_async_wait\n");}

            asm volatile("fence.sc.gpu;\n");
            __syncwarp();
            // if (warp::laneid() == 0) {printf("storer made it past syncwarp\n");}
            // Increment the semaphore for the next stage, if this is not the last one.
            if(inst.output_location.batch_idx < 0) {
                if(warp::laneid() == 0) {
                    g.Bar[{-inst.output_location.batch_idx-1, inst.output_location.seq_idx}] = g.tic;
                }
            }
            // if (warp::laneid() == 0) {printf("storer finishing pages\n");}
            if (warp::laneid() == 0) {
                s.finish_page(NUM_O_SHARED_PAGES, NUM_CONSUMER_WARPS);
            }
            // if (warp::laneid() == 0) {printf("storer finished pages\n");}
            
            __syncwarp();
            // if (warp::laneid() == 0) {printf("STORER DONE\n");}
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(mla_decode, m)
{
    m.doc() = "mla_decode python module";
    kittens::py::bind_kernel<kvm<config, globals, ReductionOp<config>>>(
        m, "mla_decode",
        &globals::instructions,
        &globals::Q,
        &globals::QV,
        &globals::K_cache,
        &globals::V_cache,
        &globals::Table,
        &globals::O,
        &globals::O_scratch,
        &globals::Lvec_scratch,
        &globals::Bar,
        &globals::softmax_scale,
        &globals::tic,
        &globals::timings
    );
    cudaGetLastError();
}
