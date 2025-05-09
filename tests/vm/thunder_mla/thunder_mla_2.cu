#define RED_TEXT "\033[31m"
#define GREEN_TEXT "\033[32m"
#define YELLOW_TEXT "\033[33m"
#define BLUE_TEXT "\033[34m"
#define MAGENTA_TEXT "\033[35m"
#define CYAN_TEXT "\033[36m"
#define WHITE_TEXT "\033[37m"
#define RESET_TEXT "\033[0m"

#include "kittens.cuh"
// #define KVM_DEBUG
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
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 4;

    // num bits required to represent num pipeline stages
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 2;

    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction.
    using instruction_t = int[INSTRUCTION_WIDTH];

    // Timing info
    static constexpr int TIMING_WIDTH = 128;
    using timing_t = int[TIMING_WIDTH];

    // How many semaphores are available for dynamic use?
    static constexpr int DYNAMIC_SEMAPHORES = 32;

    // One controller warp, one load warp, one store warp, and one mma warp.
    static constexpr int NUM_CONSUMER_WARPS = 16;
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
    static_assert(NUM_PAGES == 3, "NUM_PAGES must be 3");

    static constexpr bool TIMING_RECORD_ENABLED = true;

    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

    static constexpr int CONSUMER_REGISTERS = 104;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};

static constexpr int QKRot_D = 64, QVO_D = 512, QVO_Dd2 = QVO_D/2, NUM_ROWS = 32, PAGE_SIZE = 256, Q_HEADS = 16;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;

    instruction_layout instructions;

    using qrot_tile           = st_bf<64, QKRot_D>;
    using qvo_tile            = st_bf<64, QVO_D>;

    gl<bf16, -1, -1, -1, QKRot_D, qrot_tile> Q; // batch, new_tokens, q_heads, QKRot_D
    gl<bf16, -1, -1, -1, QVO_D, qvo_tile> QV; // batch, new_tokens, q_heads, QVO_D

    using kcache_tile         = st_bf<NUM_ROWS, QKRot_D>;
    using vcache_tile         = st_bf<NUM_ROWS, QVO_D>;

    gl<bf16, 1, -1, PAGE_SIZE, QKRot_D, kcache_tile> K_cache; // 1, num_pages, PAGE_SIZE, QKRot_D
    gl<bf16, 1, -1, PAGE_SIZE, QVO_D, vcache_tile> V_cache; // 1, num_pages, PAGE_SIZE, QVO_D

    gl<int, 1, 1, -1, -1> Table; // B, num_pages
    gl<bf16, -1, -1, -1, QVO_D, st_bf<16, QVO_Dd2>, st_bf<16, QVO_D/config::NUM_CONSUMER_WARPS>> O; // batch_size, new_tokens, q_heads, QVO_D
    
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
            num_iters = get_num_iters(start_pos, end_pos, length);
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

    static constexpr int NUM_PIPELINE_STAGES = 3;
    static constexpr int NUM_CONSUMER_WARPS = config::NUM_CONSUMER_WARPS;
    static constexpr int NUM_PAGES = config::MAX_SHARED_MEMORY / config::PAGE_SIZE;

    // Pipelined objects
    using kcache_st_type         = st_bf<NUM_ROWS, QKRot_D>;
    static constexpr int K_CACHE_TILE_SIZE = NUM_PIPELINE_STAGES * sizeof(kcache_st_type::dtype) * kcache_st_type::rows * kcache_st_type::cols;

    using vcache_st_type         = st_bf<NUM_ROWS, QVO_D>;
    static constexpr int V_CACHE_TILE_SIZE = NUM_PIPELINE_STAGES * sizeof(vcache_st_type::dtype) * vcache_st_type::rows * vcache_st_type::cols;

    // Non-pipelined objects
    using qrot_st_type           = st_bf<64, QKRot_D>;
    static constexpr int QROT_TILE_SIZE = sizeof(qrot_st_type::dtype) * qrot_st_type::rows * qrot_st_type::cols;

    using qvo_st_type            = st_bf<64, QVO_D>;
    static constexpr int QVO_TILE_SIZE = sizeof(qvo_st_type::dtype) * qvo_st_type::rows * qvo_st_type::cols;

    using attn_st_type           = st_bf<64, QVO_Dd2>;
    static constexpr int ATTN_TILE_SIZE = sizeof(attn_st_type::dtype) * attn_st_type::rows * attn_st_type::cols;

    using l_sv_type = sv_fl<64>;
    static constexpr int L_VEC_SIZE = sizeof(l_sv_type::dtype) * l_sv_type::length;

    using o_out_st_type = st_bf<16, QVO_Dd2>;
    static constexpr int O_OUT_TILE_SIZE = 4 * 2 * sizeof(o_out_st_type::dtype) * o_out_st_type::rows * o_out_st_type::cols;

    using o_st_type = st_bf<16, QVO_D/NUM_CONSUMER_WARPS>;
    static constexpr int O_TILE_SIZE = sizeof(o_st_type::dtype) * o_st_type::rows * o_st_type::cols;

    static constexpr int O_SHARED_SCRATCH_SIZE = O_TILE_SIZE * NUM_CONSUMER_WARPS;
    static constexpr int L_SHARED_SCRATCH_SIZE = sizeof(l_sv_type::dtype) * l_sv_type::length;
    using o_st_output_type = st_bf<16, QVO_D/NUM_CONSUMER_WARPS>;
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
        // if(laneid() == 0) while(
        //     *(volatile int*)&
        //     g.Bar[{
        //         barrier_x, 
        //         barrier_y
        //     }] != g.tic) {
        //     __nanosleep(20);
        //     if(clock64() - clock > 1000000) {
        //         printf(RED_TEXT "Barrier %s[%d, %d] not reached after 1 second\n" RESET_TEXT, barrier_name, barrier_x, barrier_y);
        //         asm volatile("trap;");
        //     }
        // }

        asm volatile("fence.sc.gpu;\n");
        __syncwarp();
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
                init_semaphore(inputs_finished(s, i), NUM_CONSUMER_WARPS);
            }
            // output must wait for all consumer warps
            init_semaphore(out_arrived(s),  NUM_CONSUMER_WARPS);
            
            return 2 * NUM_PIPELINE_STAGES + 1;
        }
    };


    struct loader
    {
        static __device__ void run(const globals &g, state<config> &s)
        {
            parsed_instruction inst{s};
            uint32_t bitfield = 0xFFFF0000; // 1 for arrived, 0 for finished
            
            if (laneid() <= NUM_O_SHARED_PAGES) {
                s.wait_page_ready(laneid());
                if (laneid() > NUM_O_SHARED_PAGES) {
                    s.finish_page(laneid(), 1);
                }
            }
            __syncwarp();

            for (int i = 0; i < inst.num_iters; i++) {
                int load_uid = s.instruction()[LOAD_UIDS_OFFSET + i];

                wait_barrier(g, load_uid, inst.output_location.seq_idx, "load");

                int pipeline_stage = i % NUM_PIPELINE_STAGES;

                wait(inputs_finished(s, pipeline_stage), prototype::get_phasebit<1>(bitfield, pipeline_stage));
                prototype::update_phasebit<1>(bitfield, pipeline_stage);
                
                if (warp::laneid() == 0) {
                    tma::expect_bytes(inputs_arrived(s, pipeline_stage), NUM_CONSUMER_WARPS * sizeof(o_st_type::dtype) * o_st_type::cols * o_st_type::rows + sizeof(l_sv_type::dtype) * l_sv_type::length);

                    for (int j = 0; j < NUM_CONSUMER_WARPS; j++) {
                        tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                            get_o_shared_tile(s, pipeline_stage, j), 
                            g.O_scratch,
                            {load_uid, inst.output_location.seq_idx, 0, j},
                            inputs_arrived(s, pipeline_stage)
                        );
                    }

                    tma::load_async<cache_policy::EVICT_FIRST>(
                        get_l_shared_vec(s, pipeline_stage),
                        g.Lvec_scratch,
                        {load_uid, inst.output_location.seq_idx, 0},
                        inputs_arrived(s, pipeline_stage)
                    );
                }
            }
        }
    };

    struct launcher
    { // launches mma's
        // launcher does nothing here, since this doesn't use tensor cores.
        static __device__ void run(const globals &g, state<config> &s)
        {
            s.wait_tensor_ready();
            if (laneid() == 0)
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
        }
    };

    struct consumer
    {
        static __device__ void run(const globals &g, state<config> &s)
        {
            parsed_instruction inst{s};
            uint32_t bitfield = 0xFFFF0000; // 1 for arrived, 0 for finished

            o_st_type & o_shared_scratch = get_o_shared_scratch(s, group<NUM_CONSUMER_WARPS>::warpid());
            l_sv_type & l_shared_scratch = get_l_shared_scratch(s);

            wait_barrier(g, inst.src_uid, inst.output_location.seq_idx, "consumer");
            group<NUM_CONSUMER_WARPS>::sync(12); // all warps must sync here.

            warp::load_async(
                o_shared_scratch, 
                g.O_scratch, 
                {inst.src_uid, inst.output_location.seq_idx, 0, group<NUM_CONSUMER_WARPS>::warpid()}
            );
            if(group<NUM_CONSUMER_WARPS>::warpid() == 0) { // only one warp needs to load the lvec, it is the same address for all warps.
                warp::load_async(
                    l_shared_scratch, 
                    g.Lvec_scratch, 
                    {inst.src_uid, inst.output_location.seq_idx, 0}
                );
            }

            warp::load_async_wait();
            __syncwarp();


            using o_rt_type = rt_fl<16, QVO_D/NUM_CONSUMER_WARPS>;
            using l_rv_type = col_vec<rt_fl<16, NUM_ROWS>>;
            o_rt_type o_state;
            l_rv_type l_state;

            warp::load(o_state, o_shared_scratch);
            
            group<NUM_CONSUMER_WARPS>::sync(8); // needed for l_shared_scratch to be loaded
            warp::load(l_state, l_shared_scratch);

            l_rv_type lvec, max_lvec, sum_lvec;
            o_rt_type o_in;

            for (int i = 0; i < inst.num_iters; i++) {
                int pipeline_stage = i % NUM_PIPELINE_STAGES;

                wait(inputs_arrived(s, pipeline_stage), prototype::get_phasebit<0>(bitfield, pipeline_stage));
                prototype::update_phasebit<0>(bitfield, pipeline_stage);

                warp::load(o_in, get_o_shared_tile(s, pipeline_stage, group<NUM_CONSUMER_WARPS>::warpid()));
                warp::load(lvec, get_l_shared_vec(s, pipeline_stage));
                
                __syncwarp();
                if(laneid() == 0) arrive(inputs_finished(s, pipeline_stage)); // done!

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
            }

            // store o_state and l_state
            
            if (inst.output_location.batch_idx >= 0) {
                o_st_output_type & o_out = get_o_shared_output(s, group<NUM_CONSUMER_WARPS>::warpid());

                warp::store(o_out, o_state);
            }
            else {
                warp::store(o_shared_scratch, o_state);

                if(group<NUM_CONSUMER_WARPS>::warpid() == 0) {
                    warp::store(l_shared_scratch, l_state);
                }
            }
            __syncwarp();
            if (laneid() == 0) arrive(out_arrived(s));

            for (int i = 0; i < NUM_O_SHARED_PAGES; i++) {
                if (warp::laneid() == 0) {
                    s.finish_page(i, 1);
                }
            }
        }
    };


    struct storer
    {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};

            wait(out_arrived(s), 0);

            if (inst.output_location.batch_idx >= 0) {
                if (laneid() < NUM_CONSUMER_WARPS) {

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
                if (laneid() < NUM_CONSUMER_WARPS) {
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
                } else if (laneid() == NUM_CONSUMER_WARPS) {
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
            tma::store_async_wait();

            asm volatile("fence.sc.gpu;\n");
            __syncwarp();
            // Increment the semaphore for the next stage, if this is not the last one.
            if(inst.output_location.batch_idx < 0) {
                if(group<8>::laneid() == 0) {
                    g.Bar[{-inst.output_location.batch_idx-1, inst.output_location.seq_idx}] = g.tic;
                }
            }
            
            if (laneid() == 0) {
                s.finish_page(NUM_O_SHARED_PAGES, NUM_CONSUMER_WARPS);
            }
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
