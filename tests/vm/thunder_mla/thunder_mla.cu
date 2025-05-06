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
    static constexpr int PAGE_SIZE = 4*16384;
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
    timing_layout timings;

    using qrot_tile           = st_bf<64, QKRot_D>;
    using qvo_tile            = st_bf<64, QVO_D>;

    gl<bf16, -1, -1, -1, QKRot_D, qrot_tile> Q; // batch, new_tokens, q_heads, QKRot_D
    gl<bf16, -1, -1, -1, QVO_D, qvo_tile> QV; // batch, new_tokens, q_heads, QVO_D

    using kcache_tile         = st_bf<NUM_ROWS, QKRot_D>;
    using vcache_tile         = st_bf<NUM_ROWS, QVO_D>;

    gl<bf16, 1, -1, PAGE_SIZE, QKRot_D, kcache_tile> K_cache; // 1, num_pages, PAGE_SIZE, QKRot_D
    gl<bf16, 1, -1, PAGE_SIZE, QVO_D, vcache_tile> V_cache; // 1, num_pages, PAGE_SIZE, QVO_D

    gl<int, 1, 1, -1, -1> Table; // B, num_pages
    gl<bf16, -1, -1, -1, QVO_D, st_bf<16, QVO_Dd2>, st_bf<16, QVO_D/8>> O; // batch_size, new_tokens, q_heads, QVO_D
    
    gl<float, -1, -1, Q_HEADS, QVO_D, st_fl<16, QVO_D/config::NUM_CONSUMER_WARPS>, st_fl<16,256>> O_scratch; // num_instructions, new_tokens, q_heads, QVO_D
    gl<float,  1, -1, -1, Q_HEADS, sv_fl<16>> Lvec_scratch; // num_instructions, new_tokens, q_heads

    gl<int,    1,  1,  -1, -1> Bar; // 1, 1, num_instructions, new_tokens

    int tic;

    dim3 grid() { return dim3(148); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

struct location {
    int batch_idx; // batch_idx >=0, otherwise it's the negative index, minus one, into scratch
    int seq_idx;
};

template <typename config = config, int _OP_IDX = 0>
struct ReductionOp
{
    static constexpr int opcode = 2;
    static constexpr int OP_IDX = _OP_IDX; // Op index within the layer -- controls which barrier to listen to.
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
        return parsed_instruction{s.instruction()[1], s.instruction()[2], {s.instruction()[3], s.instruction()[4]}, s.instruction()[5]};
    }


    static constexpr int NUM_PIPELINE_STAGES = 4;
    static constexpr int NUM_STAGES_PER_PAGE = 2;
    static constexpr int INPUT_PAGES = 2;
    static constexpr int AUX_PAGES = 1;
    //  semaphores 
    __device__ static inline semaphore &inputs_arrived(state<config> &s, int i) { return s.semaphores()[i]; }
    __device__ static inline semaphore &inputs_finished(state<config> &s, int i) { return s.semaphores()[NUM_PIPELINE_STAGES + i]; }

    // getters
    using o_st_type = st_fl<16, QVO_D/8>;
    __device__ static inline o_st_type (&get_o_shared_tiles(state<config> &s))[NUM_PIPELINE_STAGES][NUM_CONSUMER_WARPS] {
        o_st_type[NUM_PIPELINE_STAGES][NUM_CONSUMER_WARPS] *os;
        
        o_st_type[NUM_STAGES_PER_PAGE][NUM_CONSUMER_WARPS] os1 = reinterpret_cast<o_st_type[NUM_STAGES_PER_PAGE][NUM_CONSUMER_WARPS]&>(
            s.pages[s.pid(0)]
        )[pipeline_stage%NUM_STAGES_PER_PAGE];

        for (int i = 0; i < NUM_STAGES_PER_PAGE; i++) os[i] = os1[i];

        st_o_tile_type[NUM_STAGES_PER_PAGE][NUM_CONSUMER_WARPS] os2 = reinterpret_cast<st_o_tile_type[NUM_STAGES_PER_PAGE][NUM_CONSUMER_WARPS]&>(
            s.pages[s.pid(1)]
        )[pipeline_stage%NUM_STAGES_PER_PAGE];

        for (int i = 0; i < NUM_STAGES_PER_PAGE; i++) os[i + NUM_STAGES_PER_PAGE] = os2[i];

        return os;
    }

    using l_sv_type = st_fl<16>;
    __device__ static inline l_sv_type (&get_l_shared_vecs(state<config> &s))[NUM_PIPELINE_STAGES] {
        return reinterpret_cast<l_sv_type[NUM_PIPELINE_STAGES]&>(s.pages[s.pid(2)]);
    }

    constexpr int L_VEC_SIZE = 4 * 16;
    __device__ static inline o_st_type (&get_o_shared_scratch(state<config> &s))[NUM_CONSUMER_WARPS] {
        return reinterpret_cast<o_st_type[NUM_CONSUMER_WARPS]&>(s.pages[s.pid(3)] + L_VEC_SIZE);
    }

    constexpr int O_SCRATCH_SIZE = 4 * 16 * QVO_D;
    constexpr int LVEC_SCRATCH_POS = L_VEC_SIZE + O_SCRATCH_SIZE;
    __device__ static inline l_sv_type &get_l_shared_scratch(state<config> &s) {
        return reinterpret_cast<l_sv_type&>(s.pages[s.pid(3)] + LVEC_SCRATCH_POS);
    }
    
    using o_st_output_type = st_bf<16, QVO_D/8>;
    __device__ static inline o_st_output_type (&get_o_shared_output(state<config> &s))[NUM_CONSUMER_WARPS] {
        return reinterpret_cast<o_st_output_type[NUM_CONSUMER_WARPS]&>(s.pages[s.pid(3)] + L_VEC_SIZE);
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
            // each weight page and the input page needs exactly 1 “ready” signal
            for (int i = 0; i < NUM_PIPELINE_STAGES; i++) {
                init_semaphore(inputs_arrived(s, i), 1);
                init_semaphore(inputs_finished(s, i), NUM_CONSUMER_WARPS);
            }
            // output must wait for all 4 consumer warps
            init_semaphore(out_arrived(s),  16);
            
            return 2 * Num;
        }
    };


    struct loader
    {
        static __device__ void run(const globals &g, state<config> &s)
        {
            parsed_instruction inst{s};
            int32_t bitfield = 0xFFFF0000; // 1 for arrived, 0 for finished

            o_st_type[NUM_PIPELINE_STAGES][NUM_CONSUMER_WARPS] o_shared_inputs = get_o_shared_tiles(s);
            l_sv_type[NUM_PIPELINE_STAGES] l_shared_vecs = get_l_shared_vecs(s);

            if(laneid() == 0) while(
                *(volatile int*)&
                globals.Bar[{
                    inst.load_uid, 
                    inst.output_location.seq_idx
                }] != args.globals.tic) __nanosleep(20);

            asm volatile("fence.sc.gpu;\n");
            __syncwarp();
            
            for (int i = 0; i < NUM_PAGES; i++) {
                s.wait_page_ready(i);
            }

            for (int i = 0; i < inst.num_iters; i++) {
                int pipeline_stage = i % NUM_PIPELINE_STAGES;

                o_st_type[NUM_CONSUMER_WARPS] o_shared_input = o_shared_inputs[pipeline_stage];
                l_sv_type l_shared_vec = l_shared_vecs[pipeline_stage];

                semaphore &inputs_arrived = get_inputs_arrived(s, pipeline_stage);

                wait(inputs_finished(s, pipeline_stage), prototype::get_phasebit<1>(bitfield, pipeline_stage));
                prototype::update_phasebit<1>(bitfield, pipeline_stage);
                
                tma::expect(inputs_arrived, o_shared_input, l_shared_vec);

                for (int j = 0; j < NUM_CONSUMER_WARPS; j++) {
                    tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                        o_shared_input[j], 
                        g.O_scratch,
                        {inst.load_uid, inst.output_location.seq_idx, 0, j},
                        inputs_arrived,
                    );
                }

                tma::load_async<cache_policy::EVICT_FIRST>(
                    l_shared_vec,
                    g.Lvec_scratch,
                    {inst.load_uid, inst.output_location.seq_idx, 0},
                    inputs_arrived,
                );
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
            int32_t bitfield = 0xFFFF0000; // 1 for arrived, 0 for finished

            o_st_type[NUM_PIPELINE_STAGES][NUM_CONSUMER_WARPS] o_shared_inputs = get_o_shared_tiles(s);
            l_sv_type[NUM_PIPELINE_STAGES] l_shared_vecs = get_l_shared_vecs(s);
            o_st_type[NUM_CONSUMER_WARPS] o_shared_scratch = get_o_shared_scratch(s);
            l_sv_type l_shared_scratch = get_l_shared_scratch(s);

            if(laneid() == 0) while(
                *(volatile int*)&
                globals.Bar[{
                    inst.src_uid, 
                    inst.output_location.seq_idx
                }] != args.globals.tic) __nanosleep(20);

            asm volatile("fence.sc.gpu;\n");
            group<NUM_CONSUMER_WARPS>::sync(11); // all warps must sync here.
            
            warp::load_async(
                o_shared_scratch[group<NUM_CONSUMER_WARPS>::warpid()], 
                g.O_scratch, 
                {inst.src_uid, inst.output_location.seq_idx, 0, group<NUM_CONSUMER_WARPS>::warpid()}
            );
            if(warpid() == 0) {
                warp::load_async(
                    l_shared_scratch, 
                    g.Lvec_scratch, 
                    {inst.src_uid, inst.output_location.seq_idx, 0},
                );
            }
            warp::load_async_wait();
            __syncwarp();

            using o_rt_type = rt_fl<16, QVO_D/NUM_CONSUMER_WARPS>;
            using l_rv_type = col_vec<rt_fl<16, NUM_ROWS>>;
            o_rt_type o_state;
            l_rv_type l_state;

            warp::load(o_state, o_shared_scratch[group<NUM_CONSUMER_WARPS>::warpid()]);
            
            group<NUM_CONSUMER_WARPS>::sync(11); // needed for l_shared_scratch to be loaded
            warp::load(l_state, l_shared_scratch);
            
            warp::arrive(out_arrived(s));  // let the storer know we’re done 

            l_rv_type lvec, max_lvec, sum_lvec;
            o_rt_type o_in;

            for (int i = 0; i < inst.num_iters; i++) {
                int pipeline_stage = i % NUM_PIPELINE_STAGES;

                wait(inputs_arrived(s, pipeline_stage), prototype::get_phasebit<0>(bitfield, pipeline_stage));
                prototype::update_phasebit<0>(bitfield, pipeline_stage);

                warp::load(o, o_shared_inputs[pipeline_stage][group<NUM_CONSUMER_WARPS>::warpid()]);
                warp::load(lvec, l_shared_vecs[pipeline_stage]);
                
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
                o_st_output_type o_out = get_o_shared_output(s);

                warp::store(o_out[group<NUM_CONSUMER_WARPS>::warpid()], o_state);
            }
            else {
                warp::store(o_shared_scratch[group<NUM_CONSUMER_WARPS>::warpid()], o_state);
                warp::store(l_shared_scratch, l_state);
            }
            __syncwarp();
            if (laneid() == 0) arrive(out_finished(s));

            for (int i = 0; i < NUM_PAGES; i++) {
                s.warp_finish_page(i, 1);
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
                o_st_output_type o_out = get_o_shared_output(s)[group<NUM_CONSUMER_WARPS>::warpid()];
                tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    g.O, 
                    o_out, 
                    {
                        inst.output_location.batch_idx, 
                        inst.output_location.seq_idx, 
                        0,
                        group<NUM_CONSUMER_WARPS>::warpid(),
                    },
                );
            }
            else {
                o_st_type o_shared_scratch = get_o_shared_scratch(s)[group<NUM_CONSUMER_WARPS>::warpid()];
                l_sv_type l_shared_scratch = get_l_shared_scratch(s);
                tma::store_async<dim::ROW, cache_policy::EVICT_LAST>(
                    g.O_scratch,
                    o_shared_scratch,
                    {
                        -inst.output_location.batch_idx-1, 
                        inst.output_location.seq_idx, 
                        0, 
                        group<NUM_CONSUMER_WARPS>::warpid()
                    },
                );
                if (group<NUM_CONSUMER_WARPS>::warpid() == 0) {
                    tma::store_async<cache_policy::EVICT_LAST>(
                        g.Lvec_scratch,
                        l_shared_scratch,
                        {
                            -inst.output_location.batch_idx-1, 
                            inst.output_location.seq_idx, 
                            0,
                        },
                    );
                }
            }
            tma::store_async_wait();

            asm volatile("fence.sc.gpu;\n");
            group<8>::sync(11);
            // Increment the semaphore for the next stage, if this is not the last one.
            if(inst.output_location.batch_idx < 0) {
                if(group<8>::laneid() == 0) {
                    globals.Bar[{-inst.output_location.batch_idx-1, inst.output_location.seq_idx}] = globals.tic;
                }
            }
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(thunder_mla, m)
{
    m.doc() = "thunder_mla python module";
    kittens::py::bind_kernel<kvm<config, globals, ReductionOp<config>>>(
        m, "reduction",
        &globals::instructions,
        &globals::timings,
        &globals::Q,
        &globals::QV,
        &globals::K_cache,
        &globals::V_cache,
        &globals::Table,
        &globals::O,
        &globals::O_scratch,
        &globals::Lvec_scratch,
        &globals::Bar,
        &globals::tic
    );
    cudaGetLastError();
}