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


static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 64;

static constexpr int PIPE_DEPTH = 1;
static constexpr int NUM_CONSUMERS = 4;


using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;

    using a_tile = st_bf<Mb, Kb>;
    using b_tile = st_bf<Nb, Kb>;
    using c_tile = st_hf<Mb, 64>;
    using c_tt_t = tt<float, Mb, Nb>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using c_gl = gl<half, 1, 1, -1, -1, c_tile>;

    using barriers = gl<bf16, 1, -1, 6, 32>; 

    instruction_layout instructions;
    timing_layout timings;
    b_gl B;
    a_gl A;
    c_gl C;
    barriers Bar;
    dim3 grid() { return dim3(148); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

struct ab_smem {
    using a_tile = globals::a_tile;
    using b_tile = globals::b_tile;

    a_tile a_smem[2];
    b_tile b_smem;

    __device__ inline ab_smem(state<config> &s) {
        if (laneid() == 0) {
            int a_pg = s.pid(get_a_page(s, 0)); // NEW
            s.wait_page_ready(a_pg);
            globals::a_gl &a_tile[2] = reinterpret_cast<globals::a_gl&>(s.pages[a_pg]);
        }

        if (laneid() == 1) {
            int b_pg = s.pid(get_b_page(s, 0)); // NEW
            s.wait_page_ready(b_pg);
            globals::b_gl &b_tile = reinterpret_cast<globals::b_gl&>(s.pages[b_pg]);
        }

        __syncthreads();
    }
};

template<typename config=config, int _OP_IDX=0> struct MatmulOp {
    static constexpr int opcode = 2;
    static constexpr int OP_IDX = _OP_IDX; 

    static constexpr int PIPE_DEPTH = 1;

    struct parsed_instruction {

        int layer, row_x, row_y;

        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            layer = instruction[1]; // in units of 1
            row_x = instruction[2];
            row_y = instruction[3];
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };
    static __device__ inline parsed_instruction parse_instruction(const globals &g, state<config> &s) {
        return parsed_instruction{s.instruction()[1], s.instruction()[2], s.instruction()[3]};
    }

    __device__ static inline semaphore &inputs_arrived(state<config> &s, int id) { return s.semaphores()[id]; }
    __device__ static inline semaphore &inputs_finished(state<config> &s, int id) { return s.semaphores()[id + PIPE_DEPTH]; }
    __device__ static inline semaphore &outputs_arrived(state<config> &s) { return s.semaphores()[2 * PIPE_DEPTH]; }

    __device__ static inline int get_a_page(state<config> &s, int offset) { return s.pid(offset); }
    __device__ static inline int get_b_page(state<config> &s, int offset) { return s.pid(offset + PIPE_DEPTH); }
    __device__ static inline int get_c_page(state<config> &s, int offset) { return s.pid(offset + 2 * PIPE_DEPTH); }
    
    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            int ret_order[] = {
                0, 1, 2, 3, 4, 5
            };
            return ret_order[query];
        }

        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for(int i = 0; i < PIPE_DEPTH; i++) {
                init_semaphore(inputs_arrived(s, i), 1); // Inputs arrived.
                init_semaphore(inputs_finished(s, i), 1); // Inputs finished.
            }
            init_semaphore(outputs_arrived(s), 16); // outputs arrived.
            return 2 * PIPE_DEPTH + 1;
        }
    };

    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            // Need to clear the first few elements of the scratch buffer, since we are using atomicAdd later.
            ((int*)s.scratch())[laneid()] = 0;
            warp::sync(); // done, now we can proceed to other things.

            int bitfield = 0xFFFF0000;
            ab_smem ab_smem{s};

            for (int idx = 0; idx < iters_per_task; idx++) {
                wait(inputs_finished(s, 0), get_phasebit<1>(bitfield, 0));
                warp::tma::expect(inputs_arrived(s, 0), ab_smem.a_smem[0], ab_smem.a_smem[1], ab_smem.b_smem);
                warp::tma::load_async(
                    ab_smem.a_smem[0],
                    g.A,
                    {(inst.row_x+0), idx}, 
                    inputs_arrived(s, 0)
                );
                warp::tma::load_async(
                    ab_smem.a_smem[1],
                    g.A,
                    {(inst.row_x+1), idx}, 
                    inputs_arrived(s, 0)
                );
                warp::tma::load_async(
                    ab_smem.b_smem, 
                    g.B,
                    {inst.row_y,     idx}, 
                    inputs_arrived(s, 0)
                );
            }

            wait(inputs_finished(s, 0), get_phasebit<1>(bitfield, 0));
            arrive(outputs_arrived(s));
        }
    };

    struct launcher { // launches mma's
        // launcher does nothing here, since this doesn't use tensor cores.
        static __device__ void run(const globals &g, state<config> &s) {
            tensor_allocator<1, 2> tm_alloc{};

            int warp_id = warpgroup::warpid();
            
            if (laneid() == 2) {
                s.wait_tensor_ready();
                globals::c_tt_t c_tt = tm_alloc.allocate<globals::c_tt_t>(warpgroup::warpid()*Nb);
            }

            ab_smem ab_smem{s};

            __syncthreads(); // not needed since ab_smem already syncs.

            wait(inputs_arrived(s, 0), 0);

            warp::mm_ABt( // NEW
                c_tt, ab_smem.a_smem[0][warp_id], ab_smem.b_smem, s.page_finished[pg]
            );

            for(int idx = 1; idx < iters_per_task; idx++) {
                wait(a_arrived(s), 0);

                int pg = s.pid(laneid());
                warp::mma_ABt( // NEW
                    c_tt, a_smem[0][warp_id], b_smem[0], s.page_finished[pg]
                );
            }

            wait(outputs_finished(s), 0);
            arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
        }
    };

    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            
            using c_tt_t = globals::c_tt_t;
            using c_tile = globals::c_tile;

            extern __shared__ int __shm[]; 
            tma_swizzle_allocator al((int*)&__shm[0]);
            int warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
            
            c_tile (&c_smem)                            = al.allocate<c_tile>();


            tensor_allocator<1, 2> tm_alloc{};

            warpgroup::increase_registers<224>();
            c_tt_t c_tt = tm_alloc.allocate<c_tt_t>(warpgroupid*Nb);
            for(int task_iter = 0; true; task_iter++) {
                int2 rowcol = { 0, 0 }; // get_task_idx(g, task_iter, true);
                if(rowcol.x == -1) break;
                kittens::wait(c_arrived, task_iter%2);
                rt_hf<Mb/4, c_tile::cols> d_reg[4];
                if(warpgroupid == 1) group<8>::sync(15);
                #pragma unroll
                for(int i = 0; i < Nb/c_tile::cols; i++) {
                    warpgroup::load_async(d_reg[i], c_tt.subtile<tt<float, 128, 64>>(0, 64*i));
                }
                tensor_load_wait();
                warpgroup::sync(warpgroupid);
                if(warpgroup::laneid() == 0) 
                    // arrive(outputs_finished[warpgroupid]); // Tensor memory for warpgroup 0 is now free.
                    arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);


                if(warpgroupid == 0) group<8>::sync(15);
                if(warpgroupid == 1) group<8>::sync(14);
                warpgroup::store(c_smem, d_reg[0]);
                warpgroup::sync(warpgroupid);
                if(warpgroup::warpid() == 0) warp::tma::store_async(g.C, c_smem, {rowcol.x, 4*rowcol.y+0});
                #pragma unroll
                for(int i = 1; i < Nb/c_tile::cols; i++) {
                    tma::store_async_read_wait();
                    warpgroup::sync(warpgroupid);
                    warpgroup::store(c_smem, d_reg[i]);
                    warpgroup::sync(warpgroupid);
                    if(warpgroup::warpid() == 0) warp::tma::store_async(g.C, c_smem, {rowcol.x, 4*rowcol.y+i});
                }
                tma::store_async_read_wait();
                if(warpgroupid == 0) group<8>::sync(14);
                group<8>::sync(15); // All consumers sync here.
            }

        }
    };

    struct storer {

        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {

            int group_id = warpgroup::groupid();
            int warp_id = warpgroup::warpid(); 

            parsed_instruction inst{s};
            if(laneid() == 0) {
                wait(c_arrived(s), 0);
                s.record(125);
                void *scratch = s.scratch();
                sv_bf<16> &output = *reinterpret_cast<sv_bf<16>*>(scratch);
                tma::store_async(g.C, output, {inst.start_col/16});
                tma::store_async_wait(); // not just read wait! full wait! must be visible in global!
                s.record(126);
            }
            warp::sync();
            asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc here but I don't think so.
            if(laneid() == 0) {
                if constexpr (OP_IDX == g.Bar.rows()-1) atomicAdd(&g.Bar[{inst.layer+1, 0, 0}], 1);
                else atomicAdd(&g.Bar[{inst.layer, OP_IDX+1, 0}], 1);
            }

        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(matmul, m) {
    m.doc() = "matmul python module";
    kittens::py::bind_kernel<kvm<config, globals, MatmulOp<config>>>(m, "matmul",
        &globals::instructions,
        &globals::timings,
        &globals::A,
        &globals::B,
        &globals::C,
        &globals::Bar
    );
}