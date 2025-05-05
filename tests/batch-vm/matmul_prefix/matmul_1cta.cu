#include "kittens.cuh"
#include "prototype.cuh"
#include <iostream>

constexpr int NUM_CONSUMERS = (2); 
constexpr int NUM_PRODUCERS = (1);

using namespace kittens;

static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 128;

struct matmul_globals {
    using a_tile = st_fp8e4m3<Mb,   Kb>;
    using b_tile = st_fp8e4m3<Nb, Kb>;
    using d_tile = st_hf<Mb, 64>;

    using a_gl = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<half,    1, 1, -1, -1, d_tile>;

    a_gl a;
    b_gl b;
    d_gl d;
};

constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
constexpr int NUM_THREADS = NUM_WORKERS * kittens::WARP_THREADS;

__device__ static inline int get_iters_per_task(const matmul_globals &g) {
    return g.a.cols() / Kb;
}
template<int SUPER_M=8> __device__ static inline int2 get_task_idx(const matmul_globals &g, int task_iter, bool is_consumer) {
    constexpr int CLUSTER_M = 4*Mb, CLUSTER_N = Nb;
    int cluster_x = clusterIdx().x, ctarank = cluster_ctarank();
    int task_id = task_iter * (gridDim.x/2) + cluster_x;
    int Rblocks = g.d.rows() / CLUSTER_M, Cblocks = g.d.cols() / CLUSTER_N;
    int super_rows = (Rblocks/SUPER_M)*SUPER_M,
        final_rows = Rblocks - super_rows,
        super_repeat = SUPER_M*Cblocks;
    if (task_id < super_rows * Cblocks) {
        return { 
            (SUPER_M*(task_id/super_repeat) + task_id%SUPER_M)*4 + ctarank*2 + is_consumer*(warpgroup::groupid()),
            is_consumer ? (task_id%super_repeat)/SUPER_M : (task_id%super_repeat)/SUPER_M
            // is_consumer ? (task_id%super_repeat)/SUPER_M : 2*((task_id%super_repeat)/SUPER_M) + ctarank
        };
    }
    else if (task_id < Rblocks*Cblocks) {
        int remainder_id = task_id - super_rows*Cblocks;
        return {
            (super_rows + remainder_id%final_rows)*4 + ctarank*2 + is_consumer*(warpgroup::groupid()),
            is_consumer ? remainder_id/final_rows : (remainder_id/final_rows)
            // is_consumer ? remainder_id/final_rows : 2*(remainder_id/final_rows) + ctarank
        };
    }
    else {
        return { -1, -1 };
    }
}

__global__ __cluster_dims__(2) __launch_bounds__(NUM_THREADS, 1)
void matmul(const __grid_constant__ matmul_globals g) {

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
    int iters_per_task = get_iters_per_task(g);

    constexpr int PIPE_DEPTH = 3;

    using a_tile = matmul_globals::a_tile;
    using b_tile = matmul_globals::b_tile;
    using d_tile = matmul_globals::d_tile;
    
    a_tile (&a_smem)[PIPE_DEPTH][NUM_CONSUMERS] = al.allocate<a_tile, PIPE_DEPTH, NUM_CONSUMERS>();
    b_tile (&b_smem)[PIPE_DEPTH]                = al.allocate<b_tile, PIPE_DEPTH>();
    d_tile (&d_smem)                            = al.allocate<d_tile>();

    tensor_allocator<1, 2> tm_alloc{};
    using d_tt_t = tt<float, Mb, Nb>;

    __shared__ kittens::semaphore inputs_arrived[PIPE_DEPTH], inputs_finished[PIPE_DEPTH], outputs_arrived, outputs_finished[NUM_CONSUMERS];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 0) { 
        for(int i = 0; i < PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 1); 
            init_semaphore(inputs_finished[i], 0, 2); 
        }
        init_semaphore(outputs_arrived, 0, 1);
        for(int i = 0; i < NUM_CONSUMERS; i++) {
            init_semaphore(outputs_finished[i], 0, 1);
        }
    }

    everyone::tma::cluster::sync();
    
    if(warpgroupid == NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();
        int ctarank = cluster_ctarank(); 
        if(warpgroup::warpid() == 3) {
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx(g, task_iter, false);
                if(rowcol.x == -1) {
                    for(int idx = 0; idx < (PIPE_DEPTH); idx++) {
                        wait(inputs_finished[input_ring], prototype::get_phasebit<1>(bitfield, input_ring));
                        input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring);
                    }
                    if(laneid() == 0) arrive(outputs_arrived);
                    break;
                }
                for (int idx = 0; idx < iters_per_task; idx++) {
                    wait(inputs_finished[input_ring], prototype::get_phasebit<1>(bitfield, input_ring));
                    prototype::update_phasebit<1>(bitfield, input_ring);
                    if(task_iter>0 && idx==PIPE_DEPTH-1 && laneid() == 0) arrive(outputs_arrived); 
                    warp::tma::expect(inputs_arrived[input_ring], a_smem[0][0], a_smem[0][1], b_smem[0]);
                    warp::tma::load_async(a_smem[input_ring][0], g.a, {(rowcol.x+0), idx}, inputs_arrived[input_ring]);
                    warp::tma::load_async(a_smem[input_ring][1], g.a, {(rowcol.x+1), idx}, inputs_arrived[input_ring]);
                    warp::tma::load_async(b_smem[input_ring],    g.b, { rowcol.y,    idx}, inputs_arrived[input_ring]);
                    input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring);
                }
            }
        }
        else if((warpgroup::warpid() == 0 || warpgroup::warpid() == 1)) { // launch the MMA's
            d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(warpgroup::warpid()*Nb);
            int input_ring = 0; // tracking which input block is being loaded
            for(int task_iter = 0; true; task_iter++) {
                int2 rowcol = get_task_idx(g, task_iter, false);
                if(rowcol.x == -1) break;
                wait(outputs_finished[warpgroup::warpid()], (task_iter+1)%2); // make sure tensor memory is ready to be written to.
                wait(inputs_arrived[input_ring], prototype::get_phasebit<0>(bitfield, input_ring));
                prototype::update_phasebit<0>(bitfield, input_ring);
                warp::mm_ABt(d_tt, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring);
                for(int idx = 1; idx < iters_per_task; idx++) {
                    wait(inputs_arrived[input_ring], prototype::get_phasebit<0>(bitfield, input_ring));
                    prototype::update_phasebit<0>(bitfield, input_ring);
                    warp::mma_ABt(d_tt, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                    input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring);
                }
            }
        }
    }
    else {
        warpgroup::increase_registers<224>();
        d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(warpgroupid*Nb);
        for(int task_iter = 0; true; task_iter++) {
            int2 rowcol = get_task_idx(g, task_iter, true);
            if(rowcol.x == -1) break;
            kittens::wait(outputs_arrived, task_iter%2);
            rt_hf<Mb/4, d_tile::cols> d_reg[4];
            if(warpgroupid == 1) group<8>::sync(15);
            #pragma unroll
            for(int i = 0; i < Nb/d_tile::cols; i++) {
                warpgroup::load_async(d_reg[i], d_tt.subtile<tt<float, 128, 64>>(0, 64*i));
            }
            tensor_load_wait();
            warpgroup::sync(warpgroupid);
            if(warpgroup::laneid() == 0) arrive(outputs_finished[warpgroupid]); // Tensor memory for warpgroup 0 is now free.
            if(warpgroupid == 0) group<8>::sync(15);
            if(warpgroupid == 1) group<8>::sync(14);
            warpgroup::store(d_smem, d_reg[0]);
            warpgroup::sync(warpgroupid);
            if(warpgroup::warpid() == 0) warp::tma::store_async(g.d, d_smem, {rowcol.x, 4*rowcol.y+0});
            #pragma unroll
            for(int i = 1; i < Nb/d_tile::cols; i++) {
                tma::store_async_read_wait();
                warpgroup::sync(warpgroupid);
                warpgroup::store(d_smem, d_reg[i]);
                warpgroup::sync(warpgroupid);
                if(warpgroup::warpid() == 0) warp::tma::store_async(g.d, d_smem, {rowcol.x, 4*rowcol.y+i});
            }
            tma::store_async_read_wait();
            if(warpgroupid == 0) group<8>::sync(14);
            group<8>::sync(15); // All consumers sync here.
        }
    }
    everyone::tma::cluster::sync();
}