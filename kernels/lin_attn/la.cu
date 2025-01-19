#include "kittens.cuh"
#include <tuple>

#ifdef TORCH_COMPILE
#define TK_COMPILE_HEDGEHOG
#endif

#define NUM_WORKERS (8)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define NUM_WARPGROUPS (NUM_WORKERS/kittens::WARPGROUP_WARPS)

using namespace kittens;

#define CHUNK_SIZE 64
#define ATTN_D 128
#define ATTN_F 128
#define HALF_ATTN_F 64

struct la_globals { 
    // shapes    
    using q_tile = st_bf<CHUNK_SIZE, ATTN_F>;
    using k_tile = st_bf<CHUNK_SIZE, ATTN_F>;
    using v_tile = st_bf<CHUNK_SIZE, ATTN_D>;
    using o_tile = st_bf<CHUNK_SIZE, ATTN_D>;

    // global layouts
    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
 
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;

    // pointers
    q_gl q;
    k_gl k;
    v_gl v;

    o_gl o;

    float *slopes;
};


// should be launched with a grid of size (HEADS, BATCH) and blocks of 256 threads.
__global__ __launch_bounds__(NUM_THREADS, 1)
void linear_attn_smd (const __grid_constant__ la_globals g, int N)  
{
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int batch = blockIdx.y;
    const int head  = blockIdx.x;

    float slope = g.slopes[head];

    // smem
    using q_tile        = st_bf<CHUNK_SIZE, ATTN_F>;
    using k_tile        = st_bf<CHUNK_SIZE, ATTN_F>;
    using v_tile        = st_bf<CHUNK_SIZE, ATTN_D>;
    using o_tile        = st_bf<CHUNK_SIZE, ATTN_D>;
    using kv_state_tile = st_fl<ATTN_F, ATTN_D>;

    q_tile (&q_smem)[2] = al.allocate<q_tile, 2>();
    k_tile (&k_smem)[2] = al.allocate<k_tile, 2>();
    v_tile (&v_smem)[2] = al.allocate<v_tile, 2>();

    kv_state_tile (&kv_smem)                 = al.allocate<kv_state_tile>();
    o_tile         (&o_smem)[NUM_WARPGROUPS] = al.allocate<o_tile, NUM_WARPGROUPS>();

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/4;
    int blocks      = N / (q_tile::rows);

    int tic = 0, toc = 1;

    __shared__ semaphore qkv_semaphore;
    if (warpid == 0) {
        init_semaphore(qkv_semaphore, 0, 1);
        tma::expect_bytes(qkv_semaphore, 
            size_bytes<typeof(q_smem[0])> + 
            size_bytes<typeof(k_smem[0])> + 
            size_bytes<typeof(v_smem[0])> +
        );
        
        tma::load_async(q_smem[tic],  g.q,      {batch, head, 0, 0}, qkv_semaphore);
        tma::load_async(k_smem[ring_id+1], g.k, {batch, head, 0, 0}, qkv_semaphore);
        tma::load_async(v_smem[ring_id+1], g.v, {batch, head, 0, 0}, qkv_semaphore);
    }

    rt_fl<16, 128> local_kv;

    __syncthreads();

    for (int block = 0; block < blocks; block++, tic^=1, toc^=1) {

        wait(qkv_semaphore, tic);
        __syncthreads();

        if (warpid == 0 && block < blocks-1) {
            tma::expect_bytes(qkv_semaphore,
                size_bytes<typeof(q_smem[0])> + 
                size_bytes<typeof(k_smem[0])> + 
                size_bytes<typeof(v_smem[0])>
            );
            tma::load_async(q_smem[toc],           g.q, {batch, head, block+1, 0}, qkv_semaphore); 
            tma::load_async(k_smem[(ring_id+2)%3], g.k, {batch, head, block+1, 0}, qkv_semaphore); 
            tma::load_async(v_smem[(ring_id+2)%3], g.v, {batch, head, block+1, 0}, qkv_semaphore);
        }
        __syncthreads();

        if (warpgroupid == 0) {

        }
        else {

        }

    }

    tma::store_async_wait();
}

la_globals la_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    float *d_slopes,
    int ATTN_B, int ATTN_H, int ATTN_N
) {
    // global pointers. 
    using q_tile = st_bf<CHUNK_SIZE, ATTN_F>;
    using k_tile = st_bf<CHUNK_SIZE, ATTN_F>;
    using v_tile = st_bf<CHUNK_SIZE, ATTN_D>;
    using o_tile = st_bf<CHUNK_SIZE, ATTN_D>;
    
    using q_global = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_global = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_global = gl<bf16, -1, -1, -1, -1, v_tile>;
    using o_global = gl<bf16, -1, -1, -1, -1, o_tile>;

    using globals = la_globals;
    q_global q_arg{d_q, ATTN_B, ATTN_H, ATTN_N, ATTN_F};
    k_global k_arg{d_k, ATTN_B, ATTN_H, ATTN_N, ATTN_F};
    v_global v_arg{d_v, ATTN_B, ATTN_H, ATTN_N, ATTN_D};
    o_global o_arg{d_o, ATTN_B, ATTN_H, ATTN_N, ATTN_D};

    globals g{
        q_arg, k_arg, v_arg, 
        o_arg, d_slopes
    };
    return g;
}

#include "harness.impl"