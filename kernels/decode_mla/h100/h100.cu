#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

constexpr int CONSUMER_WARPGROUPS = (1); 
constexpr int PRODUCER_WARPGROUPS = (1); 
constexpr int NUM_WARPGROUPS      = (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS); 
constexpr int NUM_WORKERS         = (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 

using namespace kittens;
namespace cg = cooperative_groups;

template<int D> struct mla_decode_tile_dims {};
template<> struct mla_decode_tile_dims<576> {
    constexpr static int tile_width_dl = (512);
    constexpr static int tile_width_dr = (64); 
    constexpr static int tile_width_ml = (576);

    constexpr static int new_height   = (16);
    constexpr static int cache_height = (128);

    constexpr static int stages     = (1); 
    constexpr static int kv_blocks  = stages;
    constexpr static int kv_split   = cache_height * kv_blocks;
};

template<int D> struct decode_globals {
    using q1_tile = st_bf<mla_decode_tile_dims<576>::new_height, mla_decode_tile_dims<576>::tile_width_dl>;
    using q2_tile = st_bf<mla_decode_tile_dims<576>::new_height, mla_decode_tile_dims<576>::tile_width_dr>;
    using k1_tile = st_bf<mla_decode_tile_dims<576>::new_height, mla_decode_tile_dims<576>::tile_width_dl>;
    using k2_tile = st_bf<mla_decode_tile_dims<576>::new_height, mla_decode_tile_dims<576>::tile_width_dr>;

    using v_tile  = st_bf<mla_decode_tile_dims<576>::new_height, mla_decode_tile_dims<576>::tile_width_dl>;
    using o_tile  = st_bf<mla_decode_tile_dims<576>::new_height, mla_decode_tile_dims<576>::tile_width_dl>;

    using partial_tile = st_bf<mla_decode_tile_dims<576>::new_height, 64>; 
    using q_gl = gl<bf16, -1, -1, -1, -1, kittens::ducks::tma::descriptor<partial_tile, 1>>;
    using l_gl = gl<bf16, -1, -1, -1, -1, kittens::ducks::tma::descriptor<partial_tile, 1>>;
    using o_gl = gl<bf16, -1, -1, -1, -1, kittens::ducks::tma::descriptor<partial_tile, 1>>;

    q_gl q;
    l_gl l;
    o_gl o;

    float *lengths;
    float *table;

    const int new_tokens;
    const int page_size;
    const float softmax_scale;
};

// gl<bf16, -1, -1, -1, -1, tile>;
// gl<bf16, -1, -1, -1, -1, kittens::ducks::tma::descriptor<tile, 1>>;

template<int D>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void decode_kernel(const __grid_constant__ fwd_globals<D> g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = mla_decode_tile_dims<D>;

    using q1_tile = st_bf<mla_decode_tile_dims<D>::new_height,   mla_decode_tile_dims<D>::tile_width_dl>;
    using q2_tile = st_bf<mla_decode_tile_dims<D>::new_height,   mla_decode_tile_dims<D>::tile_width_dr>;
    using k1_tile = st_bf<mla_decode_tile_dims<D>::cache_height, mla_decode_tile_dims<D>::tile_width_dl>;
    using k2_tile = st_bf<mla_decode_tile_dims<D>::cache_height, mla_decode_tile_dims<D>::tile_width_dr>;
    using o_tile  = st_bf<mla_decode_tile_dims<D>::new_height,   mla_decode_tile_dims<D>::tile_width_dl>;

    using partial_tile = st_bf<mla_decode_tile_dims<D>::new_height, 64>; 

    q1_tile (&q1_smem)[1] = al.allocate<q1_tile, 1>();
    q2_tile (&q2_smem)[1] = al.allocate<q2_tile, 1>();

    k1_tile (&k1_smem)[K::stages] = al.allocate<k1_tile, K::stages>();
    k2_tile (&k2_smem)[K::stages] = al.allocate<k2_tile, K::stages>();

    o_tile  (&o_smem) = reinterpret_cast<o_tile(*)>(q1_smem);

    int kv_head_idx       = blockIdx.y;
    int global_kv_seq_idx = blockIdx.x;

    auto ZERO    = kittens::base_types::constants<bf16>::zero();
    auto NEG_INF = kittens::base_types::constants<bf16>::neg_infty();

    int batch_len = g.lengths[blockIdx.z];
    if (global_kv_seq_idx * g.page_size >= batch_len) {
        return;
    }

    int page_idx = g.table[(blockIdx.z * T) + blockIdx.x];

    __shared__ kittens::semaphore qsmem1_semaphore, qsmem2_semaphore; 
    __shared__ kittens::semaphore k1_smem_arrived[K::stages], k2_smem_arrived[K::stages]; 
    // compute_done[K::stages];
    if (threadIdx.x == 0) { 
        init_semaphore(qsmem1_semaphore, 0, 1);
        init_semaphore(qsmem2_semaphore, 0, 1);
        for(int j = 0; j < K::stages; j++) {
            init_semaphore(k1_smem_arrived[j], 0, 1);
            init_semaphore(k2_smem_arrived[j], 0, 1);
        }

        tma::expect_bytes(qsmem1_semaphore, sizeof(q1_smem)); 
        tma::expect_bytes(qsmem2_semaphore, sizeof(q2_smem)); 

        int i; 
        coord<partial_tile> q_tile_idx; 
        for (i = 0; i < (512/64); i++) {
            q_tile_idx = {blockIdx.z, 0, blockIdx.y, i};
            auto &q_subtile = subtile_inplace<16, 64>(q1_smem[0], {0, i});
            tma::load_async<1>(q_subtile, g.q, q_tile_idx, qsmem1_semaphore, NEG_INF);
        }
        q_tile_idx = {blockIdx.z, 0, blockIdx.y, i};
        tma::load_async<1>(q2_smem[0], g.q, q_tile_idx, qsmem2_semaphore, NEG_INF);

        for (int j = 0; j < K::stages; j++) {
            int i; 
            coord<partial_tile> kv_tile_idx;
            for (i = 0; i < (512/64); i++) {
                kv_tile_idx     = {page_idx, j, 0, i};
                auto &k_subtile = subtile_inplace<128, 64>(k1_smem[j], {0, i});
                tma::load_async<1>(k_subtile, g.k, kv_tile_idx, k1_smem_arrived[j], NEG_INF);
            }
            kv_tile_idx = {page_idx, j, 0, i};
            tma::load_async<1>(k2_smem[0], g.k, kv_tile_idx, k2_smem_arrived[j], NEG_INF);
        }
    }
    __syncthreads();

    int pipe_idx = seq_idx + K::stages - 1;
    
    if(warpgroupid == NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();
        
        // int kv_iters; 
        // if constexpr (is_causal) {
        //     kv_iters = (seq_idx *   (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) - 1 + (CONSUMER_WARPGROUPS * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)); 
        //     kv_iters = ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) == 0) ? (0) : ((kv_iters /   (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) - 1);
        // }
        // else { kv_iters = kv_blocks-2; }
        int kv_iters = 1; 

        // if(warpid == NUM_WORKERS-4) {
        //     for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
        //         coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx + 1, 0};
        //         tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
        //         tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));
        //         tma::load_async(k_smem[(kv_idx+1)%K::stages], g.k, kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
        //         tma::load_async(v_smem[(kv_idx+1)%K::stages], g.v, kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);
                
        //         wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
        //     }
        // }
    }
    else {
        warpgroup::increase_registers<160>();

        rt_fl<16, K::kv_height>  att_block;
        rt_bf<16, K::kv_height>  att_block_mma;
        rt_fl<16, K::tile_width> o_reg;
        
        col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;
        
        neg_infty(max_vec);
        zero(norm_vec);
        zero(o_reg);

        int kv_iters; 
        // if constexpr (is_causal) {
        //     kv_iters = (seq_idx * 4) - 1 + (CONSUMER_WARPGROUPS * 4);
        //     kv_iters = (kv_iters/8);
        // }
        // else { kv_iters = kv_blocks - 1; }
        kv_iters = 1; // ATTN_K/st_bf<mla_decode_tile_dims<D>::cache_height

        wait(qsmem_semaphore, 0);

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
        
            wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            warpgroup::mm_ABt(att_block,  q1_smem[0], k1_smem[(kv_idx)%K::stages]);
            warpgroup::mma_ABt(att_block, q2_smem[0], k2_smem[(kv_idx)%K::stages]);
            
            copy(max_vec_last_scaled, max_vec);
            if constexpr (D == 64) { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
            else                   { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }
            
            warpgroup::mma_async_wait();

            // if constexpr (is_causal) {
            //     const int q_blk = (seq_idx * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) + warpid; 
            //           int k_blk = (kv_idx  * (K::kv_height/kittens::TILE_ROW_DIM<bf16>)); 

            //     #pragma unroll
            //     for(int _ = 0; k_blk == (kv_iters-1)*(K::kv_height/kittens::TILE_ROW_DIM<bf16>) || k_blk == (kv_iters)*(K::kv_height/kittens::TILE_ROW_DIM<bf16>); k_blk+=10000) {
            //         #pragma unroll
            //         for (auto j = 0; j < (K::kv_height/kittens::TILE_ROW_DIM<bf16>); j++) {
            //             auto k_idx         = k_blk + j;
            //             auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(att_block.tiles[0][j]);

            //             if      (k_idx >  q_blk) { neg_infty  (attn_subtile); }
            //             else if (k_idx == q_blk) { make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
            //             __syncwarp();
            //         }
            //     }
            // }

            row_max(max_vec, att_block, max_vec);

            if constexpr (D == 64) { 
                mul(att_block,      att_block, 1.44269504089f*g.softmax_scale);
                mul(max_vec_scaled, max_vec,   1.44269504089f*g.softmax_scale);
            }
            else                   { 
                mul(att_block,      att_block,  1.44269504089f*g.softmax_scale); 
                mul(max_vec_scaled, max_vec,    1.44269504089f*g.softmax_scale);
            }
            
            // if constexpr (D == 64) { 
            //     mul(att_block,      att_block, 1.44269504089f*0.125f); 
            //     mul(max_vec_scaled, max_vec,   1.44269504089f*0.125f);
            // }
            // else                   { 
            //     mul(att_block,      att_block,  1.44269504089f*0.08838834764f); 
            //     mul(max_vec_scaled, max_vec,    1.44269504089f*0.08838834764f);
            // }

            sub_row(att_block, att_block, max_vec_scaled);
            exp2(att_block, att_block);
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last_scaled, max_vec_last_scaled);
            mul(norm_vec, norm_vec, max_vec_last_scaled);
            row_sum(norm_vec, att_block, norm_vec);
            add(att_block, att_block, 0.f);
            copy(att_block_mma, att_block);
            mul_row(o_reg, o_reg, max_vec_last_scaled);

            wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2); 

            warpgroup::mma_AB(o_reg, att_block_mma, k1_smem[(kv_idx)%K::stages]);
            warpgroup::mma_async_wait();

            if(warpgroup::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
        }

        div_row(o_reg, o_reg, norm_vec);
        warpgroup::store(o_smem[warpgroupid], o_reg);
        warpgroup::sync(warpgroupid+4);

        // if (warpid % 4 == 0) {
        //     coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + warpgroupid, 0};
        //     tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
        // }
        if (warpid % 4 == 0) {
            coord<o_tile> o_tile_idx = {blockIdx.z, 0, blockIdx.y, 0};
            tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
        }

        warpgroup::sync(warpgroupid+4);

        tma::store_async_wait();
    }
}

#include "harness.impl"