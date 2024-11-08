#define TORCH_COMPILE
#include "kittens.cuh"

using namespace kittens;

constexpr int NUM_WORKERS = 4; // This kernel uses 4 worker warps per block, and 2 blocks per SM.
template<int D> constexpr size_t ROWS = 16*(128/D); // height of each worker tile (rows)
template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, ROWS<D>, D, L>;
template<int D, typename T=float> using attn_tile = rt<T, ROWS<D>, ROWS<D>>;
template<int D> using shared_tile = st_bf<ROWS<D>, D>;
template<int D> using global_layout = gl<bf16, -1, -1, -1, D>; // B, H, g.Qg.rows specified at runtime, D=64 known at compile time for this kernel
template<int D> struct globals { global_layout<D> Qg, KCacheg, VCacheg, Og, KNewg, VNewg; };

template<int D> __launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__global__ void attend_ker(
    const __grid_constant__ globals<D> g,   // Q, KCache, VCache, O, KNew, VNew
    int k_seqlen,                           // KCache sequence length
    int k_new_seqlen,                       // KNew sequence length
    bool causal                             // causal attention flag
) {
    auto ZERO = kittens::base_types::constants<bf16>::zero();
    using load_group = kittens::group<2>; // pairs of workers collaboratively load k, v tiles
    int loadid = load_group::groupid(), workerid = kittens::warpid(); // which worker am I?
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;
    const int batch = blockIdx.z, head  = blockIdx.y, q_seq = blockIdx.x * NUM_WORKERS + workerid;
    const int q_seq_next = (blockIdx.x + 1) * NUM_WORKERS;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    // K and V live in shared memory. Here, we instantiate three tiles for a 3-stage pipeline.
    shared_tile<D> (&k_smem)[LOAD_BLOCKS][3] = al.allocate<shared_tile<D>, LOAD_BLOCKS, 3>();
    shared_tile<D> (&v_smem)[LOAD_BLOCKS][3] = al.allocate<shared_tile<D>, LOAD_BLOCKS, 3>();
    // We also reuse this memory to improve coalescing of DRAM reads and writes.
    shared_tile<D> (&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);
    // Initialize all of the register tiles.
    qkvo_tile<D, bf16> q_reg, k_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile<D, bf16, col_l> v_reg; // V is column layout, as we use mma_AB.
    qkvo_tile<D, float> o_reg; // Output tile.
    attn_tile<D, float> att_block; // attention tile, in float. (We want to use float wherever possible.)
    attn_tile<D, bf16> att_block_mma; // bf16 attention tile for the second mma_AB. We cast right before that op.
    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec; // these are column vectors for the in-place softmax.
    // each warp loads its own Q tile of 16x64
    if (q_seq*ROWS<D> < g.Qg.rows) {
        load<shared_tile<D>, global_layout<D>, 2>(qo_smem[workerid], g.Qg, {batch, head, q_seq, 0}, ROWS<D>, ZERO);  // going through shared memory improves coalescing of dram reads.
        __syncwarp();
        load(q_reg, qo_smem[workerid]);
    }
    __syncthreads();
    // temperature adjustment. Pre-multiplying by lg2(e), too, so we can use exp2 later.
    if constexpr(D == 64) mul(q_reg, q_reg, __float2bfloat16(0.125f * 1.44269504089));
    else if constexpr(D == 128) mul(q_reg, q_reg, __float2bfloat16(0.08838834764f * 1.44269504089));
    // initialize flash attention L, M, and O registers.
    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_reg);
    // launch the load of the first k, v tiles
    // total number of blocks we want to load
    int kv_blocks = (k_seqlen + (LOAD_BLOCKS*ROWS<D>) - 1) / (LOAD_BLOCKS*ROWS<D>);
    int kv_blocks_total = (k_seqlen + k_new_seqlen + (LOAD_BLOCKS*ROWS<D>) - 1) / (LOAD_BLOCKS*ROWS<D>);
    int tic = 0;
    load_group::load_async<shared_tile<D>, global_layout<D>, 2>(k_smem[loadid][0], g.KCacheg, {batch, head, loadid, 0}, ROWS<D>, ZERO);
    load_group::load_async<shared_tile<D>, global_layout<D>, 2>(v_smem[loadid][0], g.VCacheg, {batch, head, loadid, 0}, ROWS<D>, ZERO);
    // iterate over k, v for these q's that have been loaded
    for(auto kv_idx = 0; kv_idx < kv_blocks_total; kv_idx++, tic=(tic+1)%3) {
        int cur_load_idx = kv_idx*LOAD_BLOCKS;

        int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;
        bool load_next = true;
        if (k_new_seqlen == 0) {
            // skip if we're out of K's or we're past the causal mask
            load_next = next_load_idx*ROWS<D> < k_seqlen && (!causal || next_load_idx <= q_seq_next);
        } else {
            // skip if we're out of KNew's or we're past the causal mask for KNew
            load_next = next_load_idx*ROWS<D> < k_seqlen + k_new_seqlen && (!causal || (kv_idx + 1 - kv_blocks) * LOAD_BLOCKS + loadid <= q_seq_next);
        }
        if(load_next && next_load_idx*ROWS<D> < k_seqlen) {
            // every two workers are working together to load the next tiles, then broadcast to all workers
            // we need to load the next times for all workers, and then skip selectively in the individual worker
            int next_tic = (tic+1)%3;
            load_group::load_async<shared_tile<D>, global_layout<D>, 2>(k_smem[loadid][next_tic], g.KCacheg, {batch, head, next_load_idx, 0}, ROWS<D>, ZERO);
            load_group::load_async<shared_tile<D>, global_layout<D>, 2>(v_smem[loadid][next_tic], g.VCacheg, {batch, head, next_load_idx, 0}, ROWS<D>, ZERO);
            load_async_wait<2>(); // next k, v can stay in flight.
        }
        else if (load_next && next_load_idx*ROWS<D> < k_seqlen + k_new_seqlen) {
            // load the next tiles from KNew and VNew
            int next_tic = (tic+1)%3;
            int kv_new_idx = (kv_idx - kv_blocks + 1) * LOAD_BLOCKS + loadid;
            load_group::load_async<shared_tile<D>, global_layout<D>, 2>(k_smem[loadid][next_tic], g.KNewg, {batch, head, kv_new_idx, 0}, ROWS<D>, ZERO);
            load_group::load_async<shared_tile<D>, global_layout<D>, 2>(v_smem[loadid][next_tic], g.VNewg, {batch, head, kv_new_idx, 0}, ROWS<D>, ZERO);
            load_async_wait<2>(); // next k, v can stay in flight.
        }
        else load_async_wait(); // all must arrive
        __syncthreads(); // Everyone's memory must be ready for the next stage.
        // now each warp goes through all of the subtiles, loads them, and then does the flash attention internal alg.
        #pragma unroll LOAD_BLOCKS
        for(int subtile = 0;
            subtile < LOAD_BLOCKS &&
            kv_idx * LOAD_BLOCKS * ROWS<D> + subtile * ROWS<D> < k_seqlen + k_new_seqlen;
            subtile++) {
            if (
                causal && (
                    (
                        k_new_seqlen == 0 &&
                        kv_idx * LOAD_BLOCKS * ROWS<D> + subtile * ROWS<D> > q_seq * ROWS<D> // we've passed the diagonal for this subtile and not using KNew
                    ) ||
                    (
                        k_new_seqlen > 0 &&
                        kv_idx >= kv_blocks &&
                        (kv_idx - kv_blocks) * LOAD_BLOCKS * ROWS<D> + subtile * ROWS<D> > q_seq * ROWS<D> // we've passed the diagonal for this subtile and using KNew
                    )
                )
            ){
                break;
            }

            load(k_reg, k_smem[subtile][tic]); // load k from shared into registers
            zero(att_block); // zero 16x16 attention tile
            mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T
            if (
                causal && (
                    (
                        k_new_seqlen == 0 && 
                        kv_idx * LOAD_BLOCKS * ROWS<D> + subtile * ROWS<D> == q_seq * ROWS<D>
                    ) // we are not using KNew, and Q_tile == K_tile on the diagonal
                    || (
                        k_new_seqlen > 0 &&
                        kv_idx >= kv_blocks &&
                        ((kv_idx - kv_blocks) * LOAD_BLOCKS * ROWS<D> + subtile * ROWS<D> == q_seq * ROWS<D>)
                    ) // we are using KNew, and Q_tile == K_new_tile, on the diagonal
                )
            ) {
                // mask out the upper triangle
                make_causal(att_block, att_block, kittens::base_types::constants<float>::neg_infty());
            }
            copy(max_vec_last,  max_vec);
            row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
            sub_row(att_block, att_block, max_vec); // subtract max from attention -- now all <=0
            exp2(att_block, att_block); // exponentiate the block in-place.
            sub(max_vec_last, max_vec_last, max_vec); // subtract new max from old max to find the new normalization.
            exp2(max_vec_last, max_vec_last); // exponentiate this vector -- this is what we need to normalize by.
            mul(norm_vec, norm_vec, max_vec_last); // and the norm vec is now normalized.
            row_sum(norm_vec, att_block, norm_vec); // accumulate the new attention block onto the now-rescaled norm_vec
            copy(att_block_mma, att_block); // convert to bf16 for mma_AB
            load(v_reg, v_smem[subtile][tic]); // load v from shared into registers.
            mul_row(o_reg, o_reg, max_vec_last); // normalize o_reg in advance of mma_AB'ing onto it
            mma_AB(o_reg, att_block_mma, v_reg, o_reg); // mfma onto o_reg with the local attention@V matmul.
        }
    }
    div_row(o_reg, o_reg, norm_vec);
    __syncthreads();
    if (q_seq*ROWS<D> < g.Qg.rows) { // write out o.
        store(qo_smem[workerid], o_reg); // going through shared memory improves coalescing of dram writes.
        __syncwarp();
        store<shared_tile<D>, global_layout<D>, 2>(g.Og, qo_smem[workerid], {batch, head, q_seq, 0});

        if (k_new_seqlen > 0) {
            __syncwarp();
            int kv_blocks_orig = (k_seqlen + ROWS<D> - 1) / ROWS<D>;
            // in-place update KCache with KNew
            load<shared_tile<D>, global_layout<D>, 2>(qo_smem[workerid], g.KNewg, {batch, head, q_seq, 0}, ROWS<D>, ZERO);  // going through shared memory improves coalescing of dram reads.
            __syncwarp();
            store<shared_tile<D>, global_layout<D>, 2>(g.KCacheg, qo_smem[workerid], {batch, head, kv_blocks_orig + q_seq, 0});

            __syncwarp();
            // in-place update VCache with VNew
            load<shared_tile<D>, global_layout<D>, 2>(qo_smem[workerid], g.VNewg, {batch, head, q_seq, 0}, ROWS<D>, ZERO);  // going through shared memory improves coalescing of dram reads.
            __syncwarp();
            store<shared_tile<D>, global_layout<D>, 2>(g.VCacheg, qo_smem[workerid], {batch, head, kv_blocks_orig + q_seq, 0});
        }
    }
}

#ifdef TORCH_COMPILE

#include "common/pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

/**
 * @brief Decode attention forward pass with a KV cache and optional in-place update to the KV cache.
 * 
 * @param q The new query. (batch, num_heads, seqlen_q, head_dim)
 * @param k_cache The existing key cache. (batch, num_heads, seqlen_k_max, head_dim)
 * @param v_cache The existing value cache. (batch, num_heads, seqlen_v, head_dim)
 * @param k_new_ The new keys to update k_cache in-place. (batch, num_heads, seqlen_q, head_dim)
 * @param v_new_ The new values to update v_cache in-place. (batch, num_heads, seqlen_q, head_dim)
 * @param causal Whether to use causal attention. If k_new_ and v_new_ are provided, causal mask is only applied against the new queries and keys. If they are not provided, k_seqlen must match seqlen_q, and the causal mask is applied against all the queries and keys.
 * @param k_seqlen The sequence length of the key cache.
 * @return The output of the attention operation. (batch, num_heads, seqlen_q, head_dim)
 */
torch::Tensor
attention_decode_forward(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    c10::optional<torch::Tensor> k_new_,
    c10::optional<torch::Tensor> v_new_,
    bool causal,
    int k_seqlen
)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);

    auto batch     = q.size(0);
    auto q_seq_len = q.size(2); 
    auto k_max_len = k_cache.size(2); 
    auto head_dim  = q.size(3); 
    auto qo_heads  = q.size(1);
    auto kv_heads  = k_cache.size(1);

    TORCH_CHECK(k_seqlen % 32 == 0, "K sequence length must be divisible by 32");

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k_cache.size(0) == batch, "K cache batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v_cache.size(0) == batch, "V cache batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q_seq_len % 32 == 0, "Q sequence length must be divisible by 32");
    TORCH_CHECK(k_max_len % 32 == 0, "K cache sequence length must be divisible by 32");

    TORCH_CHECK(v_cache.size(2) == k_max_len, "V cache sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k_cache.size(3) == head_dim, "K cache head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v_cache.size(3) == head_dim, "V cache head dimension - idx 3 - must match for all non-vector inputs");

    TORCH_CHECK(qo_heads >= kv_heads, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");
    TORCH_CHECK(q.size(1) == qo_heads, "QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k_cache.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v_cache.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");

    torch::Tensor k_new, v_new;
    auto k_new_seqlen = k_new_.has_value() ? k_new_.value().size(2) : 0;
    if (k_new_.has_value()) {
        assert(v_new_.has_value());
        k_new = k_new_.value();
        v_new = v_new_.value();
        CHECK_INPUT(k_new);
        CHECK_INPUT(v_new);
        TORCH_CHECK(k_new.size(0) == batch, "K new batch dimension - idx 0 - must match for all inputs");
        TORCH_CHECK(v_new.size(0) == batch, "V new batch dimension - idx 0 - must match for all inputs");
        TORCH_CHECK(k_new.size(1) == kv_heads, "K new heads - idx 1 - must match for all inputs");
        TORCH_CHECK(v_new.size(1) == kv_heads, "V new heads - idx 1 - must match for all inputs");
        TORCH_CHECK(k_new.size(2) == q_seq_len, "K new sequence length - idx 2 - must match for all inputs");
        TORCH_CHECK(v_new.size(2) == q_seq_len, "V new sequence length - idx 2 - must match for all inputs");
        TORCH_CHECK(k_new.size(3) == head_dim, "K new head dimension - idx 3 - must match for all inputs");
        TORCH_CHECK(v_new.size(3) == head_dim, "V new head dimension - idx 3 - must match for all inputs");
    }
    
    auto hr = qo_heads / kv_heads;

    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_cache_ptr = k_cache.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_cache_ptr = v_cache.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_new_ptr = k_new_.has_value() ? k_new.data_ptr<c10::BFloat16>() : nullptr;
    c10::BFloat16* v_new_ptr = v_new_.has_value() ? v_new.data_ptr<c10::BFloat16>() : nullptr;

    bf16*  d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k_cache = reinterpret_cast<bf16*>(k_cache_ptr);
    bf16*  d_v_cache = reinterpret_cast<bf16*>(v_cache_ptr);
    bf16*  d_k_new = k_new_ptr ? reinterpret_cast<bf16*>(k_new_ptr) : nullptr;
    bf16*  d_v_new = v_new_ptr ? reinterpret_cast<bf16*>(v_new_ptr) : nullptr;
    
    // for the returned outputs
    torch::Tensor o     = torch::empty({static_cast<const uint>(batch), 
                                        static_cast<const uint>(qo_heads), 
                                        static_cast<const uint>(q_seq_len), 
                                        static_cast<const uint>(head_dim)}, q.options());
    
    bf16*  o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16*  d_o   = reinterpret_cast<bf16*>(o_ptr);

    cudaDeviceSynchronize();

    unsigned long mem_size = (kittens::MAX_SHARED_MEMORY-1000) / 2; // have the flag tell us

    if (head_dim == 64) {
        global_layout<64> qg(d_q, batch, qo_heads, q_seq_len, nullptr);
        global_layout<64> kg(d_k_cache, batch, kv_heads, k_max_len, nullptr);
        global_layout<64> vg(d_v_cache, batch, kv_heads, k_max_len, nullptr);
        global_layout<64> og(d_o, batch, qo_heads, q_seq_len, nullptr);
        global_layout<64> kg_new(d_k_new, batch, kv_heads, q_seq_len, nullptr);
        global_layout<64> vg_new(d_v_new, batch, kv_heads, q_seq_len, nullptr);
        globals<64> g(qg, kg, vg, og, kg_new, vg_new);

        cudaFuncSetAttribute(
            attend_ker<64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        dim3 grid((q_seq_len + qkvo_tile<64>::rows*NUM_WORKERS - 1) / (qkvo_tile<64>::rows*NUM_WORKERS), qo_heads, batch);
        attend_ker<64><<<grid, (32*NUM_WORKERS), mem_size>>>(
            g,
            k_seqlen,
            k_new_seqlen,
            causal
        );
    }
    else if (head_dim == 128) {
        global_layout<128> qg(d_q, batch, qo_heads, q_seq_len, nullptr);
        global_layout<128> kg(d_k_cache, batch, kv_heads, k_max_len, nullptr);
        global_layout<128> vg(d_v_cache, batch, kv_heads, k_max_len, nullptr);
        global_layout<128> og(d_o, batch, qo_heads, q_seq_len, nullptr);
        global_layout<128> kg_new(d_k_new, batch, kv_heads, q_seq_len, nullptr);
        global_layout<128> vg_new(d_v_new, batch, kv_heads, q_seq_len, nullptr);
        globals<128> g(qg, kg, vg, og, kg_new, vg_new);

        cudaFuncSetAttribute(
            attend_ker<128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        dim3 grid((q_seq_len + qkvo_tile<128>::rows*NUM_WORKERS - 1) / (qkvo_tile<128>::rows*NUM_WORKERS), qo_heads, batch);
        attend_ker<128><<<grid, (32*NUM_WORKERS), mem_size>>>(
            g,
            k_seqlen,
            k_new_seqlen,
            causal
        );
    }
    else {
        TORCH_CHECK(false, "head_dim must be 64 or 128");
    }

    CHECK_CUDA_ERROR(cudaGetLastError());

    return o;
    cudaDeviceSynchronize();
}

#endif