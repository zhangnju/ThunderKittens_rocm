from __future__ import annotations

import torch

from typing import Tuple

from triton import cdiv, autotune, jit, Config, language as tl, next_power_of_2     # noqa.

from text_generation_server.core.kernel import BaseKernel, Dynamic

__all__ = ["mla_buffers", "flash_mla_decode", "FlashMLADecode"]


@jit
def mla_decode_partials_kernel(
    # Data Pointer(s).
    q_ptr, cache_ptr, table_ptr, lengths_ptr, out_ptr, lse_ptr,
    # Stride(s).
    stride_ql, stride_cn, stride_ol, stride_oh, stride_sl,
    # Runtime Dimension(s).
    softmax_scale: tl.constexpr,
    timesteps:     tl.constexpr,
    query_heads:   tl.constexpr,
    latent_dim:    tl.constexpr,
    rope_dim:      tl.constexpr,
    page_size:     tl.constexpr,
    page_splits:   tl.constexpr,
    split_width:   tl.constexpr,
    max_pages:     tl.constexpr,
    num_splits:    tl.constexpr,
    # Block Size Constant(s),
    block_h: tl.constexpr,
    block_z: tl.constexpr,
    block_r: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    # """
    # Shape(s):

    #     • Q -> [ B, R + 1, Hq, Z ].

    #     • Latent Cache -> [ N, K, 1, Z ].

    #     • Block Table -> [ B, T ].

    #     • Cache Lengths -> [ B ].

    #     • Out -> [ B, R + 1, Hq, S, Zc ].

    #     • LSE -> [ B, R + 1, Hq, S ].

    # Legend(s):

    #     • R -> Lookahead

    #     • Hq -> 128

    #     • Z -> 576 -> [ Zc ; Zr ]

    #     • Zc -> 512

    #     • Zr -> 64

    #     • K -> 256 (Page Size)

    #     • S -> P x T | P -> Splits Per Page

    # Tiling Dimension(s):

    #     • L -> B x (R + 1)

    #     • CEIL_DIV( Hq, BLOCK_H )

    #     • S -> Splits

    # """
    # Latent Cache & Block Table Tile(s).
    split_id = tl.program_id(axis=1)

    # Query Tile(s).
    sequence_id   = tl.program_id(axis=0)
    head_block_id = tl.program_id(axis=2)

    # Batch-Step Indices.
    batch_id = sequence_id // timesteps
    step_id  = sequence_id % timesteps

    # Sequence Length Bound(s).
    key_length  = tl.load(lengths_ptr + batch_id) + step_id + 1
    total_pages = tl.cdiv(key_length, page_size)

    # Paged Block Specific(s).
    relative_page_id = split_id // page_splits

    # Ensure Valid Block(s).
    if relative_page_id < total_pages:

        # Head Dimension(s).
        joint_dim = latent_dim + rope_dim

        # Head Block Specific(s).
        head_ids  = ((head_block_id * block_h) + tl.arange(0, block_h))[:, None]
        head_mask = head_ids < query_heads

        latent_offsets = tl.arange(0, block_z)[None, :]
        rope_offsets   = latent_dim + tl.arange(0, block_r)[None, :]

        latent_mask = latent_offsets < latent_dim
        rope_mask   = rope_offsets < joint_dim

        # Load Query Block(s).
        query_shift = (sequence_id * stride_ql) + (head_ids * joint_dim)

        ql_mask = head_mask & latent_mask
        qr_mask = head_mask & rope_mask

        ql_block = tl.load(q_ptr + (query_shift + latent_offsets), mask=ql_mask, other=0.0)
        qr_block = tl.load(q_ptr + (query_shift + rope_offsets), mask=qr_mask, other=0.0)

        # Latent Block Specific(s).
        page_id = tl.load(table_ptr + (batch_id * max_pages) + relative_page_id)

        kv_offsets = tl.arange(0, block_k)[:, None]
        latent_shift = page_id * stride_cn

        # Relative Index & Absolute Prefix Length.
        split_start   = (split_id % page_splits) * split_width
        prefix_length = relative_page_id * page_size

        # Partial Accumulation(s).
        partials = tl.zeros((block_h, block_z), dtype=tl.float32)

        running_max = tl.full((block_h, 1), -float("inf"), dtype=tl.float32)
        running_sum = tl.zeros((block_h, 1), dtype=tl.float32)

        for k_shift in range(split_start, split_start + split_width, block_k):

            # Latent KV Split Block, Scores Accumulation & Mask.
            k_offsets = k_shift + kv_offsets

            kv_mask = (prefix_length + k_offsets) < key_length
            kv_shift = latent_shift + (k_offsets * joint_dim)

            scores     = tl.zeros((block_h, block_k), dtype=tl.float32)
            score_mask = head_mask & tl.trans(kv_mask)

            # Associative Processing: RoPE Dims.
            block_ptrs = cache_ptr + (kv_shift + rope_offsets)
            kr_mask    = kv_mask & rope_mask

            kr_block = tl.load(block_ptrs, mask=kr_mask, other=0.0)

            scores = tl.dot(qr_block, tl.trans(kr_block), acc=scores)

            # Associative Processing: Latent Dims.
            block_ptrs = cache_ptr + (kv_shift + latent_offsets)
            kl_mask = kv_mask & latent_mask

            kl_block = tl.load(block_ptrs, mask=kl_mask, other=0.0)

            scores = tl.dot(ql_block, tl.trans(kl_block), acc=scores)

            # Apply Softmax Scale & Bounding Mask.
            scores *= softmax_scale
            scores = tl.where(score_mask, scores, -float("inf"))

            # Update Running Tallies with Stable Exponentiation(s).
            new_max = tl.maximum(tl.max(scores, axis=1, keep_dims=True), running_max)
            rescale = tl.exp(running_max - new_max)

            stabilized = tl.exp(scores - new_max)
            running_max = new_max

            running_sum *= rescale
            running_sum += tl.sum(stabilized, axis=1, keep_dims=True)

            # Partials Rescaling & Accumulation.
            partials *= rescale
            partials = tl.dot(stabilized, kl_block.to(tl.float32), acc=partials)

        # Finalize & Store Normalized Partials & LSE.
        partials /= running_sum

        out_shifts = (
            (sequence_id * stride_ol)
            + (head_ids * stride_oh)
            + (split_id * latent_dim)
            + latent_offsets
        )

        lse_block = tl.log(running_sum) + running_max
        lse_shifts = (sequence_id * stride_sl) + (head_ids * num_splits) + split_id

        tl.store(lse_ptr + lse_shifts, lse_block, mask=head_mask)
        tl.store(out_ptr + out_shifts, partials, mask=ql_mask)


@jit
def mla_decode_reduction_kernel(
    # Data Pointer(s).
    partials_ptr, lse_ptr, lengths_ptr, out_ptr,
    # Stride(s).
    stride_pl, stride_ph, stride_sl, stride_ol,
    # Runtime Dimension(s).
    timesteps:   tl.constexpr,
    latent_dim:  tl.constexpr,
    num_splits:  tl.constexpr,
    split_width: tl.constexpr,
    # Block Size Constant(s),
    block_z: tl.constexpr,
    block_s: tl.constexpr,
) -> None:
    # """
    # Shape(s):

    #     • Partials -> [ B, R + 1, Hq, S, Zc ].

    #     • LSE -> [ B, R + 1, Hq, S ].

    #     • Out -> [ B, R + 1, Hq, Zc ].

    # Legend(s):

    #     • R -> Lookahead

    #     • Hq -> 128

    #     • Zc -> 512

    #     • S -> Total Splits

    # Tiling Dimension(s):

    #     • L -> B x (R + 1)

    #     • Hq

    # """
    # Indexing Tile(s).
    sequence_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    # Batch-Step Indices.
    batch_id = sequence_id // timesteps
    step_id = sequence_id % timesteps

    # Sequence Length & Split Bound(s).
    key_length = tl.load(lengths_ptr + batch_id) + step_id + 1
    valid_splits = tl.cdiv(key_length, split_width)

    block_offsets = tl.arange(0, block_s)[:, None]

    # Head Dimension Specific(s).
    latent_offsets = tl.arange(0, block_z)[None, :]
    latent_mask = latent_offsets < latent_dim

    # Accumulator(s).
    reduced = tl.zeros((block_z,), dtype=tl.float32)

    running_max = tl.full((1,), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((1,), dtype=tl.float32)

    for k in range(0, tl.cdiv(num_splits, block_s)):

        # Sequence Split Index.
        split_start = (k * block_s)

        if split_start < valid_splits:

            # Split Block.
            split_ids = split_start + block_offsets
            split_mask = split_ids < valid_splits

            # LSE Block.
            lse_shift = (sequence_id * stride_sl) + (head_id * num_splits) + split_ids
            lse_block = tl.load(lse_ptr + lse_shift, mask=split_mask, other=-float("inf"))

            # Partials Block.
            partials_shift = (
                (sequence_id * stride_pl)
                + (head_id * stride_ph)
                + (split_ids * latent_dim)
                + latent_offsets
            )

            partials_mask = split_mask & latent_mask
            partials_block = tl.load(partials_ptr + partials_shift, mask=partials_mask, other=0.0)

            # Update Running Tallies and Partial Reduction(s).
            new_max = tl.maximum(tl.max(lse_block, axis=0), running_max)
            rescale = tl.exp(running_max - new_max)

            row_scale = tl.exp(lse_block - new_max)
            running_max = new_max

            reduced *= rescale
            reduced += tl.sum(partials_block * row_scale, axis=0)

            running_sum *= rescale
            running_sum += tl.sum(row_scale, axis=0)

    # Finalize & Store Reduction.
    reduced /= running_sum
    finalized = reduced[None, :].to(tl.bfloat16)

    out_shift = (
        (sequence_id * stride_ol)
        + (head_id * latent_dim)
        + latent_offsets
    )

    tl.store(out_ptr + out_shift, finalized, mask=latent_mask)


def mla_buffers(
    query: torch.Tensor, max_pages: int, latent_dim: int, splits_per_page: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    batch_size, timesteps, query_heads, head_size = query.shape
    num_splits = max_pages * splits_per_page

    partial_shape = (batch_size, timesteps, query_heads, num_splits, latent_dim)
    lse_shape = (batch_size, timesteps, query_heads, num_splits)

    partials = query.new_empty(partial_shape, dtype=torch.float32)
    lse = query.new_empty(lse_shape, dtype=torch.float32)

    out = query.new_empty((batch_size, timesteps, query_heads, latent_dim))

    return partials, lse, out


def flash_mla_decode(
    query: torch.Tensor,
    latent_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    latent_dim: int,
    rope_dim: int,
    *,
    partials: torch.Tensor,
    lse: torch.Tensor,
    splits_per_page: int = 1,
) -> torch.Tensor:

    # Tuned Default(s).
    block_h = block_s = 16
    block_k = 32

    # Runtime Dimension(s).
    batch_size, timesteps, query_heads, head_size = query.shape
    num_tokens = batch_size * timesteps

    page_size = latent_cache.size(1)
    max_pages = block_table.size(1)

    num_splits = max_pages * splits_per_page
    split_width = page_size // splits_per_page

    # Stride(s).
    stride_ql = query.stride(1)
    stride_cn = latent_cache.stride(0)

    _, stride_pl, stride_ph, _, _ = partials.stride()
    stride_sl = lse.stride(1)

    # Block Size(s).
    block_z = next_power_of_2(latent_dim)
    block_r = next_power_of_2(rope_dim)

    partials_grid = (num_tokens, num_splits, cdiv(query_heads, block_h))

    # Stage 1: Partials Accumulation.
    mla_decode_partials_kernel[partials_grid](
        # Data Pointer(s).
        query, latent_cache, block_table, cache_seqlens, partials, lse,
        # Stride(s).
        stride_ql, stride_cn, stride_pl, stride_ph, stride_sl,
        # Runtime Dimension(s).
        softmax_scale,
        timesteps,
        query_heads,
        latent_dim,
        rope_dim,
        page_size,
        splits_per_page,
        split_width,
        max_pages,
        num_splits,
        # Block Size Constant(s),
        block_h,
        block_z,
        block_r,
        block_k,
        # Kernel Specific(s).
        num_warps=4,
        num_stages=3,
    )

    out = query.new_empty((batch_size, timesteps, query_heads, latent_dim))
    stride_ol = out.stride(1)

    reduction_grid = (num_tokens, query_heads)

    # Stage 2: Partials Reduction.
    mla_decode_reduction_kernel[reduction_grid](
        # Data Pointer(s).
        partials, lse, cache_seqlens, out,
        # Stride(s).
        stride_pl, stride_ph, stride_sl, stride_ol,
        # Runtime Dimension(s).
        timesteps,
        latent_dim,
        num_splits,
        split_width,
        # Block Size Constant(s),
        block_z,
        block_s,
        # Kernel Specific(s).
        num_warps=8,
        num_stages=3,
    )

    return out


class FlashMLADecode(BaseKernel):

    def meta(
        self,
        query: torch.Tensor,
        latent_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        softmax_scale: float,
        latent_dim: int,
        rope_dim: int,
        partials: torch.Tensor,
        lse: torch.Tensor,
    ) -> torch.Tensor:

        return query.new_empty(
            (query.size(0), query.size(1), query.size(2), latent_dim),
            dtype=query.dtype,
            device=query.device,
        )

    def forward(
        self,
        query: torch.Tensor,
        latent_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        softmax_scale: float,
        latent_dim: int,
        rope_dim: int,
        partials: torch.Tensor,
        lse: torch.Tensor,
    ) -> torch.Tensor:

        return flash_mla_decode(
            query,
            latent_cache,
            cache_seqlens,
            block_table,
            softmax_scale,
            latent_dim,
            rope_dim,
            partials=partials,
            lse=lse,
        )