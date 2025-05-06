import mla_decode
import torch
import numpy as np
import math
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scheduler_v2 import backward_schedule
from scheduler import sample_schedule_generator, priority_schedule_tasks, visualize_schedule, create_arguments_from_task_schedule
from timings import save_gantt_chart
from scheduler_regression import estimate_schedule_length
from scheduler_v2 import Task

torch.manual_seed(0)

GPU = 'B200'
D_Main, D_Rot = 512, 64
PAGE_SIZE = 256
# H = 16                  # set by q_heads
NUM_PAGES = 10000        # number of pages in cache
NUM_PROCESSORS = 148 if GPU == 'B200' else 132    # number of processors
MAX_NUM_PAGES = 65536 // PAGE_SIZE

ENABLE_TIMINGS = True

def init_arguments(seq_lengths: List[int], new_tokens: int, q_heads: int=16):

    B = len(seq_lengths)

    # Need to initialize QRot, QV, K_cache, V_cache, Lengths, Table    
    QRot    = torch.randn(B, new_tokens, q_heads, D_Rot, dtype=torch.bfloat16, device='cuda')
    QV      = torch.randn(B, new_tokens, q_heads, D_Main, dtype=torch.bfloat16, device='cuda')
    K_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Rot, dtype=torch.bfloat16, device='cuda')
    V_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Main, dtype=torch.bfloat16, device='cuda')
    Lengths = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    Table = torch.randint(0, NUM_PAGES, (B, MAX_NUM_PAGES), dtype=torch.int32, device='cuda')

    return QRot, QV, K_cache, V_cache, Lengths, Table

def create_thundermla_arguments(seq_lengths, new_tokens, q_heads = 16, block_size=128, prints=True):
    # Processor assignment heuristic: assign processors proportionally to sequence lengths.
    t0 = time.time()
    scheduled_tasks = []
    idx = 0

    for i in range(len(seq_lengths)):
        token_map = {
            l: [] for l in range(new_tokens)
        }

        for j in range((seq_lengths[i] + block_size - 1) // block_size):
            for k in range(0, new_tokens, 4):
                write_scratch = True#(block_size < seq_lengths[i])
                scheduled_tasks.append(
                    Task(
                        uid=idx,
                        batch_id=i,
                        tok_ids=[k + l for l in range(4) if k + l < new_tokens],
                        name="Partial",
                        task_type="partial",
                        start=idx,
                        finish=idx+1,
                        dependencies=[],
                        processor=(i+j+k) % NUM_PROCESSORS,
                        args={
                            "write_scratch": write_scratch,
                            "start": j*block_size,
                            "end": (j+1)*block_size,
                            "length": seq_lengths[i],
                        }
                    )
                )

                if write_scratch:
                    for l in range(4):
                        if k + l < new_tokens:
                            token_map[k + l].append(idx)
                idx += 1

        for k in range(0, new_tokens):
            if len(token_map[k]) > 0:
                scheduled_tasks.append(
                    Task(
                        uid=idx,
                        batch_id=i,
                        tok_ids=[k],
                        name="Reduction",
                        task_type="reduction",
                        start=idx,
                        finish=idx+1,
                        next_input_time=new_tokens,
                        dependencies=token_map[k],
                        processor=(i+k) % NUM_PROCESSORS,
                        args={"write_scratch": False}
                    )
                )
                idx += 1
    
    # token_maps = []
    # for i in range(len(seq_lengths)):
    #     token_map = {
    #         l: [] for l in range(new_tokens)
    #     }

    #     for j in range((seq_lengths[i] + block_size - 1) // block_size):
    #         for k in range(0, new_tokens, 4):
    #             write_scratch = True#(block_size < seq_lengths[i])
    #             scheduled_tasks.append(
    #                 Task(
    #                     uid=idx,
    #                     batch_id=i,
    #                     tok_ids=[k + l for l in range(4) if k + l < new_tokens],
    #                     name="Partial",
    #                     task_type="partial",
    #                     start=idx,
    #                     finish=idx+1,
    #                     dependencies=[],
    #                     processor=(i+j+k) % NUM_PROCESSORS,
    #                     args={
    #                         "write_scratch": write_scratch,
    #                         "start": j*block_size,
    #                         "end": (j+1)*block_size,
    #                         "length": seq_lengths[i],
    #                     }
    #                 )
    #             )

    #             if write_scratch:
    #                 for l in range(4):
    #                     if k + l < new_tokens:
    #                         token_map[k + l].append(idx)
    #             idx += 1

    #     token_maps.append(token_map)

    # for i in range(len(seq_lengths)):
    #     for k in range(0, new_tokens):
    #         if len(token_maps[i][k]) > 0:
    #             scheduled_tasks.append(
    #                 Task(
    #                     uid=idx,
    #                     batch_id=i,
    #                     tok_ids=[k],
    #                     name="Reduction",
    #                     task_type="reduction",
    #                     start=idx,
    #                     finish=idx+1,
    #                     next_input_time=new_tokens,
    #                     dependencies=token_maps[i][k],
    #                     processor=(i+k) % NUM_PROCESSORS,
    #                     args={"write_scratch": False}
    #                 )
    #             )
    #             idx += 1

    t1 = time.time()
    if prints:
        print(f'Time taken to create schedule: {(t1-t0)*1000} ms')

    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, new_tokens, num_processors=NUM_PROCESSORS, enable_timings=ENABLE_TIMINGS, q_heads=q_heads, prints=prints
    )
    # visualize_schedule(scheduled_tasks, NUM_PROCESSORS)
    return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings

def run_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, tic=None, prints=True):
    q_heads = QRot.shape[2]
    if tic is None:
        Semaphore.zero_()
        tic = 1
    O = torch.zeros_like(QV)
    Q_all = torch.concat([QV, QRot], dim=-1).contiguous()
    KV_all = torch.cat([V_cache, K_cache], dim=-1).contiguous()
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    mla_decode_fn = mla_decode.mla_decode_8_heads if q_heads == 8 else mla_decode.mla_decode
    if prints:
        print('Starting mla_decode')
    if Timings is not None:
        mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, Timings)
        #mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1-tic, Timings)
    else:
        mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic)
        #mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1-tic)
    torch.cuda.synchronize()
    if prints:
        print('Finished mla_decode')
    return O, Timings

def compute_thundermla_partials(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, block_size, tic=None, prints=True):
    q_heads = QRot.shape[2]
    if tic is None:
        Semaphore.zero_()
        tic = 1
    O = torch.zeros_like(QV)
    Q_all = torch.concat([QV, QRot], dim=-1).contiguous()
    KV_all = torch.cat([V_cache, K_cache], dim=-1).contiguous()
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    mla_decode_fn = mla_decode.mla_decode_8_heads if q_heads == 8 else mla_decode.mla_decode
    if prints:
        print('Starting mla_decode')
    if Timings is not None:
        mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, Timings)
        #mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1-tic, Timings)
    else:
        mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic)
        #mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1-tic)
    torch.cuda.synchronize()
    if prints:
        print('Finished mla_decode')

    Os = []
    ls = []
    idx = 0
    for b, l in enumerate(Lengths):
        Os.append([])
        ls.append([])
        for block_start in range(0, l, block_size):
            Os[b].append([])
            ls[b].append([])
            for k in range(0, QRot.shape[1], 4):
                O_block = O_scratch[idx, k:k+4]
                L_block = Lvec_scratch[idx, k:k+4]

                Os[b][-1].append(O_block)
                ls[b][-1].append(L_block)
                idx += 1
            
            Os[b][-1] = torch.cat(Os[b][-1], dim=0)
            ls[b][-1] = torch.cat(ls[b][-1], dim=0)

        idx += QRot.shape[1]
        
    return Os, ls, Timings

def profile_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, ITERS=100):
    q_heads = QRot.shape[2]
    Semaphore.zero_()
    O = torch.zeros_like(QV)
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    stream = torch.cuda.current_stream()
    # execute once to warm up
    mla_decode_fn = mla_decode.mla_decode_8_heads if q_heads == 8 else mla_decode.mla_decode
    if Timings is not None:
        mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1, Timings, stream=stream)
    else:
        mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, 1, stream=stream)
    torch.cuda.synchronize()
    t0 = time.time()
    for it in range(ITERS):
        if Timings is not None:
            mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, it%2, Timings, stream=stream)
        else:
            mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, it%2, stream=stream)
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1-t0) / ITERS

def run_mla_torch(QRot, QV, K_cache, V_cache, Lengths, Table):
    Q = torch.concat([QRot, QV], dim=-1)
    q_heads = Q.shape[2]
    full_K = torch.cat([K_cache, V_cache], dim=-1)[Table].reshape(Q.shape[0], -1, Q.shape[-1])
    full_V = V_cache[Table].reshape(Q.shape[0], -1, QV.shape[-1])
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    O = torch.zeros_like(QV)
    for b, l in enumerate(Lengths):
        # assert Q.shape[1] == 1, "Q must have shape (B, 1, H, D) for the time being."
        mask = torch.ones(Q.shape[1], l, dtype=torch.bool).tril(diagonal=l-Q.shape[1]).to(Q.device)
        O[b:b+1] = torch.nn.functional.scaled_dot_product_attention(
            Q[b:b+1].transpose(1, 2),
            full_K[b:b+1, :l].unsqueeze(-2).repeat((1,1,q_heads,1)).transpose(1, 2),
            full_V[b:b+1, :l].unsqueeze(-2).repeat((1,1,q_heads,1)).transpose(1, 2),
            is_causal=False,
            attn_mask=mask,
            scale=softmax_scale
        ).transpose(1, 2)
    return O

def compute_mla_partials_torch(QRot, QV, K_cache, V_cache, Lengths, Table, block_size):
    Q = torch.concat([QRot, QV], dim=-1).to(torch.float32) # (B, LQ, H, D)
    q_heads = Q.shape[2]
    full_K = torch.cat([K_cache, V_cache], dim=-1)[Table].reshape(Q.shape[0], -1, Q.shape[-1])
    full_V = V_cache[Table].reshape(Q.shape[0], -1, QV.shape[-1])
    Q = Q.transpose(1, 2) # (B, H, LQ, D)

    # pad K and V to be a multiple of block_size
    full_K = torch.cat([full_K, torch.zeros(full_K.shape[0], block_size - full_K.shape[1] % block_size, full_K.shape[2], dtype=full_K.dtype, device=full_K.device)], dim=1)
    full_V = torch.cat([full_V, torch.zeros(full_V.shape[0], block_size - full_V.shape[1] % block_size, full_V.shape[2], dtype=full_V.dtype, device=full_V.device)], dim=1)

    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    Os = []
    ls = []
    for b, l in enumerate(Lengths):
        Os.append([])
        ls.append([])
        for block_start in range(0, l, block_size):
            Os[b].append([])
            ls[b].append([])
            for k in range(0, Q.shape[2], 4):
                K_block = full_K[b, block_start:block_start+block_size].to(torch.float32) # (L, D)
                V_block = full_V[b, block_start:block_start+block_size].to(torch.float32) # (L, D)

                q_loc = torch.arange(l - Q.shape[2] + k, min(l, l - Q.shape[2] + k + 4), device=Q.device)
                k_loc = torch.arange(block_start, block_start + block_size, device=Q.device)
                mask = (q_loc[:, None] >= k_loc[None, :]).unsqueeze(0) # (1, LQ, L)

                QK = Q[b, :, k:k+4] @ K_block.transpose(0, 1) # (H, LQ<=4, L)
                QK = QK.masked_fill(~mask, -10000000)

                max_vec = torch.max(QK, dim=-1, keepdim=True).values # (H, LQ<=4, 1)
                QK = QK - max_vec
                exp_QK = torch.exp(QK * softmax_scale) # (H, LQ<=4, L)

                l_block = torch.sum(exp_QK, dim=-1, keepdim=True) # (H, LQ<=4, 1)

                ls[b][-1].append((1.44269504089 * max_vec * softmax_scale + torch.log2(l_block)).transpose(0, 1)[..., 0])

                O_block = exp_QK @ V_block / l_block # (H, LQ<=4, D)
                Os[b][-1].append(O_block.transpose(0, 1))

            Os[b][-1] = torch.cat(Os[b][-1], dim=0)
            ls[b][-1] = torch.cat(ls[b][-1], dim=0)

    return Os, ls

def main(seq_lengths, new_tokens, q_heads=16, block_size=128, prints=True):
    if prints:
        print(f' ----------- starting seq_lengths: {seq_lengths} new_tokens: {new_tokens} q_heads: {q_heads} block_size: {block_size} -----------')
    seq_lengths = sorted(seq_lengths)
    QRot, QV, K_cache, V_cache, Lengths, Table = init_arguments(seq_lengths, new_tokens, q_heads)
    ref = run_mla_torch(QRot, QV, K_cache, V_cache, Lengths, Table)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_thundermla_arguments(seq_lengths, new_tokens, q_heads, block_size, prints=prints)
    O, Timings = run_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, prints=prints)
    torch.cuda.synchronize()
    # save_gantt_chart(Timings, Instructions, f"gantt_chart_{round(time.time())}.png", verbose=True)
    # print("ref mean:", torch.mean(ref.abs()))
    # print("Kernel output mean", torch.mean(O.abs()))
    # print("Max absolute diff", torch.max(torch.abs(O - ref)))
    # print("Avg absolute diff", torch.mean(torch.abs(O - ref)))
    if prints:
        print('Finished!')

    return {
        "Ref mean\t\t": torch.mean(ref.abs()),
        "Kernel output mean\t": torch.mean(O.abs()),
        "Max absolute diff\t": torch.max(torch.abs(O - ref)),
        "Avg absolute diff\t": torch.mean(torch.abs(O - ref)),
    }

def main_check_partials(seq_lengths, new_tokens, q_heads=16, block_size=128, prints=True):
    if prints:
        print(f' ----------- starting seq_lengths: {seq_lengths} new_tokens: {new_tokens} q_heads: {q_heads} block_size: {block_size} -----------')
    seq_lengths = sorted(seq_lengths)
    QRot, QV, K_cache, V_cache, Lengths, Table = init_arguments(seq_lengths, new_tokens, q_heads)
    Os_torch, ls_torch = compute_mla_partials_torch(QRot, QV, K_cache, V_cache, Lengths, Table, block_size)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_thundermla_arguments(seq_lengths, new_tokens, q_heads, block_size, prints=prints)
    Os, ls, Timings = compute_thundermla_partials(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, block_size, prints=prints)
    torch.cuda.synchronize()
    # print("ref mean:", torch.mean(ref.abs()))
    # print("Kernel output mean", torch.mean(O.abs()))
    # print("Max absolute diff", torch.max(torch.abs(O - ref)))
    # print("Avg absolute diff", torch.mean(torch.abs(O - ref)))
    if prints:
        print('Finished!')
    
    # for i in range(len(Os)):
    #     for j in range(len(Os[i])):
    #         fig, axs = plt.subplots(Os[i][j].shape[0])
    #         for k in range(Os_torch[i][j].shape[0]):
    #             im = axs[k].imshow(Os_torch[i][j][k].cpu().numpy() - Os[i][j][k].cpu().numpy())
    #             fig.colorbar(im, ax=axs[k])
            
    #         plt.savefig(f"Os[{i}][{j}].png")
    #         plt.close()


    stats_dict = {}

    for i in range(len(Os)):
        for j in range(len(Os[i])):
            stats_dict[f"\nOs_torch[{i}][{j}] mean\t"] = torch.mean(Os_torch[i][j].abs())
            stats_dict[f"Os[{i}][{j}] mean\t\t"] = torch.mean(Os[i][j].abs())

            stats_dict[f"\nls_torch[{i}][{j}] mean\t"] = torch.mean(ls_torch[i][j].abs())
            stats_dict[f"ls[{i}][{j}] mean\t\t"] = torch.mean(ls[i][j].abs())

    for i in range(len(Os)):
        for j in range(len(Os[i])):
            if i == 0 and j == 0:
                stats_dict[f"\n*************\nOs[{i}][{j}] max diff\t"] = torch.max(torch.abs(Os_torch[i][j] - Os[i][j]))
            else:
                stats_dict[f"\nOs[{i}][{j}] max diff\t"] = torch.max(torch.abs(Os_torch[i][j] - Os[i][j]))
            
            stats_dict[f"Os[{i}][{j}] mean diff\t"] = torch.mean(torch.abs(Os_torch[i][j] - Os[i][j]))

            stats_dict[f"\nls[{i}][{j}] max diff\t"] = torch.max(torch.abs(ls_torch[i][j] - ls[i][j]))
            stats_dict[f"ls[{i}][{j}] mean diff\t"] = torch.mean(torch.abs(ls_torch[i][j] - ls[i][j]))

    return stats_dict


def test_inputs(inputs, prints=False, check_partials=False):
    for input in inputs:
        seq_lengths = input[0]
        new_tokens = input[1]
        q_heads = input[2]
        block_size = input[3]
        num_trials = input[4]

        for seq_length in seq_lengths:
            assert seq_length >= new_tokens, "seq_length must be greater than or equal to new_tokens"
        
        assert q_heads % 4 == 0, "q_heads must be divisible by 4"
        assert block_size % 32 == 0, "block_size must be divisible by 32"

        outs = None
        combined_outs = []

        print(f'\n\n----------- starting seq_lengths: {seq_lengths} new_tokens: {new_tokens} q_heads: {q_heads} block_size: {block_size} -----------')
        for i in range(num_trials):
            torch.manual_seed(0)
            if check_partials:
                out = main_check_partials(seq_lengths, new_tokens, q_heads, block_size, prints=prints)
            else:
                out = main(seq_lengths, new_tokens, q_heads, block_size, prints=prints)
            combined_outs.append(out)

            if outs is None:
                outs = out
            else:
                for key in outs.keys():
                    if torch.allclose(outs[key], out[key]):
                        continue
                    else:
                        print(f"Mismatch at {key}: {outs[key]} != {out[key]}")
                        break

        for key in outs.keys():
            print(key, end="")
            for out in combined_outs:
                print(f"{out[key].item():.2e}", end="\t")
            print()

if __name__ == "__main__":
    # test_inputs([
    #     ([1], 1, 16, 128, 10),
    #     ([8], 8, 16, 128, 10),
    #     ([100], 8, 16, 32, 10),
    #     ([100], 1, 16, 32, 10),
    #     ([32], 1, 16, 32, 10),
    #     ([128], 1, 16, 32, 10),
    #     ([128], 1, 16, 128, 10),
    #     ([128, 256, 512, 1024, 2048], 1, 16, 128, 10),
    #     ([423, 511, 245, 11, 2], 1, 16, 32, 10),
    #     ([4230, 5110, 10230, 110, 20], 8, 16, 512, 10),
    # ])

    # test_inputs([
    #     ([260, 310], 8, 16, 32, 2),
    # ], check_partials=True)

    test_inputs([
        ([260, 310, 12, 8, 348, 332], 8, 16, 32, 2),
    ])#, check_partials=True)