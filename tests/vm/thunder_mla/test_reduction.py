import random
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
    QRot    = torch.randn(B, new_tokens, q_heads, D_Rot, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    QV      = torch.randn(B, new_tokens, q_heads, D_Main, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    K_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Rot, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    V_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Main, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    Lengths = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    Table = torch.randint(0, NUM_PAGES, (B, MAX_NUM_PAGES), dtype=torch.int32, device='cuda')

    return QRot, QV, K_cache, V_cache, Lengths, Table

def init_o_and_lvec_scratch(o_scratch, lvec_scratch, reduction_tasks):
    all_dependencies = []
    all_token_ids = []
    for task in reduction_tasks:
        all_dependencies.extend(task.dependencies)
        all_token_ids.extend(task.tok_ids * len(task.dependencies))

    all_dependencies = list(set((dependency, token_id) for dependency, token_id in zip(all_dependencies, all_token_ids)))
    
    for i, (dependency, token_id) in enumerate(all_dependencies):
        o_scratch[dependency, token_id:token_id+4] = (torch.randn((o_scratch.shape[2], o_scratch.shape[3]), dtype=torch.float32, device='cuda')).to(torch.float8_e4m3fn)
        lvec_scratch[dependency, token_id:token_id+4] = torch.randn((lvec_scratch.shape[2],), dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)

    return o_scratch, lvec_scratch

def get_correct_O_output(o_scratch, lvec_scratch, reduction_tasks, B, new_tokens, q_heads, D_Main):
    O = torch.zeros((B, new_tokens, q_heads, D_Main), dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)

    for task in reduction_tasks:
        if not task.args["write_scratch"]:
            relevant_lvecs = torch.clone(lvec_scratch[task.dependencies, task.tok_ids[0]])
            max_lvec = torch.max(relevant_lvecs, dim=0).values
            relevant_lvecs = relevant_lvecs - max_lvec
            exp_relevant_lvecs = torch.exp2(relevant_lvecs)
            l_block = torch.sum(exp_relevant_lvecs, dim=0)
            normed_relevant_lvecs = exp_relevant_lvecs / l_block

            relevant_o_blocks = torch.clone(o_scratch[task.dependencies, task.tok_ids[0]])
            relevant_o_blocks = relevant_o_blocks * normed_relevant_lvecs.unsqueeze(2)
            O[task.batch_id, task.tok_ids[0]] = torch.sum(relevant_o_blocks, dim=0)

    return O

def get_correct_O_and_Lvec_scratch_output(o_scratch, lvec_scratch, reduction_tasks, B, new_tokens, q_heads, D_Main):
    max_uid = max(task.uid for task in reduction_tasks)
    O_scratch = o_scratch.clone()
    Lvec_scratch = lvec_scratch.clone()

    for task in reduction_tasks:
        if task.args["write_scratch"]:
            relevant_lvecs = torch.clone(Lvec_scratch[task.dependencies, task.tok_ids[0]]).to(torch.float32)
            max_lvec = torch.max(relevant_lvecs, dim=0).values
            relevant_lvecs = relevant_lvecs - max_lvec
            exp_relevant_lvecs = torch.exp2(relevant_lvecs)
            l_block = torch.sum(exp_relevant_lvecs, dim=0)
            normed_relevant_lvecs = exp_relevant_lvecs / l_block

            relevant_o_blocks = torch.clone(O_scratch[task.dependencies, task.tok_ids[0]]).to(torch.float32)
            relevant_o_blocks = relevant_o_blocks * normed_relevant_lvecs.unsqueeze(2)

            O_scratch[task.uid, task.tok_ids[0]] = torch.sum(relevant_o_blocks, dim=0)
            Lvec_scratch[task.uid, task.tok_ids[0]] = torch.log2(l_block) + max_lvec

    return O_scratch, Lvec_scratch

def get_correct_outputs(O_scratch, Lvec_scratch, reduction_tasks, B, new_tokens, q_heads, D_Main):
    O = get_correct_O_output(O_scratch, Lvec_scratch, reduction_tasks, B, new_tokens, q_heads, D_Main)
    O_scratch, Lvec_scratch = get_correct_O_and_Lvec_scratch_output(O_scratch, Lvec_scratch, reduction_tasks, B, new_tokens, q_heads, D_Main)

    return O, O_scratch, Lvec_scratch

def partial_reduction_task(seq_lengths, new_tokens=8, q_heads = 16, block_size=128, write_scratch=False, prints=True):
    scheduled_tasks = []
    idx = 10
    for seq_length in seq_lengths:
        for i in range(new_tokens):
            scheduled_tasks.append(
                Task(
                    uid=idx,
                    batch_id=0,
                    tok_ids=[i],
                    name="Reduction",
                    task_type="reduction",
                    start=idx,
                    finish=idx+1,
                    dependencies=list(set([random.randint(0,idx-1) for _ in range(random.randint(1, 10))])),
                    processor=idx % NUM_PROCESSORS,
                    args={"write_scratch": write_scratch}
                )
            )
            idx += 1

            scheduled_tasks.append(
                Task(
                    uid=idx,
                    batch_id=0,
                    tok_ids=[i],
                    name="Reduction",
                    task_type="reduction",
                    start=idx,
                    finish=idx+1,
                    dependencies=list(set([random.randint(0,idx-1) for _ in range(random.randint(1, 10))])),
                    processor=idx % NUM_PROCESSORS,
                    args={"write_scratch": write_scratch}
                )
            )
            idx += 1

            scheduled_tasks.append(
                Task(
                    uid=idx,
                    batch_id=0,
                    tok_ids=[0],
                    name="Reduction",
                    task_type="reduction",
                    start=idx,
                    finish=idx+1,
                    dependencies=[idx-1, idx-2],
                    processor=idx % NUM_PROCESSORS,
                    args={"write_scratch": write_scratch}
                )
            )

    print([task.dependencies for task in scheduled_tasks])

    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, 
        new_tokens, 
        num_processors=NUM_PROCESSORS, 
        enable_timings=ENABLE_TIMINGS, 
        q_heads=q_heads, 
        prints=prints,
    )

    return scheduled_tasks, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings

def run_thundermla_reductions(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, scheduled_tasks, O_scratch, Lvec_scratch, Semaphore, Timings, tic=None, prints=True):
    q_heads = QRot.shape[2]
    if tic is None:
        Semaphore.zero_()
        Semaphore[10, 0] = 1
        Semaphore[11, 0] = 1
        tic = 0
    O = torch.zeros_like(QV)
    Q_all = torch.concat([QV, QRot], dim=-1).contiguous()
    KV_all = torch.cat([V_cache, K_cache], dim=-1).contiguous()
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    mla_decode_fn = mla_decode.mla_decode_8_heads if q_heads == 8 else mla_decode.mla_decode
    if prints:
        print('Starting mla_decode')

    O_scratch = O_scratch.clone()
    Lvec_scratch = Lvec_scratch.clone()
    
    start = time.time()
    if Timings is not None:
        mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, Timings)
    else:
        mla_decode_fn(Instructions, QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic)
    torch.cuda.synchronize()
    end = time.time()

    if prints:
        print('Finished mla_decode')
        print(f"Time taken to run mla_decode: {end - start} seconds")
    
    return O, O_scratch, Lvec_scratch

def main(seq_lengths, new_tokens, q_heads=16, block_size=128, prints=True, write_scratch=False):
    if prints:
        print(f' ----------- starting seq_lengths: {seq_lengths} new_tokens: {new_tokens} q_heads: {q_heads} block_size: {block_size} -----------')
    seq_lengths = sorted(seq_lengths)
    QRot, QV, K_cache, V_cache, Lengths, Table = init_arguments(seq_lengths, new_tokens, q_heads)
    scheduled_tasks, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = partial_reduction_task(seq_lengths, new_tokens, q_heads, block_size, write_scratch, prints=prints)
    o_scratch, lvec_scratch = init_o_and_lvec_scratch(O_scratch, Lvec_scratch, scheduled_tasks)
    O_correct, O_scratch_correct, Lvec_scratch_correct = get_correct_outputs(O_scratch, Lvec_scratch, scheduled_tasks, len(seq_lengths), new_tokens, q_heads, D_Main)

    if prints:
        print('Finished!')

    if not write_scratch:
        print("O max diff \t\tO mean diff")
    else:
        print("O_scratch max diff \t\tO_scratch mean diff \t\tLvec_scratch max diff \t\tLvec_scratch mean diff")

    for i in range(100):
        O_out, O_scratch_out, Lvec_scratch_out = run_thundermla_reductions(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, scheduled_tasks, o_scratch, lvec_scratch, Semaphore, Timings, prints=prints)
        torch.cuda.synchronize()

        if not write_scratch:
            O_out = O_out.to(torch.float32)
            O_correct = O_correct.to(torch.float32)
            # fig = plt.figure()
            # fig.set_size_inches(10, 2)
            # plt.imshow(O_out[0, 0].cpu().numpy() - O_correct[0, 0].cpu().numpy())
            # for x in range(0, 512, 512//16):
            #     plt.axvline(x, linestyle=":", color='black', linewidth=0.5)
            # plt.savefig(f"O_out_diff.png", dpi=300)
            
            print(f"{torch.max(torch.abs(O_out - O_correct)):.2e}", end="\t\t")
            print(f"{torch.mean(torch.abs(O_out - O_correct)):.2e}")
        else:
            # fig = plt.figure()
            # fig.set_size_inches(10, 2)
            # axes = fig.subplots(3,1)

            # for i in range(3):
            #     if i == 0:
            #         axes[i].imshow(O_scratch_out[10, 0].cpu().numpy())
            #     elif i == 1:
            #         axes[i].imshow(O_scratch_correct[10, 0].cpu().numpy())
            #     else:
            #         axes[i].imshow(O_scratch_out[10, 0].cpu().numpy() - O_scratch_correct[10, 0].cpu().numpy())
            #     for x in range(0, 512, 512//16):
            #         axes[i].axvline(x, linestyle=":", color='black', linewidth=0.5)
            # fig.tight_layout()
            # plt.savefig(f"O_scratch_out_diff.png", dpi=300)

            print(f"{torch.max(torch.abs(O_scratch_out - O_scratch_correct)):.2e}", end="\t\t\t")
            print(f"{torch.mean(torch.abs(O_scratch_out - O_scratch_correct)):.2e}", end="\t\t\t")
            print(f"{torch.max(torch.abs(Lvec_scratch_out - Lvec_scratch_correct)):.2e}", end="\t\t\t")
            print(f"{torch.mean(torch.abs(Lvec_scratch_out - Lvec_scratch_correct)):.2e}")

        input()

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
    num_tokens = 1
    main([128], num_tokens, 16, 32, prints=False, write_scratch=True)
    #main([128], num_tokens, 16, 32, prints=False, write_scratch=True)