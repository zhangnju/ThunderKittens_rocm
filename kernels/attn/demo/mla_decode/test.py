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

torch.manual_seed(0)

D_Main, D_Rot = 512, 64
PAGE_SIZE = 256
H = 16                  # H heads
NUM_PAGES = 1000        # number of pages in cache
NUM_PROCESSORS = 132    # number of processors
MAX_NUM_PAGES = 65536 // PAGE_SIZE

def init_arguments(seq_lengths: List[int], NEW_TOKENS: int):

    B = len(seq_lengths)

    # Need to initialize QRot, QV, K_cache, V_cache, Lengths, Table    
    QRot    = torch.randn(B, NEW_TOKENS, H, D_Rot, dtype=torch.bfloat16, device='cuda')
    QV      = torch.randn(B, NEW_TOKENS, H, D_Main, dtype=torch.bfloat16, device='cuda')
    K_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Rot, dtype=torch.bfloat16, device='cuda')
    V_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Main, dtype=torch.bfloat16, device='cuda')
    Lengths = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    # Table = torch.arange(MAX_NUM_PAGES, dtype=torch.int32, device='cuda').reshape(1, -1).repeat(B, 1)
    Table = torch.randint(0, NUM_PAGES, (B, MAX_NUM_PAGES), dtype=torch.int32, device='cuda')

    return QRot, QV, K_cache, V_cache, Lengths, Table

def create_thundermla_arguments(seq_lengths, NEW_TOKENS):
    # Processor assignment heuristic: assign processors proportionally to sequence lengths.
    num_processors = [math.floor(s / sum(seq_lengths) * NUM_PROCESSORS) for s in seq_lengths]
    while sum(num_processors) < NUM_PROCESSORS:
        min_idx = num_processors.index(min(num_processors))
        num_processors[min_idx] += 1
    # while min(num_processors) < 4:
    #     max_idx = num_processors.index(max(num_processors))
    #     min_idx = num_processors.index(min(num_processors))
    #     num_processors[max_idx] -= 1
    #     num_processors[min_idx] += 1
    # Create schedule
    start_processors = [sum(num_processors[:i]) for i in range(len(num_processors))]
    scheduled_tasks = []
    partial_uid, reduction_uid = 0, NUM_PROCESSORS
    for batch_id, (seq_l, start_p, num_p) in enumerate(zip(seq_lengths, start_processors, num_processors)):
        new_tasks, partial_uid, reduction_uid = backward_schedule(
            list(range(start_p, start_p + num_p)), batch_id, seq_l, list(range(NEW_TOKENS)), partial_uid, reduction_uid
        )
        scheduled_tasks.extend(new_tasks)
    breakpoint()
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, NEW_TOKENS, num_processors=NUM_PROCESSORS
    )
    print('Finished generating schedule + arguments')
    return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings

def run_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, tic=None):
    if tic is None:
        Semaphore.zero_()
        tic = 1
    O = torch.zeros_like(QV)
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    mla_decode.mla_decode(Instructions,QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, Timings)
    return O

def run_mla_torch(QRot, QV, K_cache, V_cache, Lengths, Table):
    Q = torch.concat([QRot, QV], dim=-1)
    full_K = torch.cat([K_cache, V_cache], dim=-1)[Table].reshape(Q.shape[0], -1, Q.shape[-1])
    full_V = V_cache[Table].reshape(Q.shape[0], -1, QV.shape[-1])
    softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
    O = torch.zeros_like(QV)
    for b, l in enumerate(Lengths):
        assert Q.shape[1] == 1, "Q must have shape (B, 1, H, D) for the time being."
        O[b:b+1] = torch.nn.functional.scaled_dot_product_attention(
            Q[b:b+1].transpose(1, 2),
            full_K[b:b+1, :l].unsqueeze(-2).repeat((1,1,H,1)).transpose(1, 2),
            full_V[b:b+1, :l].unsqueeze(-2).repeat((1,1,H,1)).transpose(1, 2),
            is_causal=False,
            scale=softmax_scale
        ).transpose(1, 2)
    return O

def run_mla_sdpa(QRot, QV, K_cache, V_cache, Lengths, Table):
    pass

def main():
    seq_lengths=sorted([256])
    # seq_lengths=sorted([32768*2])
    # seq_lengths=sorted([4641,45118,1730,1696])
    NEW_TOKENS = 1
    QRot, QV, K_cache, V_cache, Lengths, Table = init_arguments(seq_lengths, NEW_TOKENS)
    ref = run_mla_torch(QRot, QV, K_cache, V_cache, Lengths, Table)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_thundermla_arguments(seq_lengths, NEW_TOKENS)
    O = run_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings)
    print('O', O)
    print('Ref', ref)
    for b in range(len(seq_lengths)):
        print("ref mean:", torch.mean(ref[b].abs()))
        print("Kernel output mean:", torch.mean(O[b].abs()))
        print("Max absolute diff:", torch.max(torch.abs(O[b] - ref[b])))
        print("Avg absolute diff:", torch.mean(torch.abs(O[b] - ref[b])))
    save_gantt_chart(Timings, Instructions)
    breakpoint()

if __name__ == "__main__":
    main()