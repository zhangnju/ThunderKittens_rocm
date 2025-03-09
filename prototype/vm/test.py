from test import test
from timings import save_gantt_chart
import torch
import time

def make_test_args(num_instructions, instruction_width=32, timing_width=128, num_processors=132, active_processors=None):
    if active_processors is None:
        active_processors = num_processors
    instructions = torch.zeros((num_processors, num_instructions, instruction_width), dtype=torch.int32, device=0)
    instructions[:active_processors, :num_instructions, :2] = torch.tensor([[1, 10]], dtype=torch.int32, device=0)
    timing = torch.zeros((num_processors, num_instructions, timing_width), dtype=torch.int32, device=0)
    semaphore = torch.zeros((num_processors, num_instructions), dtype=torch.int32, device=0)
    return instructions, timing, semaphore

def time_op(func, *args, **kwargs):
    iters = kwargs.get('iters', 100)
    # warmup
    for _ in range(kwargs.get('warmup', 5)):
        func(*args)
    # time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        func(*args)
    torch.cuda.synchronize()
    return (time.time() - start) / iters

def flops(M,N,K):
    return 2*M*N*K

if __name__ == "__main__":
    instructions, timing, semaphore = make_test_args(5, num_processors=132, active_processors=24)
    print(instructions[:4])

    print('Launching kernel...')
    test(instructions, timing, semaphore)
    print('Awaiting kernel completion...')
    torch.cuda.synchronize()
    print('Kernel completed.')

    save_gantt_chart(timing, instructions, 'test')
