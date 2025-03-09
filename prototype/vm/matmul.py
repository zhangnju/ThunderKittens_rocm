from matmul import matmul
from timings import save_gantt_chart
import torch
import time

def make_test_args(M, K, N, instruction_width=32, timing_width=128, num_processors=132):
    assert M % 128 == 0
    assert N % 256 == 0
    assert K % 64 == 0
    a = torch.randn(M, K, dtype=torch.bfloat16, device=0).contiguous()
    b = torch.randn(K, N, dtype=torch.bfloat16, device=0).contiguous()
    c = torch.zeros(M, N, dtype=torch.bfloat16, device=0).contiguous()

    tasks = []
    for i in range(M//128):
        for j in range(N//256):
            tasks.append([1, (K+63)//64, i, j])
    tasks = torch.tensor(tasks, dtype=torch.int32, device=0)
    instructions_per_processor = (len(tasks)+num_processors-1) // num_processors
    instructions = torch.zeros((num_processors, instructions_per_processor, instruction_width), dtype=torch.int32, device=0)
    instructions.view(-1, instruction_width)[:tasks.shape[0],:4] = tasks
    timing = torch.zeros((num_processors, instructions_per_processor, timing_width), dtype=torch.int32, device=0)
    semaphore = torch.zeros((num_processors, instructions_per_processor), dtype=torch.int32, device=0)
    return instructions, timing, semaphore, a, b, c

def time_op(func, *args, **kwargs):
    iters = kwargs.get('iters', 100)
    # warmup
    for _ in range(kwargs.get('warmup', 5)):
        func(*args)
    # time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        func(*args)
    end_event.record()
    
    end_event.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # convert ms to seconds
    return elapsed_time / iters

def flops(M,N,K):
    return 2*M*N*K

if __name__ == "__main__":
    # M,N,K = 256, 256, 256
    M,N,K = 2048, 2048, 2048
    instructions, timing, semaphore, a, b, c = make_test_args(M, K, N)

    print('Shapes:', a.shape, b.shape, c.shape)
    print(hex(b.data_ptr())) # hex
    print('Launching kernel...')
    matmul(instructions, timing, semaphore, a, b, c)
    torch.cuda.synchronize()
    c_ref = a @ b
    print(c_ref[:4])
    print(c[:4])
    print(f'all close? {torch.allclose(c, c_ref)}')

    # save_gantt_chart(timing, instructions, 'matmul')

    total_flops = flops(M, N, K)

    scheduler_time = time_op(matmul, instructions, timing, semaphore, a, b, c)
    print(f'scheduler time: {scheduler_time}')
    print(f'tflops: {total_flops / scheduler_time / 1e12}')

    torch_time = time_op(torch.matmul, a, b)
    print(f'torch time: {torch_time}')
    print(f'tflops: {total_flops / torch_time / 1e12}')
