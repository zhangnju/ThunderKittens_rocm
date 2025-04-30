import torch
from matmul import matmul
import time


###
#   Global Parameters
###
NUM_DEVICES = 8
NUM_ITERS = 5
OPCODE = 725
# M, K, N = 3072, 4096, 3072
# M, K, N = 512, 256, 256
M, K, N = 16384, 3072, 16384
# M, K, N = 3072, 16384*2, 3072
# M, K, N = 256, 4096, 256

if M%256 != 0: raise ValueError("M must be divisible by 256")
if K%128 != 0: raise ValueError("K must be divisible by 128")
if N%256 != 0: raise ValueError("N must be divisible by 256")


###
#   Prepare Inputs
###
print(f'Starting test with M={M}, K={K}, N={N}')
print('\nGenerating inputs...')
torch.manual_seed(42)
dev_ids = [i for i in range(NUM_DEVICES)]
torch_devices = [torch.device(f"cuda:{dev_id}") for dev_id in dev_ids]
A = (torch.randn((M, K), device='cpu', dtype=torch.float32) / K**.25).to(dtype=torch.float8_e4m3fn)
B = (torch.randn((N, K), device='cpu', dtype=torch.float32) / K**.25).to(dtype=torch.float8_e4m3fn)
C =  torch.zeros((M, N), device='cpu', dtype=torch.float8_e4m3fn)

# Shard the inputs
As = [tensor.to(torch_devices[i]) for i, tensor in enumerate(A.chunk(len(torch_devices), dim=0))]
Bs = [tensor.to(torch_devices[i]) for i, tensor in enumerate(B.chunk(len(torch_devices), dim=1))]
Cs = [tensor.to(torch_devices[i]) for i, tensor in enumerate(C.chunk(len(torch_devices), dim=1))]


###
#   Prepare Instructions, Timings, and Barriers
###
print('\nGenerating instructions and timings...')
instructions = []
timings = []
for torch_device in range(len(torch_devices)):
    dev_instructions = [[] for _ in range(148)]
    instruction_idx = 0
    for row in range(M // 256): # ceil
        for col in range(N // 256):
            dev_instructions[instruction_idx % 148].append([OPCODE, row * 2, col * 2, K // 128] + [0] * 28)
            instruction_idx += 1
    while instruction_idx%148 != 0:
        dev_instructions[instruction_idx%148].append([0]*32)
        instruction_idx += 1
    dev_instructions = torch.tensor(dev_instructions, dtype=torch.int32, device=torch_device)
    instructions.append(dev_instructions)
    timings.append(torch.zeros((148, instruction_idx // 148, 128), dtype=torch.int32, device=torch_device))


print(instructions[0])
print(timings[0])

###
#   Launch the kernel and benchmark
###
print("\nLaunching the kernel...")
matmul(instructions, timings, As, Bs, Cs, dev_ids)
torch.cuda.synchronize()

# print('\nKernel finished, now benchmarking...')
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
# start_event.record()
# for i in range(NUM_ITERS):
#     matmul(instructions[0], timings[0], As, Bs, Cs, 0)
# end_event.record()
# torch.cuda.synchronize()

elapsed_time = start_event.elapsed_time(end_event)
time_per_iter_us = elapsed_time * 1e3 / NUM_ITERS
print(f'Time per iter: {time_per_iter_us} us')
print(f'TFLOP/s: {(2*M*N*K*1e-12)/(time_per_iter_us*1e-6)}') # Theoretical max is 4,500 TFLOps for BF16 and 9,000 TFLops for FP8


###
#   Check for correctness
###
print("\nChecking for correctness...")
C = C.to(torch.float32).cpu().numpy()
print(C.shape)
C_ref = (A.to(torch.float16) @ B.to(torch.float16).T).to(torch.float8_e4m3fn)
C_ref = C_ref.to(torch.float32).cpu().numpy()
print(C_ref.shape)
print('Max abs diff:', abs(C-C_ref).max())
print('Mean abs diff:', abs(C-C_ref).mean())
