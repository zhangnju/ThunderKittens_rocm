import torch
from matmul import matmul
import sys
import time

# M, K, N = 3072, 4096, 3072
# M, K, N = 512, 256, 256
# M, K, N = 8192, 16384*3, 8192
# M, K, N = 16384, 3072, 16384
# M, K, N = 8192, 8192, 8192
# M, K, N = 3072, 16384*2, 3072
# M, K, N = 256, 4096, 256
M, K, N = 4096, 4096, 4096

SM_COUNT = 132
M_BLOCK = 128
K_BLOCK = 128
N_BLOCK = 256

if M%M_BLOCK != 0: raise ValueError("M must be divisible by 256")
if K%K_BLOCK != 0: raise ValueError("K must be divisible by 128")
if N%N_BLOCK != 0: raise ValueError("N must be divisible by 256")

print("Starting test...")

# Create input and output tensors
torch.manual_seed(1)
A = (torch.randn((M, K), device=0, dtype=torch.float32) / K**.25).to(torch.float8_e4m3fn)
B = (torch.randn((N, K), device=0, dtype=torch.float32) / K**.25).to(torch.float8_e4m3fn)
C =  torch.zeros((M, N), device=0, dtype=torch.float8_e4m3fn)

print("Input tensors created, of shapes", A.shape, B.shape, C.shape)
sys.stdout.flush()

arr = [[] for _ in range(SM_COUNT)]
SUPER_M = 3072
instruction_idx = 0
for i in range((M+SUPER_M-1)//SUPER_M): # ceil
    for col in range(N//256):
        for k in range(SUPER_M//256):
            row = (SUPER_M//256)*i + k
            if row >= M//256:
                break
            arr[instruction_idx%SM_COUNT].append([1, 2*row, 2*col, K//128]+[0]*28)
            instruction_idx += 1
while instruction_idx%SM_COUNT != 0:
    arr[instruction_idx%SM_COUNT].append([0]*32)
    instruction_idx += 1

instructions = torch.tensor(arr, dtype=torch.int32).to(0)
timings = torch.zeros((SM_COUNT, instruction_idx//SM_COUNT, 128), dtype=torch.int32).to(0)

print(f"Instruction and timing tensors created, of shapes {instructions.shape} and {timings.shape}")
sys.stdout.flush()

# Run the matmul kernel
print("Launching kernel...")
matmul(instructions, timings, A, B, C)
torch.cuda.synchronize()

print('Starting timing loop...')
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for i in range(5):
    matmul(instructions, timings, A, B, C)
torch.cuda.synchronize()
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
t1 = time.time()  # Keep this for compatibility with the time_per_iter calculation
t0 = t1 - (elapsed_time / 1000)  # Convert ms to seconds
time_per_iter = ((t1-t0)*1e6)/5
print(f'Time per iter: {time_per_iter} us')
print(f'TFLOP/s: {(2*M*N*K*1e-12)/(time_per_iter*1e-6)}')

print("Test completed successfully!")

C = C.to(torch.float32).cpu().numpy()
print(C.shape)

C2 = (A.to(torch.float16)@B.to(torch.float16).T).to(torch.float8_e4m3fn)
C2 = C2.to(torch.float32).cpu().numpy()
print(C2.shape)

print('abs diff max:', abs(C-C2).max())
print('abs diff mean:', abs(C-C2).mean())
