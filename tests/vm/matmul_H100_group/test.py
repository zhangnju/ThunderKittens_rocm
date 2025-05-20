import torch
from group_matmul import group_matmul

# M, K, N = 3072, 4096, 3072
# M, K, N = 512, 256, 256
# M, K, N = 8192, 16384*3, 8192
# M, K, N = 16384, 3072, 16384
# M, K, N = 8192, 8192, 8192
# M, K, N = 3072, 16384*2, 3072
# M, K, N = 256, 4096, 256
M, K, N = 8192, 16384, 8192

OPCODE = 1
SM_COUNT = 132
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
NUM_ITERS = 10
NUM_WARMUP_ITERS = 2

NUM_GROUPS = 16 # must match that of group_matmul.cu
M_BLOCK = 128
K_BLOCK = 128
N_BLOCK = 256

if M%M_BLOCK != 0: raise ValueError(f'M must be divisible by {M_BLOCK}')
if K%K_BLOCK != 0: raise ValueError(f'K must be divisible by {K_BLOCK}')
if N%N_BLOCK != 0: raise ValueError(f'N must be divisible by {N_BLOCK}')

print('Starting test...')

# Create input and output tensors
torch.manual_seed(1)
A = (torch.randn((M, K), device=0, dtype=torch.float32) / K**.25).to(torch.float8_e4m3fn)
B = (torch.randn((N, K), device=0, dtype=torch.float32) / K**.25).to(torch.float8_e4m3fn)
C =  torch.zeros((M, N), device=0, dtype=torch.float8_e4m3fn)
print('Input tensors created, of shapes', A.shape, B.shape, C.shape)

A = [A for _ in range(NUM_GROUPS)]
B = [B for _ in range(NUM_GROUPS)]
C = [C for _ in range(NUM_GROUPS)]

instructions = [[] for _ in range(SM_COUNT)]
SUPER_M = 1024
num_iters = K // K_BLOCK
instruction_idx = 0
for row_outer in range((M + SUPER_M - 1) // SUPER_M): # ceil
    for col in range(N // N_BLOCK):
        rows_per_super = SUPER_M // M_BLOCK
        row_start = rows_per_super * row_outer
        for row_inner in range(rows_per_super):
            row = row_start + row_inner
            if row >= M // M_BLOCK:
                break
            instructions[instruction_idx%SM_COUNT].append([OPCODE, row, col, num_iters] + [0]*(INSTRUCTION_WIDTH-4))
            instruction_idx += 1

# Pad instructions
max_instructions = -1
for i in range(SM_COUNT):
    max_instructions = max(max_instructions, len(instructions[i]))
for i in range(SM_COUNT):
    while len(instructions[i]) < max_instructions:
        instructions[i].append([0] * INSTRUCTION_WIDTH)

instructions = torch.tensor(instructions, dtype=torch.int32, device=0)
timings = torch.zeros((SM_COUNT, instructions.shape[1], TIMING_WIDTH), dtype=torch.int32, device=0)
print(f'Instruction and timing tensors created, of shapes {instructions.shape} and {timings.shape}')

# Run the group_matmul kernel
print('Launching kernel...')
group_matmul(instructions, timings, A, B, C)
torch.cuda.synchronize()

print('Starting timing loop...')
for i in range(NUM_WARMUP_ITERS):
    group_matmul(instructions, timings, A, B, C)
torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for i in range(NUM_ITERS):
    group_matmul(instructions, timings, A, B, C)
torch.cuda.synchronize()
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
sec_per_iter = ((elapsed_time / 1000) / NUM_ITERS)
print(f'Time per iter: {sec_per_iter * 1e6} us')
print(f'TFLOP/s: {(2*M*N*K*1e-12)/(sec_per_iter)}')

print('Test completed successfully!')

C = C[0].to(torch.float32).cpu().numpy()
C_ref = (A[0].to(torch.float16) @ B[0].to(torch.float16).T).to(torch.float8_e4m3fn)
C_ref = C_ref.to(torch.float32).cpu().numpy()
assert C.shape == C_ref.shape
print('abs diff max:', abs(C - C_ref).max())
print('abs diff mean:', abs(C - C_ref).mean())
