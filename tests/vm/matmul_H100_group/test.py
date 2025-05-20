import torch
from group_matmul import group_matmul

# Must match that of group_matmul.cu
NUM_GROUPS = 16 

# In the order of M, K, N
# Keep K in multiples of 512 for best performance
GROUP_SHAPES = [
    (3072, 8192, 4096),
    (4096, 16384*3, 8192),
    (8192, 8192, 8192),
    (3072, 16384*2, 3072),
    (8192, 16384, 8192),
    (8192, 8192, 16384),
    (8192, 16384, 8192),
    (4096, 8192, 3072),
    (8192, 16384*3, 8192),
    (16384, 16384, 16384),
    (8192, 8192, 8192),
    (3072, 16384*2, 3072),
    (8192, 16384, 8192),
    (4096, 4096, 4096),
    (16384, 16384, 4096),
    (8192, 16384, 8192),
]

OPCODE = 1
SM_COUNT = 132
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
NUM_ITERS = 10
NUM_WARMUP_ITERS = 3

M_BLOCK = 128
K_BLOCK = 128
N_BLOCK = 256
SUPER_M = 1024 # for better L2 cache utilization

assert len(GROUP_SHAPES) == NUM_GROUPS
for M, K, N in GROUP_SHAPES:
    if M%M_BLOCK != 0: raise ValueError(f'M must be divisible by {M_BLOCK}')
    if K%K_BLOCK != 0: raise ValueError(f'K must be divisible by {K_BLOCK}')
    if N%N_BLOCK != 0: raise ValueError(f'N must be divisible by {N_BLOCK}')

# Create input and output tensors
print('Starting test...')
torch.manual_seed(1)
A = [(torch.randn((M, K), device=0, dtype=torch.float32) / K**.25).to(torch.float8_e4m3fn) for M, K, N in GROUP_SHAPES]
B = [(torch.randn((N, K), device=0, dtype=torch.float32) / K**.25).to(torch.float8_e4m3fn) for M, K, N in GROUP_SHAPES]
C = [torch.zeros((M, N), device=0, dtype=torch.float8_e4m3fn) for M, K, N in GROUP_SHAPES]
print('Input tensors created')
print('A shapes:', [a.shape for a in A])
print('B shapes:', [b.shape for b in B])
print('C shapes:', [c.shape for c in C])

instruction_idx = 0
instructions = [[] for _ in range(SM_COUNT)]
for group_id, (M, K, N) in enumerate(GROUP_SHAPES):
    num_iters = K // K_BLOCK
    for row_outer in range((M + SUPER_M - 1) // SUPER_M): # ceil
        for col in range(N // N_BLOCK):
            rows_per_super = SUPER_M // M_BLOCK
            row_start = rows_per_super * row_outer
            for row_inner in range(rows_per_super):
                row = row_start + row_inner
                if row >= M // M_BLOCK:
                    break
                instructions[instruction_idx%SM_COUNT].append([OPCODE, group_id, row, col, num_iters] + [0]*(INSTRUCTION_WIDTH-5))
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
tflops = sum([2*M*K*N for M, K, N in GROUP_SHAPES]) * 1e-12
print(f'Time per iter: {sec_per_iter * 1e6} us')
print(f'TFLOP/s: {tflops/(sec_per_iter)}')

print('Test completed successfully!')

for group_id in range(NUM_GROUPS):
    C_impl = C[group_id].to(torch.float32).cpu().numpy()
    C_ref = (A[group_id].to(torch.float16) @ B[group_id].to(torch.float16).T).to(torch.float8_e4m3fn).to(torch.float32).cpu().numpy()
    assert C_impl.shape == C_ref.shape
    print(f'Group {group_id} abs diff max:', abs(C_impl - C_ref).max())
    print(f'Group {group_id} abs diff mean:', abs(C_impl - C_ref).mean())
