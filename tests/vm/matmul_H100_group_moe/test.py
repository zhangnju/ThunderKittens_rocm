import torch
from group_matmul import group_matmul

# Must match that of group_matmul.cu

# Keep K in multiples of 512 for best performance
# For now, K must be >= 2048
NUM_EP = 64
M = 1408
K = 65536
N = 2048

OPCODE = 1
SM_COUNT = 132
NUM_ITERS = 10
NUM_WARMUP_ITERS = 3

M_BLOCK = 128
K_BLOCK = 128
N_BLOCK = 256
SCALE_BLOCK = 128

if M%M_BLOCK != 0: raise ValueError(f'M must be divisible by {M_BLOCK}')
if K%K_BLOCK != 0: raise ValueError(f'K must be divisible by {K_BLOCK}')
if N%N_BLOCK != 0: raise ValueError(f'N must be divisible by {N_BLOCK}')

# Create input and output tensors
print('Starting test...')
torch.manual_seed(1)
A = (torch.randn((M, K), device=0, dtype=torch.float32) / K**.25).to(torch.float8_e4m3fn)
A_scale = torch.randn((M//SCALE_BLOCK, K), device=0, dtype=torch.float32)
B = (torch.randn((N, K), device=0, dtype=torch.float32) / K**.25).to(torch.float8_e4m3fn)
B_scale = torch.randn((N//SCALE_BLOCK, K), device=0, dtype=torch.float32)
C = torch.zeros((NUM_EP, M, N), device=0, dtype=torch.float32)
tokens_per_ep = torch.zeros((NUM_EP,), device=0, dtype=torch.int32)
print('Input tensors created')
print('A shape:', A.shape)
print('A_scale shape:', A_scale.shape)
print('B shape:', B.shape)
print('B_scale shape:', B_scale.shape)
print('C shape:', C.shape)
print('tokens_per_ep shape:', tokens_per_ep.shape)

# Set up tokens_per_ep
tokens_left = K
for group_id in range(NUM_EP):
    tokens_per_ep[group_id] = 1024
    tokens_left -= 1024
assert(tokens_left == 0)
print('tokens_per_ep:', tokens_per_ep)

# Run the group_matmul kernel
print('Launching kernel...')
group_matmul(A, A_scale, B, B_scale, C, tokens_per_ep)
torch.cuda.synchronize()

print('Starting timing loop...')
for i in range(NUM_WARMUP_ITERS):
    group_matmul(A, A_scale, B, B_scale, C, tokens_per_ep)
    torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for i in range(NUM_ITERS):
    group_matmul(A, A_scale, B, B_scale, C, tokens_per_ep)
    torch.cuda.synchronize()
end_event.record()
torch.cuda.synchronize()
print('Elapsed time:', start_event.elapsed_time(end_event))
elapsed_time = start_event.elapsed_time(end_event)
sec_per_iter = ((elapsed_time / 1000) / NUM_ITERS)
tflops = 2*M*K*N * 1e-12
print(f'Time per iter: {sec_per_iter * 1e6} us')
print(f'TFLOP/s: {tflops/(sec_per_iter)}')

print('Test completed successfully!')

token_index = 0
for group_id in range(NUM_EP):
    C_impl = C[group_id].cpu().numpy()
    C_ref = (
        A[:, token_index:token_index+tokens_per_ep[group_id]].to(torch.float16) @ 
        B[:, token_index:token_index+tokens_per_ep[group_id]].to(torch.float16).T
    ).to(torch.float32).cpu().numpy()
    token_index += tokens_per_ep[group_id]
    assert C_impl.shape == C_ref.shape
    print(f'Group {group_id} abs diff max:', abs(C_impl - C_ref).max())
    print(f'Group {group_id} abs diff mean:', abs(C_impl - C_ref).mean())
