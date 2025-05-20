import torch
# from grouped_gemm_backward_fp8 import grouped_gemm_backward_fp8

# Must match that of grouped_gemm_backward_fp8.cu
NUM_GROUPS = 16 

NUM_EXPERTS = 64
NUM_TOKENS = 387968
HIDDEN_DIM = 2048
INTERMEDIATE_DIM = 1408
SCALE_BLOCK_SIZE = 128
EXPERT_ROUTING_BLOCK_SIZE = 128

OPCODE = 1
NUM_TARGETS = 3
SM_COUNT = 132
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
NUM_ITERS = 10
NUM_WARMUP_ITERS = 3

M_BLOCK = 128
K_BLOCK = 128
N_BLOCK = 256
SUPER_M = 1024 # for better L2 cache utilization

# Create input and output tensors
print('Starting test...')
torch.manual_seed(1)
m_indices = torch.zeros(NUM_TOKENS, dtype=torch.int32, device=0)
x_fp8 = torch.rand((NUM_TOKENS, HIDDEN_DIM), dtype=torch.bfloat16, device=0).to(dtype=torch.float8_e4m3fn)
x_sc = torch.rand((NUM_TOKENS, HIDDEN_DIM // SCALE_BLOCK_SIZE), dtype=torch.float32, device=0)
dgate_fp8 = torch.rand((NUM_TOKENS, INTERMEDIATE_DIM), dtype=torch.bfloat16, device=0).to(dtype=torch.float8_e4m3fn)
dgate_sc = torch.rand((NUM_TOKENS, INTERMEDIATE_DIM // SCALE_BLOCK_SIZE), dtype=torch.float32, device=0)
dup_fp8 = torch.rand(( NUM_TOKENS, INTERMEDIATE_DIM), dtype=torch.bfloat16, device=0).to(dtype=torch.float8_e4m3fn)
dup_sc = torch.rand((NUM_TOKENS, INTERMEDIATE_DIM // SCALE_BLOCK_SIZE), dtype=torch.float32, device=0)
hidden_fp8 = torch.rand((NUM_TOKENS, INTERMEDIATE_DIM), dtype=torch.bfloat16, device=0).to(dtype=torch.float8_e4m3fn)
hidden_sc = torch.rand((NUM_TOKENS, INTERMEDIATE_DIM // SCALE_BLOCK_SIZE), dtype=torch.float32, device=0)
dy_fp8 = torch.rand((NUM_TOKENS, HIDDEN_DIM), dtype=torch.bfloat16, device=0).to(dtype=torch.float8_e4m3fn)
dy_sc = torch.rand((NUM_TOKENS, HIDDEN_DIM // SCALE_BLOCK_SIZE), dtype=torch.float32, device=0)
dw1_param_view = torch.zeros((NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM), dtype=torch.bfloat16, device=0).to(dtype=torch.float8_e4m3fn)
dv1_param_view = torch.zeros((NUM_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM), dtype=torch.bfloat16, device=0).to(dtype=torch.float8_e4m3fn)
dw2_param_view = torch.zeros((NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM), dtype=torch.bfloat16, device=0).to(dtype=torch.float8_e4m3fn)

# Fill up m_indices
i = 6144; m_indices[i-6144:i] =  0
i += 6144; m_indices[i-6144:i] =  1
i += 6144; m_indices[i-6144:i] =  2
i += 5632; m_indices[i-5632:i] =  3
i += 6144; m_indices[i-6144:i] =  4
i += 6144; m_indices[i-6144:i] =  5
i += 6016; m_indices[i-6016:i] =  6
i += 6272; m_indices[i-6272:i] =  7
i += 6272; m_indices[i-6272:i] =  8
i += 6016; m_indices[i-6016:i] =  9
i += 6144 ; m_indices[i-6144:i] = 10
i += 6016 ; m_indices[i-6016:i] = 11
i += 6272 ; m_indices[i-6272:i] = 12
i += 6272 ; m_indices[i-6272:i] = 13
i += 6144 ; m_indices[i-6144:i] = 14
i += 6144 ; m_indices[i-6144:i] = 15
i += 6272 ; m_indices[i-6272:i] = 16
i += 5888 ; m_indices[i-5888:i] = 17
i += 6016 ; m_indices[i-6016:i] = 18
i += 5760 ; m_indices[i-5760:i] = 19
i += 6144 ; m_indices[i-6144:i] = 20
i += 5760 ; m_indices[i-5760:i] = 21
i += 6016 ; m_indices[i-6016:i] = 22
i += 5760 ; m_indices[i-5760:i] = 23
i += 6016 ; m_indices[i-6016:i] = 24
i += 5888 ; m_indices[i-5888:i] = 25
i += 5760 ; m_indices[i-5760:i] = 26
i += 6144 ; m_indices[i-6144:i] = 27
i += 6016 ; m_indices[i-6016:i] = 28
i += 5760 ; m_indices[i-5760:i] = 29
i += 6528 ; m_indices[i-6528:i] = 30
i += 5888 ; m_indices[i-5888:i] = 31
i += 5888 ; m_indices[i-5888:i] = 32
i += 6272 ; m_indices[i-6272:i] = 33
i += 6016 ; m_indices[i-6016:i] = 34
i += 6528 ; m_indices[i-6528:i] = 35
i += 6144 ; m_indices[i-6144:i] = 36
i += 5888 ; m_indices[i-5888:i] = 37
i += 5632 ; m_indices[i-5632:i] = 38
i += 6016 ; m_indices[i-6016:i] = 39
i += 5888 ; m_indices[i-5888:i] = 40
i += 6528 ; m_indices[i-6528:i] = 41
i += 6144 ; m_indices[i-6144:i] = 42
i += 6016 ; m_indices[i-6016:i] = 43
i += 6144 ; m_indices[i-6144:i] = 44
i += 6016 ; m_indices[i-6016:i] = 45
i += 6272 ; m_indices[i-6272:i] = 46
i += 5888 ; m_indices[i-5888:i] = 47
i += 6144 ; m_indices[i-6144:i] = 48
i += 6144 ; m_indices[i-6144:i] = 49
i += 6144 ; m_indices[i-6144:i] = 50
i += 6016 ; m_indices[i-6016:i] = 51
i += 6016 ; m_indices[i-6016:i] = 52
i += 6144 ; m_indices[i-6144:i] = 53
i += 6144 ; m_indices[i-6144:i] = 54
i += 6144 ; m_indices[i-6144:i] = 55
i += 6144 ; m_indices[i-6144:i] = 56
i += 6016 ; m_indices[i-6016:i] = 57
i += 6016 ; m_indices[i-6016:i] = 58
i += 6016 ; m_indices[i-6016:i] = 59
i += 6016 ; m_indices[i-6016:i] = 60
i += 6016 ; m_indices[i-6016:i] = 61
i += 6144 ; m_indices[i-6144:i] = 62
i += 5888 ; m_indices[i-5888:i] = 63

print('Input tensors created')
print('x_fp8 shape:', x_fp8.shape)
print('x_sc shape:', x_sc.shape)
print('dgate_fp8 shape:', dgate_fp8.shape)
print('dgate_sc shape:', dgate_sc.shape)
print('dup_fp8 shape:', dup_fp8.shape)
print('dup_sc shape:', dup_sc.shape)
print('hidden_fp8 shape:', hidden_fp8.shape)
print('hidden_sc shape:', hidden_sc.shape)
print('dy_fp8 shape:', dy_fp8.shape)
print('dy_sc shape:', dy_sc.shape)
print('dw1_param_view shape:', dw1_param_view.shape)
print('dv1_param_view shape:', dv1_param_view.shape)
print('dw2_param_view shape:', dw2_param_view.shape)

instruction_idx = 0
instructions = [[] for _ in range(SM_COUNT)]

# Assume that expert routing is done in a block-wise manner
boundaries = torch.nonzero(torch.diff(m_indices) != 0).squeeze() + 1
indices = torch.cat([torch.tensor([0], device=0), boundaries, torch.tensor([m_indices.shape[0]], device=0)])

for target in range(NUM_TARGETS):
    reduction_idx = 0
    for i in range(0, indices.shape[0] - 1):
        start_idx = indices[i].item()
        end_idx = indices[i+1].item()
        if target == 0: # dgate_fp8.T @ x_fp8
            num_row_blocks = INTERMEDIATE_DIM // M_BLOCK
            num_col_blocks = HIDDEN_DIM // N_BLOCK
            num_iters = (end_idx - start_idx) // K_BLOCK
        elif target == 1: # dup_fp8.T @ x_fp8
            num_row_blocks = INTERMEDIATE_DIM // M_BLOCK
            num_col_blocks = HIDDEN_DIM // N_BLOCK
            num_iters = (end_idx - start_idx) // K_BLOCK
        elif target == 2: # dy_fp8.T @ hidden_fp8
            num_row_blocks = HIDDEN_DIM // M_BLOCK
            num_col_blocks = INTERMEDIATE_DIM // N_BLOCK
            num_iters = (end_idx - start_idx) // K_BLOCK

        for row in range(num_row_blocks):
            for col in range(num_col_blocks):
                instructions[instruction_idx%SM_COUNT].append([OPCODE, target, row, col, reduction_idx, num_iters] + [0]*(INSTRUCTION_WIDTH-6))
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

# Run the grouped_gemm_backward_fp8 kernel
print('Launching kernel...')
# grouped_gemm_backward_fp8(
#     instructions, 
#     timings, 
#     x_fp8, 
#     x_sc, 
#     dgate_fp8, 
#     dgate_sc, 
#     dup_fp8, 
#     dup_sc, 
#     hidden_fp8, 
#     hidden_sc, 
#     dy_fp8, 
#     dy_sc, 
#     dw1_param_view, 
#     dv1_param_view, 
#     dw2_param_view
# )
torch.cuda.synchronize()

# print('Starting timing loop...')
# for i in range(NUM_WARMUP_ITERS):
#     grouped_gemm_backward_fp8(instructions, timings, A, B, C)
# torch.cuda.synchronize()
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
# start_event.record()
# for i in range(NUM_ITERS):
#     grouped_gemm_backward_fp8(instructions, timings, A, B, C)
# torch.cuda.synchronize()
# end_event.record()
# torch.cuda.synchronize()
# elapsed_time = start_event.elapsed_time(end_event)
# sec_per_iter = ((elapsed_time / 1000) / NUM_ITERS)
# tflops = sum([2*M*K*N for M, K, N in GROUP_SHAPES]) * 1e-12
# print(f'Time per iter: {sec_per_iter * 1e6} us')
# print(f'TFLOP/s: {tflops/(sec_per_iter)}')

# Check correctness
print('Checking correctness...')
dw1_param_view_ref = torch.zeros_like(dw1_param_view)
dv1_param_view_ref = torch.zeros_like(dv1_param_view)
dw2_param_view_ref = torch.zeros_like(dw2_param_view)

def _dequantize_fp8(t_fp8: torch.Tensor, scales: torch.Tensor, block_size: int = SCALE_BLOCK_SIZE):
    """simple dequantization that broadcasts the per-block scales."""
    if t_fp8.numel() == 0:
        return t_fp8.to(torch.bfloat16)
    scale_broadcast = scales.repeat_interleave(block_size, dim=-1)
    return (t_fp8.to(torch.float32) * scale_broadcast).to(torch.bfloat16)

for expert_idx in range(NUM_EXPERTS):
    token_mask = (m_indices == expert_idx)
    if token_mask.sum() == 0:
        continue

    x_e = _dequantize_fp8(x_fp8[token_mask], x_sc[token_mask])
    dgate_e = _dequantize_fp8(dgate_fp8[token_mask], dgate_sc[token_mask])
    dup_e = _dequantize_fp8(dup_fp8[token_mask], dup_sc[token_mask])
    hid_e = _dequantize_fp8(hidden_fp8[token_mask], hidden_sc[token_mask])
    dy_e = _dequantize_fp8(dy_fp8[token_mask], dy_sc[token_mask])

    dw1_local = torch.matmul(dgate_e.t(), x_e).to(torch.float8_e4m3fn)
    dv1_local = torch.matmul(dup_e.t(), x_e).to(torch.float8_e4m3fn)
    dw2_local = torch.matmul(dy_e.t(), hid_e).to(torch.float8_e4m3fn)

    # store into param-aligned views, transposing if necessary to match shape
    if dw1_local.shape != dw1_param_view[expert_idx].shape:
        dw1_local = dw1_local.t()
    if dv1_local.shape != dv1_param_view[expert_idx].shape:
        dv1_local = dv1_local.t()
    if dw2_local.shape != dw2_param_view[expert_idx].shape:
        dw2_local = dw2_local.t()

    dw1_param_view_ref[expert_idx].copy_(dw1_local)
    dv1_param_view_ref[expert_idx].copy_(dv1_local)
    dw2_param_view_ref[expert_idx].copy_(dw2_local)

print('dw1_param_view_ref shape:', dw1_param_view_ref.shape)
print('dv1_param_view_ref shape:', dv1_param_view_ref.shape)
print('dw2_param_view_ref shape:', dw2_param_view_ref.shape)

print(dw1_param_view_ref)
print(dv1_param_view_ref)
print(dw2_param_view_ref)
