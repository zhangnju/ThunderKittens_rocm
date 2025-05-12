import math
import torch
from einops import einsum, rearrange
from kvm_llama import kvm_llama

TORCH_DEVICE = torch.device('cuda:2')

NUM_SMS = 148
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
GQA_ATTENTION_OPCODE = 3
NUM_OPS = 8
KV_PAGE_SIZE = 16

N_max = 131072
H_q = 64
H_kv = 8
D_h = 128

# Kernel parameters
LAYER_IDX = 3
ATTN_SCALE = 1 / math.sqrt(D_h)
ATTN_BLOCK_SIZE = 16
BATCH_SIZE = 128

# Unit testing
BATCH_IDX = 0
POS_ID = 15
KV_H_IDX = 0

def generate_tensor_inputs():
    '''
    Generate tensor inputs for the attention kernel.
    '''
    torch.manual_seed(42)

    # Calculate number of pages needed
    num_pages = (N_max + KV_PAGE_SIZE - 1) // KV_PAGE_SIZE

    Q   = torch.randn(H_q * D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    O   = torch.zeros(H_q * D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)

    # KV cache in paged format: (max_num_pages, head_idx, page_size, head_dim)
    K_c = torch.randn(num_pages, H_kv, KV_PAGE_SIZE, D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)
    V_c = torch.randn(num_pages, H_kv, KV_PAGE_SIZE, D_h, dtype=torch.bfloat16, device=TORCH_DEVICE)

    num_pages_needed = (POS_ID + 1 + KV_PAGE_SIZE - 1) // KV_PAGE_SIZE

    kv_indices = []
    for page in range(num_pages_needed):
        kv_indices.append(page)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32, device=TORCH_DEVICE)
    
    kv_indptr = torch.zeros(BATCH_SIZE, dtype=torch.int32, device=TORCH_DEVICE)
    kv_indptr[BATCH_IDX] = 0

    return Q, K_c, V_c, O, kv_indices, kv_indptr

def generate_itb(): # instruction, timings, barriers
    instructions = [[] for _ in range(NUM_SMS)]
    instruction_idx = 0

    # Single instruction, for testing (LayerIdx, H_kv index, BatchIdx)
    instruction = [GQA_ATTENTION_OPCODE, LAYER_IDX, KV_H_IDX, BATCH_IDX]
    instruction += [0] * (INSTRUCTION_WIDTH - len(instruction))  # Pad to INSTRUCTION_WIDTH
    instructions[0].append(instruction)
    instruction_idx += 1

    # All SMs must have same number of instructions
    max_instructions = -1
    for i in range(NUM_SMS):
        max_instructions = max(max_instructions, len(instructions[i]))
    for i in range(NUM_SMS):
        while len(instructions[i]) < max_instructions:
            instructions[i].append([0] * INSTRUCTION_WIDTH)
        instruction_idx += 1

    # (BlockIdx, InstructionIdx, Instruction)
    # If opcode (instructions[:, :, 0]) is invalid, the instruction is ignored
    instructions = torch.tensor(instructions, dtype=torch.int32).to(device=TORCH_DEVICE)
    timings = torch.zeros((NUM_SMS, instruction_idx // NUM_SMS, TIMING_WIDTH), dtype=torch.int32).to(device=TORCH_DEVICE)
    barriers = torch.zeros((7, NUM_OPS, H_q + 2 * H_kv), dtype=torch.uint32).to(device=TORCH_DEVICE)

    # Fill in the barrier
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, KV_H_IDX * 8 + 0] = 8
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, KV_H_IDX * 8 + 1] = 8
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, KV_H_IDX * 8 + 2] = 8
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, KV_H_IDX * 8 + 3] = 8
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, KV_H_IDX * 8 + 4] = 8
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, KV_H_IDX * 8 + 5] = 8
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, KV_H_IDX * 8 + 6] = 8
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, KV_H_IDX * 8 + 7] = 8
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, H_q + KV_H_IDX] = 8
    barriers[LAYER_IDX, GQA_ATTENTION_OPCODE - 1, H_q + H_kv + KV_H_IDX] = 8

    return instructions, timings, barriers

def reference_gqa(K_c, V_c, Q_post_rope, kv_indices, kv_indptr, pos_id, attn_scale):
    kv_page_start = kv_indptr[BATCH_IDX]
    num_kv_pages = (pos_id + 1 + KV_PAGE_SIZE - 1) // KV_PAGE_SIZE

    gqa_ratio = H_q // H_kv

    running_max   = torch.full((gqa_ratio,), -float("inf"), device=TORCH_DEVICE)
    running_denom = torch.zeros((gqa_ratio,),               device=TORCH_DEVICE)
    running_out   = torch.zeros((gqa_ratio, D_h),           device=TORCH_DEVICE)

    q = rearrange(Q_post_rope, '(h d) -> h d', h=H_q)
    q_slice = q[KV_H_IDX * 8: (KV_H_IDX + 1) * 8].float()

    for i in range(num_kv_pages):
        cur_page_idx = kv_indices[kv_page_start + i]
        k = K_c[cur_page_idx, KV_H_IDX, :, :].float()
        v = V_c[cur_page_idx, KV_H_IDX, :, :].float()
        
        qk = einsum(q_slice, k, 'q d, k d -> q k')
        scaled_qk = qk * attn_scale

        block_max = scaled_qk.amax(dim=1)
        new_max = torch.maximum(running_max, block_max)

        exp_old = torch.exp(running_max - new_max)
        exp_block = torch.exp(scaled_qk - new_max.unsqueeze(1))

        num_block = torch.matmul(exp_block, v)
        running_out = running_out * exp_old.unsqueeze(1) + num_block

        denom_block = exp_block.sum(dim=1)
        running_denom = running_denom * exp_old + denom_block

        running_max = new_max

    out = running_out / running_denom.unsqueeze(1)
    out_flat = out.view(-1)

    return out_flat

# Generate inputs
print('\nGenerating inputs...')
instructions, timings, barriers = generate_itb()
Q_post_rope, K_c, V_c, Attn_out, kv_indices, kv_indptr = generate_tensor_inputs()

# Run the kernel
print('Instruction shape:', instructions.shape)
print('Barrier shape:', barriers.shape)
print('Timings shape:', timings.shape) 
print('Q_post_rope shape:', Q_post_rope.shape)
print('K_c shape:', K_c.shape)
print('V_c shape:', V_c.shape)
print('Attn_out shape:', Attn_out.shape)


print('\nRunning the kernel...')

kvm_llama(
    barriers, 
    instructions, 
    timings,
    K_c, 
    V_c, 
    Q_post_rope,
    Attn_out, 
    kv_indices,
    kv_indptr,
    POS_ID, 
    ATTN_SCALE
)
torch.cuda.synchronize(TORCH_DEVICE)


attn_out_ref = reference_gqa(K_c, V_c, Q_post_rope, kv_indices, kv_indptr, POS_ID, ATTN_SCALE)
kvm_out_slice = Attn_out[KV_H_IDX * D_h: (KV_H_IDX + 8) * D_h]

print(attn_out_ref)
print(kvm_out_slice)

# Compare reference and kernel outputs
abs_diff = torch.abs(attn_out_ref - kvm_out_slice)

print("\nOutput comparison:")
print(f"Max absolute difference: {abs_diff.max().item():.6f}")
print(f"Mean absolute difference: {abs_diff.mean().item():.6f}")

