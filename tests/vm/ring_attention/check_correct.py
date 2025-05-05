import torch
from ring_attention import ring_attention, can_access_peer, enable_p2p_access
import numpy as np


###
#   Global Parameters
###
NUM_DEVICES = 4
NUM_COMMS = 8 # this is the magic number that works the best
ATTN_OPCODE = 725
COMM_OPCODE = 97
B, H, N, D_h = 16, 16, 256*16, 64

assert N%NUM_DEVICES==0, "N must be divisible by NUM_DEVICES"
assert (N//NUM_DEVICES)%256==0, "N_per_dev must be divisible by 256 (QO Block Size * 2)"
assert D_h==64, "D_h must be 64"

N_per_dev = N // NUM_DEVICES
num_qo_blocks = N_per_dev // 256
num_kv_blocks = N_per_dev // 128
num_comps = B * H * num_qo_blocks
num_ring_stages = NUM_DEVICES


###
#   Prepare Inputs
###
print(f'Starting test with B={B}, H={H}, N={N}, D_h={D_h}, NUM_DEVICES={NUM_DEVICES}')
print('\nGenerating inputs...')
torch.manual_seed(42)
dev_ids = [i for i in range(NUM_DEVICES)]
torch_devices = [torch.device(f"cuda:{dev_id}") for dev_id in dev_ids]
Qs, K0s, K1s, V0s, V1s, Os, Ls, Ms = [], [], [], [], [], [], [], []
Ks, Vs = [], [] # for correctness check
for torch_device in torch_devices:
    torch.manual_seed(42 + torch_device.index)
    Qs.append(torch.randn((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    K0s.append(torch.randn((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    K1s.append(torch.zeros((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    V0s.append(torch.randn((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    V1s.append(torch.zeros((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    Os.append(torch.zeros((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    Ls.append(torch.zeros((B, H, N_per_dev), device=torch_device, dtype=torch.float32))
    Ms.append(torch.zeros((B, H, N_per_dev), device=torch_device, dtype=torch.float32))
    Ks.append(K0s[-1].to('cpu')) # for correctness check
    Vs.append(V0s[-1].to('cpu'))


###
#   Prepare Instructions
###
print('\nGenerating instructions...')
instructions = []
for torch_device in torch_devices:
    dev_idx = torch_device.index
    prev_dev_idx = (dev_idx + NUM_DEVICES - 1) % NUM_DEVICES
    next_dev_idx = (dev_idx + 1) % NUM_DEVICES
    dev_instructions = [[] for _ in range(148)]

    # Comm Ops
    # num_comms_per_kv = NUM_COMMS // 2
    # num_chunks_N = N_per_dev // 128
    # total_chunks = B * H * num_chunks_N
    # num_chunks_per_comm = total_chunks // (num_comms_per_kv)
    # print('total_chunks:', total_chunks)
    # print('num_chunks_per_comm:', num_chunks_per_comm)
    # print('num_chunks_N:', num_chunks_N)
    # assert NUM_COMMS % 2 == 0, "NUM_COMMS must be even"
    # assert total_chunks % (num_comms_per_kv) == 0, "total_chunks must be divisible by NUM_COMMS / 2"
    # num_comps = B * H * num_qo_blocks
    # for i in range(NUM_COMMS):
    #     k_or_v = i // (num_comms_per_kv)
    #     comm_idx = i % (num_comms_per_kv)
    #     dev_instructions[comm_idx].append(
    #         [COMM_OPCODE, k_or_v, num_chunks_per_comm, comm_idx, num_comms_per_kv, num_comps, num_chunks_N, H, dev_idx, prev_dev_idx, next_dev_idx] 
    #         + [0]*21
    #     )

    # Compute Ops
    instruction_idx = 0
    # num_ring_stages = 1 # TESTTETS
    NUM_COMMS = 0
    num_comps = 0
    for ring_stage in range(num_ring_stages):
        for batch_idx in range(B):
            for head_idx in range(H):
                for qo_idx in range(num_qo_blocks):
                    dev_instructions[NUM_COMMS+(instruction_idx%(148-NUM_COMMS))].append(
                        [ATTN_OPCODE, batch_idx, head_idx, qo_idx*2, num_kv_blocks, ring_stage, NUM_COMMS, num_comps, dev_idx]
                        + [0]*23
                    )
                    instruction_idx += 1
    print('Number of QO blocks per device:', num_qo_blocks)
    print('Number of KV blocks per device:', num_kv_blocks)
    print('Number of compute instructions per device:', instruction_idx)

    # Paddings
    max_instruction_len = 0
    for sm_instructions in dev_instructions:
        max_instruction_len = max(max_instruction_len, len(sm_instructions))
    for sm_instructions in dev_instructions:
        while len(sm_instructions) < max_instruction_len:
            sm_instructions.append([0]*32)

    # Append
    instructions.append(torch.tensor(dev_instructions, dtype=torch.int32, device=torch_device))

print(f'Instructions shape: {instructions[0].shape}')


###
#   Prepare Timings and Barriers
###
print('\nGenerating barriers and timings...')
barriers = []
timings = []
for torch_device in torch_devices:
    barriers.append(torch.zeros((NUM_DEVICES,), dtype=torch.uint32, device=torch_device))
    timings.append(torch.zeros((148, instructions[0].shape[1], 128), dtype=torch.int32, device=torch_device))

print(f'Barriers shape: {barriers[0].shape}')
print(f'Timings shape: {timings[0].shape}')


###
#  Enable P2P access
###
print('\nEnabling cross-device access...')
for i in dev_ids:
    for j in dev_ids:
        if i != j:
            assert can_access_peer(i, j), f'Device {i} cannot access device {j}'
            enable_p2p_access(i, j)


###
#  Launch the kernel
###
print('\nLaunching kernel...')
for dev_id in dev_ids:
    barriers[dev_id].zero_()
    torch.cuda.synchronize(dev_id)
ring_attention(
    instructions, barriers, timings,
    Qs, K0s, K1s, V0s, V1s, Os, Ls, Ms
)
for dev_id in dev_ids:
    torch.cuda.synchronize(dev_id)

print('DONE')
# ###
# #  Verify correctness
# ###
# def check_diff(x, y):
#     x = x.to(dtype=torch.float32, device='cpu')
#     y = y.to(dtype=torch.float32, device='cpu')
#     abs_diff = abs(x - y)
#     print('Max abs diff:', abs_diff.max())
#     print('Mean abs diff:', abs_diff.mean())


###
#  Launch the PyTorch version
###
# def pytorch_mha(q, k, v):
#     QK = torch.matmul(q.float(), k.float().transpose(-2, -1))
#     QK /= (q.size(-1) ** 0.5)
#     QK = torch.nn.functional.softmax(QK, dim=-1)
#     out = torch.matmul(QK, v.float()).bfloat16()
#     return out

# for i in range(NUM_DEVICES):
#     O_ref = pytorch_mha(Qs[i], K0s[i], V0s[i])
#     print(torch.max(O_ref - Os[i]))
#     print(torch.mean(O_ref - Os[i]))

# breakpoint()



# def ring_mha_torch(Qs, Ks, Vs):
#     As = []
#     num_QO_blocks = len(Qs)
#     num_KV_blocks = len(Ks)

#     for i in range(num_QO_blocks): 
#         # "Outer loop". Done in parallel on `num_QO_blocks` devices
#         # Qs[i] stay on device i, Ks[i] and Vs[i] are rotated
#         torch.cuda.set_device(Qs[i].device)

#         # We only need to scale once
#         Qi = Qs[i] / (Qs[i].size(-1) ** 0.5)

#         # Accumulating variables
#         numerator = torch.zeros_like(Qi, device=Qi.device) # (B, H, N_per_dev, D_h)
#         denominator = torch.zeros(Qi.shape[:-1], dtype=Qi.dtype, device=Qi.device, layout=Qi.layout) # (B, H, N_per_dev)
#         local_max = torch.full(denominator.shape, float('-inf'), dtype=Qi.dtype, device=Qi.device, layout=Qi.layout) # (B, H, N)

#         for rotation_idx in range(num_KV_blocks):
#             # "Inner loop". Done sequentially on each device. 
#             # `num_KV_blocks` ring rotations of Ks and Vs

#             # device i starts with Ks[i] and Vs[i]
#             j = (i + rotation_idx) % num_KV_blocks
#             Kj = Ks[j].to(device=Qi.device) # (B, H, N_per_dev, D_h)
#             Vj = Vs[j].to(device=Qi.device) # (B, H, N_per_dev, D_h)

#             # Blockwise attention
#             QiKj = torch.matmul(Qi, Kj.transpose(-1, -2)) # (B, H, N_per_dev, N_per_dev)
#             new_max = torch.max(local_max, torch.max(QiKj, dim=-1).values) # (B, H, N_per_dev)
#             exp_QiKj = torch.exp(QiKj - new_max.unsqueeze(-1)) # (B, H, N_per_dev, N_per_dev)
#             if rotation_idx > 0:
#                 rescaler = torch.exp(local_max - new_max)
#                 numerator *= rescaler.unsqueeze(-1)
#                 denominator *= rescaler
#             numerator += torch.matmul(exp_QiKj, Vj)
#             denominator += torch.sum(exp_QiKj, dim=-1)
#             local_max = new_max

#         # Normalize and store
#         Ai = numerator / denominator.unsqueeze(-1) # (B, H, N_per_dev, D_h)
#         As.append(Ai)

#     return As

# As_ref = ring_mha_torch(Qs, K0s, V0s)
# torch.cuda.empty_cache()
