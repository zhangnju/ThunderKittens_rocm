import torch
from time import time


###
#   Global Parameters
###
NUM_DEVICES = 8
NUM_COMMS = 8 # this is the magic number that works the best
NUM_ITERS = 5
ATTN_OPCODE = 725
COMM_OPCODE = 97
B, H, N, D_h = 8, 8, 128, 64
assert N%NUM_DEVICES==0, "N must be divisible by NUM_DEVICES"


###
#   Prepare Inputs
###
print(f'Starting test with B={B}, H={H}, N={N}, D_h={D_h}, NUM_DEVICES={NUM_DEVICES}')
print('\nGenerating inputs...')
torch.manual_seed(42)
dev_ids = [i for i in range(NUM_DEVICES)]
torch_devices = [torch.device(f"cuda:{dev_id}") for dev_id in dev_ids]
N_per_dev = N // NUM_DEVICES
Qs, K0s, K1s, V0s, V1s = [], [], [], [], []
for torch_device in torch_devices:
    torch.manual_seed(42 + torch_device.index)
    Qs.append(torch.randn((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    K0s.append(torch.randn((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    K1s.append(torch.zeros((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    V0s.append(torch.randn((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    V1s.append(torch.zeros((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))


###
#  Enable P2P access
###
# print('\nEnabling cross-device access...')
# for i in dev_ids:
#     for j in dev_ids:
#         if i != j:
#             assert can_access_peer(i, j), f'Device {i} cannot access device {j}'
#             enable_p2p_access(i, j)


###
#  Launch the PyTorch version
###
def ring_mha_torch(Qs, Ks, Vs):
    As = []
    num_QO_blocks = len(Qs)
    num_KV_blocks = len(Ks)

    for i in range(num_QO_blocks): 
        # "Outer loop". Done in parallel on `num_QO_blocks` devices
        # Qs[i] stay on device i, Ks[i] and Vs[i] are rotated
        torch.cuda.set_device(Qs[i].device)

        # We only need to scale once
        Qi = Qs[i] / (Qs[i].size(-1) ** 0.5)

        # Accumulating variables
        numerator = torch.zeros_like(Qi, device=Qi.device) # (B, H, N_per_dev, D_h)
        denominator = torch.zeros(Qi.shape[:-1], dtype=Qi.dtype, device=Qi.device, layout=Qi.layout) # (B, H, N_per_dev)
        local_max = torch.full(denominator.shape, float('-inf'), dtype=Qi.dtype, device=Qi.device, layout=Qi.layout) # (B, H, N)

        for rotation_idx in range(num_KV_blocks):
            # "Inner loop". Done sequentially on each device. 
            # `num_KV_blocks` ring rotations of Ks and Vs

            # device i starts with Ks[i] and Vs[i]
            j = (i + rotation_idx) % num_KV_blocks
            Kj = Ks[j].to(device=Qi.device) # (B, H, N_per_dev, D_h)
            Vj = Vs[j].to(device=Qi.device) # (B, H, N_per_dev, D_h)

            # Blockwise attention
            QiKj = torch.matmul(Qi, Kj.transpose(-1, -2)) # (B, H, N_per_dev, N_per_dev)
            new_max = torch.max(local_max, torch.max(QiKj, dim=-1).values) # (B, H, N_per_dev)
            exp_QiKj = torch.exp(QiKj - new_max.unsqueeze(-1)) # (B, H, N_per_dev, N_per_dev)
            if rotation_idx > 0:
                rescaler = torch.exp(local_max - new_max)
                numerator *= rescaler.unsqueeze(-1)
                denominator *= rescaler
            numerator += torch.matmul(exp_QiKj, Vj)
            denominator += torch.sum(exp_QiKj, dim=-1)
            local_max = new_max

        # Normalize and store
        Ai = numerator / denominator.unsqueeze(-1) # (B, H, N_per_dev, D_h)
        As.append(Ai)

    return As

As_ref = ring_mha_torch(Qs, K0s, V0s)
torch.cuda.empty_cache()
