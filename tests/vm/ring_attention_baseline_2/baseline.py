import torch
import torch.distributed
import os

# Flash attention: https://github.com/Dao-AILab/flash-attention (used both in reference & ring attention)
from flash_attn import flash_attn_func

# Baseline repo: https://github.com/xdit-project/xDiT
from yunchang.kernels import AttnType
from xfuser.core.long_ctx_attention.ring.ring_flash_attn import xdit_ring_flash_attn_func
from xfuser.core.distributed import init_distributed_environment


###
#   Global Parameters
###
NUM_DEVICES = int(os.environ['WORLD_SIZE'])
CURRENT_DEVICE = int(os.environ['RANK'])
NUM_ITERS = 1
NUM_WARMUPS = 0
B, H, N, D_h = 1, 16, 4096*NUM_DEVICES, 128
CHECK_CORRECT = True

assert os.environ['RANK']==os.environ['LOCAL_RANK'], 'Must be run on single node'
assert NUM_DEVICES>=1, 'NUM_DEVICES must be >= 1'
assert N%NUM_DEVICES==0, 'N must be divisible by NUM_DEVICES'


###
#   Prepare CUDA environment
###
world_size = NUM_DEVICES
rank = CURRENT_DEVICE
print(f'Process started with rank (device ID) {rank} and world size {world_size}.')
torch_device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(torch_device)
init_distributed_environment(rank=rank, world_size=world_size, local_rank=-1, backend='nccl')


###
#   Prepare Inputs
###
print(f'Rank {rank}: Starting test with B={B}, H={H}, N={N}, D_h={D_h}, NUM_DEVICES={NUM_DEVICES}')
print(f'Rank {rank}: Generating inputs...')
torch.manual_seed(42+rank)
if rank == 0:
    Q = torch.randn((B, N, H, D_h), device=torch_device, dtype=torch.bfloat16, requires_grad=False)
    K = torch.randn((B, N, H, D_h), device=torch_device, dtype=torch.bfloat16, requires_grad=False)
    V = torch.randn((B, N, H, D_h), device=torch_device, dtype=torch.bfloat16, requires_grad=False)
else:
    Q = torch.empty((B, N, H, D_h), device=torch_device, dtype=torch.bfloat16, requires_grad=False)
    K = torch.empty((B, N, H, D_h), device=torch_device, dtype=torch.bfloat16, requires_grad=False)
    V = torch.empty((B, N, H, D_h), device=torch_device, dtype=torch.bfloat16, requires_grad=False)

print(f'Rank {rank}: Sharding inputs...')
torch.distributed.broadcast(Q, src=0)
torch.distributed.broadcast(K, src=0)
torch.distributed.broadcast(V, src=0)
local_Q = Q.chunk(world_size, dim=1)[rank]
local_K = K.chunk(world_size, dim=1)[rank]
local_V = V.chunk(world_size, dim=1)[rank]


###
#   Run the baseline ring attention
###
print(f'Rank {rank}: Running baseline ring attention...')
O = xdit_ring_flash_attn_func(
    q=local_Q,
    k=local_K,
    v=local_V,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
    group=torch.distributed.group.WORLD,
    alibi_slopes=None,
    deterministic=True,
    return_attn_probs=False,
    attn_type=AttnType.FA,
    attn_processor=None,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy='none'
)


###
#  Verify correctness
###
def pytorch_mha(q, k, v):
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= (q.size(-1) ** 0.5)
    QK = torch.nn.functional.softmax(QK, dim=-1)
    out = torch.matmul(QK, v)
    return out

def check_diff(x, y):
    abs_diff = abs(x - y)
    print('Max abs diff:', abs_diff.max())
    print('Mean abs diff:', abs_diff.mean())

if CHECK_CORRECT:
    print(f'Rank {rank}: Verifying correctness...')
    O_ref = pytorch_mha(Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2))
    O_ref = O_ref.transpose(1, 2)
    O_ref = O_ref.chunk(world_size, dim=1)[rank]

    assert O.shape==O_ref.shape, 'Output shapes do not match'
    for i in range(NUM_DEVICES):
        if i == rank:
            check_diff(O, O_ref)
        torch.distributed.barrier()


###
#  Clean up
###
print(f'Rank {rank}: Test complete.')
torch.distributed.destroy_process_group()
