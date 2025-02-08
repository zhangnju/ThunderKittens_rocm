from __future__ import annotations

import torch

from math import sqrt

from torch.nn import init
from torch.nn.functional import linear, scaled_dot_product_attention as sdpa

from tqdm import trange

DTYPE: torch.dtype = torch.bfloat16
DEVICE: torch.device = torch.device('cuda:0')

def mitchell(w: torch.Tensor) -> torch.Tensor:
    
    sigma = 1 / sqrt(w.size(-1))
    bound = 3 * sigma

    init.trunc_normal_(w, mean=0.0, std=sigma, a=-bound, b=bound)

torch.random.manual_seed(42)

# QK Dimension(s).
B, R, H, Z, Zc, Zr = 4, 0, 16, 576, 512, 64

# use sdpa to do multi-query gentests

# Input Dimension(s).
L = B * (R + 1)
D = 7168

# Softmax Scale.
Sz = 1 / sqrt(Z)

# Latent Cache Specific(s).
# N, K = 128, 256
N, K   = 128, 128

# Sequence Length Specific(s).
T = 32768 // K

# Sl, Sh = 32, 1024
Sl, Sh   = 128, 129

with torch.device(DEVICE):

    # Initialize Input(s) and Cache(s).
    hidden = torch.randn(L, D, dtype=DTYPE)
    cache  = torch.empty(N, K, 1, Z, dtype=DTYPE)

    mitchell(weight := torch.empty(H * Z, D, dtype=DTYPE))

    # query = (
    #     linear(hidden, weight)
    #     .view(B, R + 1, H, Z)
    # )
    query = torch.ones(B, R + 1, H, Z, dtype=DTYPE)

    # Define Sequence Length and Block Size(s).
    lengths = torch.randint(Sl, Sh, (B,), dtype=torch.int32)
    sizes   = (lengths + (K - 1)).floor_divide(K)

    total   = lengths.sum().item()
    maximum = lengths.max().item()

    # Allocate Block Table.
    table = torch.zeros(B, T, dtype=torch.int32)

    sequence_ids, position_ids = (
        torch.arange(T, dtype=torch.int32)[None, :]
        .expand(B, -1)
        .lt(sizes.view(-1, 1))
        .nonzero(as_tuple=True)
    )

    table[sequence_ids, position_ids] = (
        torch.randperm(N)
        [:sequence_ids.size(0)]
        .sort()
        .values
        .int()
    )

    # Prefill Latent Cache & Extract Padded.
    latent   = torch.ones(total, 1, Z, dtype=DTYPE)
    expanded = latent.expand(total, H, Z)

    sequence_ids, position_ids = (
        torch.arange(maximum)[None, :]
        .expand(B, -1)
        .lt(lengths.view(-1, 1))
        .nonzero(as_tuple=True)
    )

    padded_key   = torch.zeros(B, maximum, H, Z, dtype=DTYPE)
    padded_value = torch.zeros(B, maximum, H, Zc, dtype=DTYPE)

    padded_key[sequence_ids, position_ids]   = expanded
    padded_value[sequence_ids, position_ids] = expanded[..., :Zc]

    entry_ids  = table[sequence_ids, position_ids.floor_divide(K)]
    column_ids = position_ids.fmod(K)

    cache[entry_ids, column_ids] = latent

    # Construct (Optional) Attention Mask.
    mask = None

    if R > 0:
        bounds = (
            torch.arange(R + 1, dtype=torch.int32)[None, :]
            + lengths[:, None]
            - R
        )
        mask = (
            torch.arange(maximum, dtype=torch.int32)[None, None, None, :]
            .expand(B, H, R + 1, -1)
            .lt(bounds[:, None, :, None].expand(B, H, R + 1, 1))
        )
        
# execution barrier
torch.cuda.current_stream().synchronize()

padded_o1 = sdpa(
    query=query.transpose(1, 2),
    key=padded_key.transpose(1, 2),
    value=padded_value.transpose(1, 2),
    attn_mask=mask,
    dropout_p=0.0,
    is_causal=False,
    scale=Sz,
    enable_gqa=False,
)

o1 = padded_o1.transpose(1, 2)
torch.cuda.current_stream().synchronize()

# inputs
### query
### cache
### lengths
### table

# outputs
### output

print("--------------------------------------")
print("--------------------------------------")
print("Q       shape: ", query.shape)
print("Cache   shape: ", cache.shape)
print("Lengths shape: ", lengths.shape)
print("Table   shape: ", table.shape)
print("Output  shape: ", o1.shape)
print("--------------------------------------")
print("--------------------------------------")

filename = f'ones_B{B}_R{R}_H{H}_Z{Z}_Zc{Zc}_Zr{Zr}_N{N}_K{K}_Sl{Sl}_Sh{Sh}.txt'

with open(filename, 'w') as f:
    qf       = query.to(torch.float32).flatten().detach().cpu().numpy()
    cachef   = cache.to(torch.float32).flatten().detach().cpu().numpy()
    lengthsf = lengths.to(torch.float32).flatten().detach().cpu().numpy()
    tablef   = table.to(torch.float32).flatten().detach().cpu().numpy()
    
    outputf  = o1.to(torch.float32).flatten().detach().cpu().numpy()
    
    for i in trange(len(qf)):
        f.write(repr(float(qf[i])))
        f.write(' ')
    
    for i in trange(len(cachef)):
        f.write(repr(float(cachef[i])))
        f.write(' ')
    
    for i in trange(len(lengthsf)):
        f.write(repr(float(lengthsf[i])))
        f.write(' ')
    
    for i in trange(len(tablef)):
        f.write(repr(float(tablef[i])))
        f.write(' ')
    
    for i in trange(len(outputf)):
        f.write(repr(float(outputf[i])))
        f.write(' ')



