#
# Implementation from the original authors of the ring attention paper
# https://github.com/haoliuhl/ringattention
#

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec, NamedSharding
from functools import partial
import torch
from time import time
import numpy as np

# Install: pip install ringattention
from ringattention import ringattention, blockwise_feedforward

###
#   Global Parameters
###
SM_COUNT = 148
NUM_DEVICES = 4
NUM_COMMS = 8 # this is the magic number that works the best
NUM_ITERS = 10
NUM_WARMUPS = 2
ATTN_OPCODE = 725
COMM_OPCODE = 97
B, H, N, D_h = 16, 16, 4096*NUM_DEVICES, 128

assert N%NUM_DEVICES==0, "N must be divisible by NUM_DEVICES"
assert D_h==128, "D_h must be 128"


###
#   Prepare Inputs
###
print(f'Starting test with B={B}, H={H}, N={N}, D_h={D_h}, NUM_DEVICES={NUM_DEVICES}')
print('\nGenerating inputs...')
mesh = jax.make_mesh((1, 1, NUM_DEVICES, 1), ("dp", "fsdp", "sp", "tp"))
QKVO_ps = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
bias_ps = PartitionSpec(("dp", "fsdp"), None, None, None)
seg_ids_ps = PartitionSpec(("dp", "fsdp"), None)
key = jax.random.PRNGKey(42)
key_Q, key_K, key_V = jax.random.split(key, 3)
Q = jax.device_put(jax.random.normal(key_Q, (B, N, H, D_h)), NamedSharding(mesh, QKVO_ps))
K = jax.device_put(jax.random.normal(key_K, (B, N, H, D_h)), NamedSharding(mesh, QKVO_ps))
V = jax.device_put(jax.random.normal(key_V, (B, N, H, D_h)), NamedSharding(mesh, QKVO_ps))
attn_bias = jax.device_put(jnp.zeros((B, 1, 1, N)), NamedSharding(mesh, bias_ps))
seg_ids = jax.device_put(jnp.zeros((B, N), dtype=jnp.int32), NamedSharding(mesh, seg_ids_ps))


###
#   Prepare the JIT'ed function
###
print('\nJITing the function...')
ring_attn_sharded = shard_map( # shard_map automatically JITs the function
    partial(
        ringattention,
        axis_name="sp",
        float32_logits=False,
        cache_idx=None,
        blockwise_kwargs=dict(
            causal_block_size=None, # no causal mask
            deterministic=True, # or false
            dropout_rng=None, # or other value
            attn_pdrop=0.0, # or other value
            query_chunk_size=N//8, # or other value
            key_chunk_size=N//8, # or other value
            dtype=jax.numpy.bfloat16, # or other value
            policy=jax.checkpoint_policies.nothing_saveable,
            precision=None, # or other value
            prevent_cse=True, # or other value
        )
    ),
    mesh=mesh,
    in_specs=(
        QKVO_ps,
        QKVO_ps,
        QKVO_ps,
        bias_ps,
        seg_ids_ps,
    ),
    out_specs=QKVO_ps,
    check_rep=False
)

# Correctness check
print('\nChecking correctness...')
O = ring_attn_sharded(Q, K, V, attn_bias, seg_ids)

# Warmup
print('\nWarming up...')
for _ in range(NUM_WARMUPS):
    __ = ring_attn_sharded(Q, K, V, attn_bias, seg_ids)

# Benchmark
print('\nBenchmarking...')
times = []
for _ in range(NUM_ITERS):
    start = time()
    __ = ring_attn_sharded(Q, K, V, attn_bias, seg_ids)
    end = time()
    times.append(end - start)
avg_time = sum(times) / NUM_ITERS
total_tflops = (4 * B * H * N * N * D_h + 4 * B * H * N * N) * 1e-12
print(f'Average time per iter: {avg_time * 1e6} us')
print(f'Total TFLOP/s: {total_tflops / avg_time}')
print(f'Per-device TFLOP/s: {(total_tflops / NUM_DEVICES) / avg_time}')
