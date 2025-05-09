from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec, NamedSharding
from time import time

# https://github.com/haoliuhl/ringattention
from ringattention import ringattention, ringattention_inference


###
#   Global Parameters
###
NUM_DEVICES = 4 
NUM_ITERS = 5
NUM_WARMUPS = 2
B, H, N, D_h = 16, 16, 4096*NUM_DEVICES, 64
CHECK_CORRECT = False

assert NUM_DEVICES>=1, 'NUM_DEVICES must be >= 1'
assert N%NUM_DEVICES==0, 'N must be divisible by NUM_DEVICES'


###
#   Prepare Inputs
###
print(f'Starting test with B={B}, H={H}, N={N}, D_h={D_h}, NUM_DEVICES={NUM_DEVICES}')
print('\nGenerating inputs...')
key_Q, key_K, key_V = jax.random.split(jax.random.PRNGKey(42), 3)
Q = jax.random.normal(key_Q, (B, H, N, D_h), dtype=jnp.bfloat16)
K = jax.random.normal(key_K, (B, H, N, D_h), dtype=jnp.bfloat16)
V = jax.random.normal(key_V, (B, H, N, D_h), dtype=jnp.bfloat16)


###
#   Run the reference implementation
###
if CHECK_CORRECT:
    print('\nRunning Reference Implementation...')
    _ = jnp.matmul(Q, K.transpose(0, 1, 3, 2)).astype(jnp.float32)
    _ /= (Q.shape[-1] ** 0.5)
    _ = jax.nn.softmax(_, axis=-1).astype(jnp.bfloat16)
    O_ref = jnp.matmul(_, V)


###
#   Run the original ring attention
###
print('\nRunning Original Ring Attention...')
mesh = jax.make_mesh((1, 1, NUM_DEVICES, 1), ("dp", "fsdp", "sp", "tp"))
QKVO_ps = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
bias_ps = PartitionSpec(("dp", "fsdp"), None, None, None)
seg_ids_ps = PartitionSpec(("dp", "fsdp"), None)
ring_attn_sharded = shard_map( # shard_map automatically JITs the function
    partial(
        ringattention,
        axis_name="sp",
        float32_logits=False,
        cache_idx=None,
        blockwise_kwargs=dict(
            causal_block_size=None, # no causal mask
            deterministic=True,
            dropout_rng=None,
            attn_pdrop=0.0,
            query_chunk_size=N//NUM_DEVICES, # as large as possible for speed
            key_chunk_size=N//NUM_DEVICES,
            dtype=jax.numpy.bfloat16,
            policy=None,
            precision=None,
            prevent_cse=True,
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

# Reshape and shard inputs accordingly
# The original implementation uses (B, N, H, D) format
Q = jax.device_put(Q.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
K = jax.device_put(K.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
V = jax.device_put(V.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))

# We don't care about bias / segments
attn_bias = jax.device_put(jnp.zeros((B, 1, 1, N)), NamedSharding(mesh, bias_ps))
seg_ids = jax.device_put(jnp.zeros((B, N), dtype=jnp.int32), NamedSharding(mesh, seg_ids_ps))

# Calculate and reshape output
O = ring_attn_sharded(Q, K, V, attn_bias, seg_ids)
O = O.transpose(0, 2, 1, 3) # back to (B, N, H, D)


###
#  Verify correctness
###
def check_diff(x, y):
    assert(x.shape == y.shape)
    assert(x.dtype == y.dtype)
    x = jax.device_put(x, device=jax.devices('cpu')[0])
    y = jax.device_put(y, device=jax.devices('cpu')[0])
    abs_diff = jnp.abs(x - y)
    mean_abs_diff = jnp.mean(abs_diff)
    max_abs_diff = jnp.max(abs_diff)
    print("Mean Abs Diff:", mean_abs_diff)
    print("Max Abs Diff:", max_abs_diff)

if CHECK_CORRECT:
    print('\nChecking correctness...')
    check_diff(O, O_ref)


###
#  Check speed
###
print('\nKernel finished, now benchmarking...')
for i in range(NUM_WARMUPS):
    _ = ring_attn_sharded(Q, K, V, attn_bias, seg_ids)
    jax.block_until_ready(_)
times = []
for i in range(NUM_ITERS):
    start_time = time()
    _ = ring_attn_sharded(Q, K, V, attn_bias, seg_ids)
    jax.block_until_ready(_)
    end_time = time()
    times.append(end_time - start_time)
avg_time = sum(times) / NUM_ITERS
total_tflops = (4 * B * H * N * N * D_h + 4 * B * H * N * N) * 1e-12
print(f'Average time per iter: {avg_time * 1e6} us')
print(f'Total TFLOP/s: {total_tflops / avg_time}')
print(f'Per-device TFLOP/s: {(total_tflops / NUM_DEVICES) / avg_time}')
