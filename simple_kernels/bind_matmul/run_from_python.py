import simple_tk
import torch

B = 1
N = 16
D = 32

"""
Reference Logic: https://github.com/HazyResearch/ThunderKittens/blob/tk_gen/simple_kernels/micro_add/
"""


# input = torch.ones((B, N, D), dtype=torch.bfloat16, device='cuda')

input = torch.ones((B, N, D), dtype=torch.bfloat16, device='cuda')
print("Input tensor:", input.mean().item(), "dtype:", input.dtype)  # Debug input

def matmul(x):
    """
    o = x @ x.T
    input: x is bfloat16
    output: o is float32 (output accumulator)
    """
    o = torch.matmul(x, x.transpose(1, 2))
    # print("Result tensor:", result.mean().item(), "dtype:", result.dtype)  # Debug output
    return o

output_ref = matmul(input)
print("Ref output mean:", output_ref.mean().item())  # Debug final output

input_copy = input.clone()

output_tk = torch.zeros_like(input_copy, dtype=torch.float32)

simple_tk.dispatch_micro(input_copy, output_tk)
print("TK tensor mean:", output_tk.mean().item(), "dtype:", output_tk.dtype)  # Debug output

# Okay sofar this is wrong!
if torch.allclose(output_ref, output_tk):
    print("[Passed] TK Kernel matches reference")
else:
    print("[Failed] TK Kernel does not match reference")
    print(output_tk)
