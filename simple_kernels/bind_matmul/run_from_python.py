import simple_tk
import torch

B = 1
H = 4
N = 16
D = 16

"""
Reference Logic: https://github.com/HazyResearch/ThunderKittens/blob/tk_gen/simple_kernels/micro_add/
"""


# input = torch.ones((B, N, D), dtype=torch.bfloat16, device='cuda')

input_x = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
input_y = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda') * 2 
print("Input tensor:", input_x.mean().item(), "dtype:", input_x.dtype) 

def matmul(x, y):
    """
    o = x @ x.T
    input: x is bfloat16
    output: o is float32 (output accumulator)
    """
    o = torch.matmul(x, y.transpose(-2, -1)).to(torch.float32)
    return o

output_ref = matmul(input_x, input_y)
print("Ref output mean:", output_ref.mean().item())  # Debug final output

input_x_copy = input_x.clone()
input_y_copy = input_y.clone()

output_tk = torch.zeros((B, H, N, N), dtype=torch.float32, device='cuda')

simple_tk.dispatch_micro(input_x_copy, input_y_copy, output_tk)
print("TK tensor mean:", output_tk.mean().item(), "dtype:", output_tk.dtype)  # Debug output

print(output_tk)


# 1, 16, 32 x 1, 32 ,16 _.
# # Okay so far this is wrong!
# if torch.allclose(output_ref, output_tk):
#     print("[Passed] TK Kernel matches reference")
# else:
#     print("[Failed] TK Kernel does not match reference")
#     print(output_tk)
