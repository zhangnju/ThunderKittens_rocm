import pack_unpack_global_fp8
import torch

a = torch.randn(1024, 1024, 1024, 1024, dtype=torch.bfloat16, device="cuda")
b = torch.randn(1024, 1024, 1024, 512, dtype=torch.bfloat16, device="cuda")
c = torch.randn(1024, 1024, 1024, 1024, dtype=torch.bfloat16, device="cuda")

pack_unpack_global_fp8.pack_bf16_to_fp8(a, b)
pack_unpack_global_fp8.unpack_fp8_to_bf16(c, b)

print(torch.max(torch.abs(a - c)))
print(torch.mean(torch.abs(a - c)))