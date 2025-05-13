import test
import torch

tile_in_1 = torch.zeros(128, 128, dtype=torch.float16).to(torch.float8_e4m3fn).to("cuda")
tile_in_1[0, :] = torch.arange(128, dtype=torch.float16).to(torch.float8_e4m3fn)

# for i in range(128):
#     for j in range(0, 128, 8):
#         tile_in_1[i, j:j+8] = torch.arange(8, dtype=torch.float16).to(torch.float8_e4m3fn)


tile_in_2 = torch.randn(128, 128, dtype=torch.float16).to(torch.float8_e4m3fn).to("cuda")
tile_1_copy_out = torch.zeros(128, 128, dtype=torch.float16).to(torch.float8_e4m3fn).to("cuda")
tile_12_matmul_out = torch.randn(128, 128, dtype=torch.float16).to(torch.float8_e4m3fn).to("cuda")

test.test(tile_in_1, tile_in_2, tile_1_copy_out, tile_12_matmul_out)

torch.cuda.synchronize()

print("tile_in_1")
for i in range(128):
    for j in range(128):
        print(f"{tile_in_1[i, j].item():.2f}", end="\t")
    print()
print("tile_1_copy_out")
for i in range(128):
    for j in range(128):
        print(f"{tile_1_copy_out[i, j].item():.2f}", end="\t")
    print()