import torch
import torch.nn.functional as F
from tqdm import trange
import sys

D_QK = 128
D_VO = 128

def generate_inputs(B, H, N):

    q = torch.randn((B, H, N, D_QK), dtype=torch.float32, device='cuda')
    k = torch.randn((B, H, N, D_QK), dtype=torch.float32, device='cuda')
    v = torch.randn((B, H, N, D_VO), dtype=torch.float32, device='cuda') / 5

    s = torch.rand((H,), dtype=torch.float32, device='cuda')
    return q, k, v, s

def get_mask(n, slope=1):

    mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
    for i in range(n):
        x = torch.arange(i + 1)
        y = slope * x
        mask[i, :i + 1] = -torch.flip(y, [0])
    return torch.exp(mask)

def get_full_mask(n, slopes):
    
    arr = []
    for slope in slopes:
        arr.append(get_mask(n, slope.item()))
    mask = torch.stack(arr, dim=0)
    return mask

def linear_attn(q, k, v, s):
    
    b, h, n, d = q.shape
    mask = get_full_mask(n, s).to(q.device).to(torch.float32)
    qk = torch.matmul(q, k.transpose(2, 3))
    qk = (qk.to(torch.float32) * mask).to(q.dtype)
    o = torch.matmul(qk, v)
    
    return o

def save_test_case(q, k, v, s, o):
    
    with open('randn.txt', 'w') as f:
        sf = s.to(torch.float32).flatten().cpu().numpy().tolist()
        qf = q.to(torch.float32).flatten().cpu().numpy().tolist()
        kf = k.to(torch.float32).flatten().cpu().numpy().tolist()
        vf = v.to(torch.float32).flatten().cpu().numpy().tolist()
        of = o.to(torch.float32).flatten().cpu().numpy().tolist()

        for i in trange(len(sf)):
            f.write(repr(sf[i]))
            f.write(' ')

        for i in trange(len(qf)):
            f.write(repr(qf[i]))
            f.write(' ')
            
        for i in trange(len(kf)):
            f.write(repr(kf[i]))
            f.write(' ')

        for i in trange(len(vf)):
            f.write(repr(vf[i]))
            f.write(' ')

        for i in trange(len(of)):
            f.write(repr(of[i]))
            f.write(' ')

def main():
    B, H, N = 2, 2, 1024 
    q, k, v, s = generate_inputs(B, H, N)
    o = linear_attn(q, k, v, s)
    save_test_case(q, k, v, s, o)
    print("Generated random test case")

if __name__ == "__main__":
    main()