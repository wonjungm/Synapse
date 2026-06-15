# CUDA_VISIBLE_DEVICES=2 python over.py
import torch
import time
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("CUDA device not available!")

print(f"[Dummy Overload] Using device: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

dummy = torch.randn(16384, 16384, device=device)  # 텐서 크기 대폭 증가
while True:
    for _ in range(500):  # 반복 횟수 대폭 증가
        dummy = dummy @ dummy
    torch.cuda.synchronize()
    time.sleep(0.001)  # 거의 쉬지 않음