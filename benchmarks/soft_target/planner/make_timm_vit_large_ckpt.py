import os
import torch
import timm

# 저장 경로
out_dir = os.path.abspath("../results/base/base-vit-large")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "vit_large_checkpoint")

# timm ViT-Large pretrained 모델 생성
model_name = "vit_large_patch16_224"
model = timm.create_model(model_name, pretrained=True, num_classes=1000)

# teacher 설정: forward-only
model.eval()
for p in model.parameters():
    p.requires_grad = False

# state_dict 저장
torch.save(model.state_dict(), out_path)

print("Saved timm ViT-Large checkpoint")
print("model:", model_name)
print("path :", out_path)
