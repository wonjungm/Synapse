import os
import torch
from models.factory import create_model

out_dir = "./results/base/base-i100-vit-bn-base"
os.makedirs(out_dir, exist_ok=True)

model = create_model(
    name="vit_bn_base",
    num_class=100,
    image_size=224
)

torch.save(
    {"net": model.state_dict()},
    os.path.join(out_dir, "model_best.pth.tar")
)

print("Saved vit_bn_base teacher checkpoint at:", out_dir)
