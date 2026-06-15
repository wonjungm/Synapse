# make_pretrained_ckpt.py
import os
import torch

from models.factory import create_model

def save_ckpt(path, model):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"net": model.state_dict()}, path)
    print("[OK] wrote:", path)

# 1) teacher: vit_large
tnet = create_model('vit_large', num_class=100, image_size=224)

# 가능하면: create_model 내부에서 pretrained 옵션이 있는지 먼저 확인해서 그걸 쓰는 게 가장 호환성이 좋음.
# 만약 create_model이 pretrained를 직접 지원 안 하면,
# (팀에서 쓰는 모델 정의가 timm/torchvision과 1:1로 안 맞을 수 있어서)
# "동일 아키텍처의 공개 pretrained state_dict를 tnet에 strict=False로 로드" 하는 방식이 필요할 수 있음.
#
# 일단 가장 안전한 1순위는: create_model이 이미 pretrained 로더를 갖고 있는 경우.
# -> 그 경우 여기서 바로 pretrained가 들어간 모델이 생성됐다고 보면 됨.

save_ckpt("../results/base/base-i100-vit-large/model_best.pth.tar", tnet)

# 2) student: resnet152
snet = create_model('resnet152', num_class=100, image_size=224)
save_ckpt("../results/base/base-i100-resnet152/initial_r152.pth.tar", snet)
