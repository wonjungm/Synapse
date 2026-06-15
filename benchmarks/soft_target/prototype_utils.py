import torch.nn as nn
from collections import OrderedDict
from functools import partial
from transformers import ViTForImageClassification
from timm.models.vision_transformer import _cfg
from models.vits import VisionTransformerMoCo

def vit_tiny(num_classes=1000, img_size=224, **kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def get_flattened_vit(model: nn.Module):
    layers = OrderedDict()

    # 1. Patch embedding
    layers["patch_embed"] = model.patch_embed
    layers["pos_drop"] = model.pos_drop

    # 2. Transformer blocks
    for i, block in enumerate(model.blocks):
        layers[f"block_{i}"] = block

    # 3. Norm and classification head
    layers["norm"] = model.norm
    layers["pre_logits"] = model.pre_logits
    layers["head"] = model.head

    return list(layers.items()) 

def get_flattened_teacher_layers(model: ViTForImageClassification) -> list[tuple[str, nn.Module]]:
    vit = model.vit
    encoder = vit.encoder

    layers = OrderedDict()
    layers["embeddings"] = vit.embeddings

    for i, block in enumerate(encoder.layer):
        layers[f"block_{i}"] = block

    layers["classifier"] = model.classifier
    return list(layers.items())

