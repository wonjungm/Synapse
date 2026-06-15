import torch
import torch.nn as nn
import timm

from tspipe.model_base import SequentialableModel


class DeiTWrapper(SequentialableModel):
    """
    timm DeiT wrapper for TSPipe
    - Uses official fb_in1k checkpoints
    - Keeps token + pos_embed atomic
    """

    def __init__(self, name: str, num_classes: int, image_size: int = 224):
        super().__init__()

        timm_name_map = {
            "deit_tiny":  "deit_tiny_patch16_224",
            "deit_small": "deit_small_patch16_224",
            "deit_base":  "deit_base_patch16_224",
        }


        if name not in timm_name_map:
            raise ValueError(f"Unsupported DeiT name: {name}")

        self.model = timm.create_model(
            timm_name_map[name],
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def to_sequential(self) -> nn.Sequential:
        m = self.model
        blocks = list(m.blocks.children())

        class TokenPosEmbed(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                B = x.shape[0]

                # patch embedding
                x = self.model.patch_embed(x)

                # EXACT timm logic
                cls = self.model.cls_token.expand(B, -1, -1)
                if hasattr(self.model, "dist_token") and self.model.dist_token is not None:
                    dist = self.model.dist_token.expand(B, -1, -1)
                    x = torch.cat((cls, dist, x), dim=1)
                else:
                    x = torch.cat((cls, x), dim=1)

                x = x + self.model.pos_embed
                x = self.model.pos_drop(x)
                return x

        return nn.Sequential(
            TokenPosEmbed(m),
            *blocks,
            m.norm,
            m.head
        )
