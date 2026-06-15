import os
import argparse
import torch
import torchvision.models as tv_models

from tspipe.model_base import FlattenWrapper


def remove_inplace_ops(module: torch.nn.Module):
    if hasattr(module, "inplace") and module.inplace:
        module.inplace = False
    for c in module.children():
        remove_inplace_ops(c)


def build_resnet50_sequential(num_class: int, pretrained: bool):
    if pretrained:
        m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        m = tv_models.resnet50(weights=None)

    in_features = m.fc.in_features
    fc = torch.nn.Linear(in_features, num_class)
    flatten = FlattenWrapper(1)

    children = []
    for child in list(m.children())[:-1]:
        if isinstance(child, torch.nn.Sequential):
            children.extend(list(child.children()))
        else:
            children.append(child)

    seq = torch.nn.Sequential(*children, flatten, fc)

    remove_inplace_ops(seq)
    return seq.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_class", type=int, default=100)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--out_path", type=str, required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    student = build_resnet50_sequential(num_class=args.num_class, pretrained=args.pretrained)

    sd = student.state_dict()
    torch.save(
        {
            "net": sd,          
            "state_dict": sd,   
        },
        args.out_path
    )

    print("[OK] saved:", args.out_path)
    print("[INFO] pretrained:", args.pretrained)


if __name__ == "__main__":
    main()