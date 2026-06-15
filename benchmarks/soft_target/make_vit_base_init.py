import os
import argparse
import torch
from models.vits import vit_base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_class", type=int, default=100)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--out_path", type=str, required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    teacher = vit_base(num_classes=args.num_class, img_size=args.image_size).eval()

    sd = teacher.state_dict()
    torch.save(
        {
            "net": sd,         
            "state_dict": sd,   
        },
        args.out_path
    )
    print("[OK] saved:", args.out_path)


if __name__ == "__main__":
    main()