import torch

from models.factory import create_model
from tspipe.model_base import SequentialableModel


def main():
    # 1. 모델 생성 (DeiT base)
    model = create_model(
        name="deit_base",
        num_class=100,
        image_size=224
    )

    # 2. Sequential 변환
    seq_model = model.to_sequential()

    print("=" * 60)
    print("Model type:", type(model))
    print("Sequential length:", len(seq_model))
    print("=" * 60)

    # 3. forward equivalence test
    ok = SequentialableModel.test_sequential_validity(
        model_orig=model,
        test_tensor_size=(1, 3, 224, 224)
    )

    print("Sequential valid:", ok)
    print("=" * 60)

    # 4. (선택) 실제 forward shape 확인
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y_base = model(x)
        y_seq = seq_model(x)

    print("Base output shape:", y_base.shape)
    print("Seq  output shape:", y_seq.shape)
    print("Max abs diff:", (y_base - y_seq).abs().max().item())


if __name__ == "__main__":
    main()
