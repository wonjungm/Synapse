import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
import argparse

from tqdm import tqdm
from transformers import ViTForImageClassification
from typing import Iterable, List, cast
from collections import OrderedDict

from prototype_dataset import get_dataloaders
from prototype_utils import vit_tiny, get_flattened_vit, get_flattened_teacher_layers

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split(module_children: Iterable[torch.nn.Module], balance: List[int]) -> List[torch.nn.Sequential]:
    """
    Code from TSPipe module
    """
    j = 0
    partitions = []
    layers: OrderedDict[str, torch.nn.Module] = OrderedDict()
    for name, layer in module_children:
        layer_input: torch.nn.Module = layer

        if len(layers) == 0:
            # make this layer as leaf
            for param in layer_input.parameters():
                param.detach_()
                param.requires_grad = True
                assert param.is_leaf
            
        layers[name] = layer_input

        if len(layers) == balance[j]:
            # Group buffered layers as a partition.
            partition = torch.nn.Sequential(layers)
            partitions.append(partition)

            # Prepare for the next partition.
            layers.clear()
            j += 1
    print([len(part) for part in partitions])
    return cast(List[torch.nn.Sequential], partitions)


# ----------------------------
# Teacher/Student Model + Partitioning
# ----------------------------
teacher_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
teacher_model.eval()
# flattened_teacher = get_flattened_teacher_layers(teacher_model)
# partition_teacher = split(flattened_teacher, [4, 4, 3, 3])

student_model = vit_tiny().to(device)
student_model.train()
# flattened_student = get_flattened_vit(student_model)
# partition_student = split(flattened_student, [4, 4, 4, 5])


# ----------------------------
# Dataset
# ----------------------------
train_loader, val_loader = get_dataloaders(
    dataset_name="imagenet2012", # cifar100, cifar10, imagenet2012, imagenet_subset_100k
    root="/nas-ssd/datasets/imagenet2012/imagenet",
    batch_size=256,
    num_workers=4,
    image_size=224
)

# ----------------------------
# KD Loss Function
# ----------------------------
def kd_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=4.0):
    kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kd + (1 - alpha) * ce

optimizer = torch.optim.AdamW(student_model.parameters(), lr=3e-4, weight_decay=0.05)

# ----------------------------
# Evaluation
# ----------------------------


# ----------------------------
# Training Loop
# ----------------------------
def run_naive_kd(epochs:int = 50):
    save_path = f"checkpoint/vittiny_i1000_e{epochs}_naive_{time.strftime('%Y%m%d_%H%M%S')}"
    
    def evaluate(model, dataloader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"[Eval] Accuracy: {acc * 100:.2f}%")
        wandb.log({"val/accuracy": acc * 100})
        model.train()
        
    try:
        for epoch in range(epochs):
            epoch_loader = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
            for step, (x, y) in enumerate(epoch_loader):
                x, y = x.to(device), y.to(device)

                start_time = time.time()
                with torch.no_grad():
                    teacher_logits = teacher_model(x).logits
                    teacher_preds = teacher_logits.argmax(dim=1)
                    teacher_acc = (teacher_preds == y).float().mean().item()

                student_logits = student_model(x)
                loss = kd_loss(student_logits, teacher_logits, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                duration = time.time() - start_time

                if step % 50 == 0:
                    student_acc = (student_logits.argmax(dim=1) == y).float().mean().item()
                    print(f"[Epoch {epoch} Step {step}] Loss: {loss.item():.4f} | Student Acc: {student_acc * 100:.2f}% | Teacher Acc: {teacher_acc * 100:.2f}% | Time: {duration:.2f}s")
                    # wandb.log({"epoch": epoch, "step": step, "train/loss": loss.item(), "train/student_acc": student_acc * 100, "train/teacher_acc": teacher_acc * 100, "train/step_time_sec": duration})

            evaluate(student_model, val_loader)

    except Exception as e:
        save_path = save_path + '_temp'
        print(f"[Error] Exception occurred: {e}. Saving temporary checkpoint...")
        raise e

    finally:
        if not os.path.exists('checkpoint/'):
            os.makedirs('checkpoint/')
        print(f"Saving model to {save_path}.pt ...")
        torch.save(student_model.state_dict(), f'{save_path}.pt')
        # wandb.save(f'{save_path}.pt')
        # wandb.finish()
        print("Model saved. Cleanup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run naive KD training.")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    args = parser.parse_args()

    # wandb.init(
    #     project="vit-kd-pipeline",
    #     entity="eunjin-lee", 
    #     name=f"vit_kd_i1000_{time.strftime('%Y%m%d_%H%M%S')}",
    #     config={
    #         "epochs": args.epochs,
    #         "batch_size": 256,
    #         "learning_rate": 6e-4,
    #         "weight_decay": 0.05,
    #         "temperature": 4.0,
    #         "alpha": 0.5,
    #         "model_teacher": "google/vit-base-patch16-224",
    #         "model_student": "vit-tiny",
    #         "dataset": "imagenet2012"
    #     }
    # )

    run_naive_kd(epochs=args.epochs)