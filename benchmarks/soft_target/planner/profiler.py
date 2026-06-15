import os
import time
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOFT_TARGET_DIR = SCRIPT_DIR.parent
REPO_ROOT = SOFT_TARGET_DIR.parent.parent

_TMPDIR_CANDIDATES = [Path("/dev/shm") / "synapse_profiler_tmp",
                      Path("/tmp") / "synapse_profiler_tmp",
                      SCRIPT_DIR / ".tmp"]
for _candidate in _TMPDIR_CANDIDATES:
    try:
        _candidate.mkdir(parents=True, exist_ok=True)
        _TMPDIR = _candidate
        break
    except OSError:
        continue
else:
    raise RuntimeError("No usable temporary directory available for profiler.py")

os.environ.setdefault("TMPDIR", str(_TMPDIR))
os.environ.setdefault("TEMP", str(_TMPDIR))
os.environ.setdefault("TMP", str(_TMPDIR))

sys.path.append(str(SOFT_TARGET_DIR))
import csv
import argparse

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as tv_datasets
import torchmodules.torchprofiler as torchprofiler
import numpy as np
from functools import partial
from tqdm import tqdm

import dataset.datasets as small_datasets
from models.factory import create_model
from utils import load_pretrained_model
from tspipe.batch_ops import defaultScatterGatherFn

TEACHER_CKPT = SOFT_TARGET_DIR / "results/base/base-i100-vit-large/model_best.pth.tar"
STUDENT_CKPT = SOFT_TARGET_DIR / "results/base/base-i100-resnet152/initial_r152.pth.tar"
DATA_ROOT = REPO_ROOT / "results/base/imagenet100_mini"

tnet = create_model('vit_large', num_class=100, image_size=224)
checkpoint = torch.load(str(TEACHER_CKPT), map_location='cpu')
load_pretrained_model(tnet, checkpoint['net'])
if not isinstance(tnet, torch.nn.Sequential):
    tnet = tnet.to_sequential()
tnet.cuda()
print("Teacher model loaded: %s" % tnet)
    
snet = create_model('resnet152', num_class=100, image_size=224)
checkpoint = torch.load(str(STUDENT_CKPT), map_location='cpu')
load_pretrained_model(snet, checkpoint['net'])
if not isinstance(snet, torch.nn.Sequential):
    snet = snet.to_sequential()
snet.cuda()

'''
# --- Teacher (vit_base) ---
tnet = create_model('vit_base', num_class=100, image_size=224)
checkpoint = torch.load('../results/base/base-i100-vit-base/teacher_init_vit_base.pth.tar', map_location='cpu')
load_pretrained_model(tnet, checkpoint['net'])
if not isinstance(tnet, torch.nn.Sequential):
    tnet = tnet.to_sequential()
tnet.cuda()

# --- Student (resnet50) ---
snet = create_model('resnet50', num_class=100, image_size=224)
checkpoint = torch.load('../results/base/base-i100-resnet50/student_init_resnet50.pth.tar', map_location='cpu')
load_pretrained_model(snet, checkpoint['net'])
if not isinstance(snet, torch.nn.Sequential):
    snet = snet.to_sequential()
snet.cuda()
'''
# --- Teacher (vit_small) ---
#tnet = create_model('vit_small', num_class=100, image_size=224)
#checkpoint = torch.load('../results/base/base-i100-vit-small/initial_rall.pth.tar', map_location='cpu')
#load_pretrained_model(tnet, checkpoint['net'])
#if not isinstance(tnet, torch.nn.Sequential):
#    tnet = tnet.to_sequential()
#tnet.cuda()

# --- Student (efficientnet-b0) ---
#snet = create_model('efficientnet-b0', num_class=100, image_size=224)
#checkpoint = torch.load('../results/base/base-i100-efficientnet-b0/initial_rentnet-b0.pth.tar', map_location='cpu')
#load_pretrained_model(snet, checkpoint['net'])
#if not isinstance(snet, torch.nn.Sequential):
 #   snet = snet.to_sequential()
#snet.cuda()

# print("Student model loaded: %s" % snet)
# sys.exit(0)

optimizer = torch.optim.SGD(snet.parameters(), lr=0.1,
                            momentum=0.9,
                            weight_decay=1e-4,
                            nesterov=True)

cudnn.benchmark = True

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

print(f"[Info] Profiling with batch size = {args.batch_size}")

dataset = small_datasets.ImageNet100
train_dataset = partial(dataset, split='train')

train_loader = torch.utils.data.DataLoader(
    train_dataset(
        root=str(DATA_ROOT),
        transform=train_transform
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=24,
    pin_memory=True
)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def kd_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=4.0):
    kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kd + (1 - alpha) * ce

NUM_EPOCHS = 1 
IS_VERBOSE = True # FIXME
NUM_MICROBATCHES = 3
NUM_STEPS_TO_RUN = 3  # number of steps to profile
TARGET_STEP = 2 # the step to analyze in detail

def profile_train(train_loader, tnet, snet, optimizer, profile_target, microbatch_chunks=NUM_MICROBATCHES):
    mini_times_meter = AverageMeter()
    # micro_times_meter = AverageMeter()

    profile_target_model = tnet if profile_target == 'tnet' else snet
    tnet.eval()
    snet.train()

    minibatch_profiles  = []
    pbar = tqdm(train_loader)
    for step, minibatches in enumerate(pbar):
        microbatches = defaultScatterGatherFn.scatter(minibatches, chunks=microbatch_chunks)
        
        mini_start_time = time.time()
        optimizer.zero_grad()
        accumulated_loss = 0.0 # for logging

        microbatch_profiles = []
        for ustep, (micro_x, micro_y) in enumerate(microbatches):
            micro_x, micro_y = micro_x.cuda(), micro_y.cuda()

            with torchprofiler.Profiling(profile_target_model, module_whitelist=[]) as micro_profiler:
                with torch.no_grad():
                    t_logits = tnet(micro_x)
                s_logits = snet(micro_x)  # shape: [B, C]                
                loss = kd_loss(s_logits, t_logits, micro_y) / microbatch_chunks
                loss.backward() # Backward pass, gradients are accumulated in .grad buffers
                accumulated_loss += loss.item() # for logging

            microbatch_profiles.append(micro_profiler.processed_results())
        
        minibatch_profiles.append(microbatch_profiles)
        torch.nn.utils.clip_grad_norm_(snet.parameters(), max_norm=1.0)
        optimizer.step()

        mini_time = time.time() - mini_start_time
        mini_times_meter.update(mini_time*1000)
        # print(f"[Step {step}] loss: {accumulated_loss:.4f}")

        if step >= NUM_STEPS_TO_RUN:
            break

    print(f"[Info] Average mini-batch time: {mini_times_meter.avg:.3f} ms")

    num_layers = len(minibatch_profiles[0][0])
    layer_profile = []
    target_minibatches = minibatch_profiles[1:]

    print(f"========== Averaging over {NUM_STEPS_TO_RUN} steps ==========")

    for layer_idx in range(num_layers):
        fwd_times, bwd_times = [], []
        input_sizes, output_sizes = [], []
        interm_sizes, param_sizes = [], []

        for step_microbatches in target_minibatches:
            fwd_time_ms, bwd_time_ms = 0, 0
            input_size_kb, output_size_kb = 0, 0
            interm_size_kb, param_size_kb = 0, 0
            for layer_record in step_microbatches:
                if layer_idx >= len(layer_record):
                    continue
                layer = layer_record[layer_idx]
                fwd_time_ms += layer[1]
                bwd_time_ms += layer[2] if layer[2] >= 0 else 0

                input_shape = layer[3]
                output_shape = layer[4]

                input_kb = int(np.prod(input_shape)) * 4 / 1024 if input_shape is not None else 0
                output_kb = int(np.prod(output_shape)) * 4 / 1024 if output_shape is not None else 0
                input_size_kb += input_kb
                output_size_kb += output_kb

                interm_size_kb = layer[6] / 1024
                param_size_kb = layer[5] / 1024
            
            fwd_times.append(fwd_time_ms)
            bwd_times.append(bwd_time_ms)
            input_sizes.append(input_size_kb)
            output_sizes.append(output_size_kb)
            interm_sizes.append(interm_size_kb)
            param_sizes.append(param_size_kb)

        micro_accum_profiles = {
            'layer': f'layer{layer_idx}',
            'forward_time_ms': np.mean(fwd_times),
            'backward_time_ms': np.mean(bwd_times),
            'input_activation_size_kb': np.mean(input_sizes),
            'output_activation_size_kb': np.mean(output_sizes),
            'accum_activation_size_kb': np.mean(interm_sizes),
            'parameter_size_kb': np.mean(param_sizes),
        }
        layer_profile.append(micro_accum_profiles)


    OUTPUT_DIR = './profile'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{profile_target}.csv")
    
    if not layer_profile:
        print("[Warning] layer_profile is empty. No file written.")
        return

    try:
        with open(OUTPUT_PATH, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=layer_profile[0].keys())
            writer.writeheader()
            writer.writerows(layer_profile)
        print(f"[Info] Profile written to {OUTPUT_PATH}")
    except Exception as e:
        print(f"[Error] Failed to write profile CSV: {e}")

    return layer_profile


print("Collecting profile...")

profile_start_time = time.perf_counter()

t_layer_profile = profile_train(train_loader, tnet, snet, optimizer, profile_target='tnet')
s_layer_profile = profile_train(train_loader, tnet, snet, optimizer, profile_target='snet')

dp_duration = (time.perf_counter() - profile_start_time) * 1000
print(f"\n[Info] Profile Duration of : {dp_duration:.3f} ms of {NUM_STEPS_TO_RUN} iterations\n")

if IS_VERBOSE:
    print("\nTEACHER MODEL ========================================================================================================")
    print("Layer    Forward Time (ms)    Backward Time (ms)    Output Act Size (KB)   Interm Act Size (KB)   Layer Param Size (KB)")
    print("======================================================================================================================")
    for i, layer in enumerate(t_layer_profile):
        print(f"{layer['layer']:>7} {layer['forward_time_ms']:>20.3f} {layer['backward_time_ms']:>20.3f} "
              f"{layer['output_activation_size_kb']:>20} {layer['accum_activation_size_kb']:>20} {layer['parameter_size_kb']:>20} ")

    print("\nSTUDENT MODEL ========================================================================================================")
    print("Layer    Forward Time (ms)    Backward Time (ms)    Output Act Size (KB)   Interm Act Size (KB)   Layer Param Size (KB)")
    print("======================================================================================================================")
    for i, layer in enumerate(s_layer_profile):
        print(f"{layer['layer']:>7} {layer['forward_time_ms']:>20.3f} {layer['backward_time_ms']:>20.3f} "
              f"{layer['output_activation_size_kb']:>20} {layer['accum_activation_size_kb']:>20} {layer['parameter_size_kb']:>20} ")
