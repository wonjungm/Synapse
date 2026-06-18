from __future__ import absolute_import, division, print_function

import torch.cuda.nvtx as nvtx

import argparse
import json
import logging
import os
import random
import sys
import time
import yaml
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torchvision.datasets as dst
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataset.datasets as small_datasets
from kd_losses import Logits, SoftTarget
from models.factory import create_model
from tspipe import TSPipe
from tspipe.tspipe import TSPipeMode
from tspipe.slowdown_detector import SlowdownDetector  # NEW: Slowdown monitoring
from planner.mathematical_optimizer import MathematicalFailoverOptimizer  # NEW: Math-based policy
from planner.stage_time_predictor import PartitionConfig
from utils import (AverageMeter, accuracy, count_parameters,
                   count_parameters_in_MB, create_exp_dir,
                   load_pretrained_model, save_checkpoint)

parser = argparse.ArgumentParser(description='train kd')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--s_init', type=str, required=True, help='initial parameters of student model')
parser.add_argument('--t_model', type=str, required=True, help='path name of teacher model')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')

# 빠른 실험/디버깅을 위한 옵션: 한 epoch당 최대 step 수를 제한
parser.add_argument('--max-steps-per-epoch', type=int, default=0,
                    help='If >0, limit number of training steps per epoch (for quick e2e tests)')

parser.add_argument('--tspipe-enable', action='store_true', default=False, dest='tspipe')

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')

# net and dataset choosen
parser.add_argument('--data_name', type=str, required=True, help='name of dataset')  # cifar10/cifar100
parser.add_argument('--t_name', type=str, required=True, help='name of teacher')     # resnet20/resnet110
parser.add_argument('--s_name', type=str, required=True, help='name of student')     # resnet20/resnet110

# hyperparameter
parser.add_argument('--kd_mode', type=str, required=True, help='mode of kd, which can be:'
                                                               'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                                                               'sp/sobolev/cc/lwm/irg/vid/ofd/afd')
parser.add_argument('--lambda_kd', type=float, default=1.0, help='trade-off parameter for kd loss')
parser.add_argument('--T', type=float, default=4.0, help='temperature for ST')
parser.add_argument('--p', type=float, default=2.0, help='power for AT')
parser.add_argument('--w_dist', type=float, default=25.0, help='weight for RKD distance')
parser.add_argument('--w_angle', type=float, default=50.0, help='weight for RKD angle')
parser.add_argument('--m', type=float, default=2.0, help='margin for AB')
parser.add_argument('--gamma', type=float, default=0.4, help='gamma in Gaussian RBF for CC')
parser.add_argument('--P_order', type=int, default=2, help='P-order Taylor series of Gaussian RBF for CC')
parser.add_argument('--w_irg_vert', type=float, default=0.1, help='weight for IRG vertex')
parser.add_argument('--w_irg_edge', type=float, default=5.0, help='weight for IRG edge')
parser.add_argument('--w_irg_tran', type=float, default=5.0, help='weight for IRG transformation')
parser.add_argument('--sf', type=float, default=1.0, help='scale factor for VID, i.e. mid_channels = sf * out_channels')
parser.add_argument('--init_var', type=float, default=5.0, help='initial variance for VID')
parser.add_argument('--att_f', type=float, default=1.0, help='attention factor of mid_channels for AFD')
parser.add_argument('--dryrun-failover-cycle', action='store_true', default=False,
                    help='Run deterministic failover restart/resume dry-run without real training data')
parser.add_argument('--failover-inject-scenario', type=str, default='',
                    help='Inject synthetic slowdown scenario (e.g., KEEP_REPLAN_DEGRADE) for testing failover policies')
parser.add_argument('--inject-slowdown-gpu', type=int, default=None, help='GPU index to inject slowdown (relative to CUDA_VISIBLE_DEVICES)')
parser.add_argument('--slowdown-mode', type=str, default='ratio', choices=['ratio', 'fixed'],
                    help='Synthetic slowdown mode: ratio (baseline-scaled) or fixed (constant sleep)')
parser.add_argument('--slowdown-factor', type=float, default=None, help='Slowdown factor (e.g., 1.5 for 1.5x slower)')
parser.add_argument('--slowdown-fixed-ms', type=float, default=None,
                    help='Fixed sleep time per task in ms when --slowdown-mode=fixed (e.g., 50)')
parser.add_argument('--slowdown-duration', type=int, default=None, help='Number of steps to inject slowdown (default: scenario-specific)')
parser.add_argument('--slowdown-start', type=int, default=None, help='Start step for slowdown injection (default: scenario-specific)')
parser.add_argument('--slowdown-end', type=int, default=None, help='End step for slowdown injection (default: scenario-specific)')
parser.add_argument('--slowdown-warmup-sec', type=float, default=None,
                    help='Warm-up duration before wall-clock slowdown injection starts')
parser.add_argument('--slowdown-duration-sec', type=float, default=None,
                    help='Wall-clock duration for slowdown injection after warm-up')
parser.add_argument('--slowdown-task-scope', type=str, default='compute',
                    choices=['compute', 'comm', 'both'],
                    help='Inject slowdown inside worker task path: compute / comm / both')

args, unparsed = parser.parse_known_args()

# ✅ NEW: Read FAILOVER_INJECT_SCENARIO from environment variable (for E2E launcher compatibility)
if not args.failover_inject_scenario:
    args.failover_inject_scenario = os.environ.get("FAILOVER_INJECT_SCENARIO", "").strip()

if args.failover_inject_scenario:
    logging.info(f"🧪 FAILOVER_INJECT_SCENARIO enabled: {args.failover_inject_scenario}")

args.save_root = os.path.join(args.save_root, args.note)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


class _ResumeAwareBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Build deterministic epoch batches and jump directly to a batch offset."""

    def __init__(
        self,
        dataset_len: int,
        batch_size: int,
        epoch: int,
        *,
        start_batch: int = 0,
        max_batches: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset_len = int(dataset_len)
        self.batch_size = max(1, int(batch_size))
        self.epoch = int(epoch)
        self.start_batch = max(0, int(start_batch))
        self.max_batches = int(max_batches)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        if self.drop_last:
            self.total_batches = self.dataset_len // self.batch_size
        else:
            self.total_batches = (self.dataset_len + self.batch_size - 1) // self.batch_size

        if self.max_batches > 0:
            self.effective_total = min(self.total_batches, self.max_batches)
        else:
            self.effective_total = self.total_batches

        self.start_batch = min(self.start_batch, self.effective_total)

    def _build_epoch_indices(self) -> List[int]:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + max(0, self.epoch - 1))
            indices = torch.randperm(self.dataset_len, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_len))

        if self.drop_last:
            usable = self.total_batches * self.batch_size
            indices = indices[:usable]

        return indices

    def __iter__(self):
        indices = self._build_epoch_indices()
        for batch_idx in range(self.start_batch, self.effective_total):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch = indices[start:end]
            if len(batch) < self.batch_size and self.drop_last:
                break
            if batch:
                yield batch

    def __len__(self) -> int:
        return max(0, self.effective_total - self.start_batch)


def _extract_cli_option(unparsed_args, name: str) -> Optional[str]:
    """Read option value from parse_known_args() leftovers."""
    for idx, token in enumerate(unparsed_args):
        if token == name and idx + 1 < len(unparsed_args):
            value = unparsed_args[idx + 1]
            logging.error(f"🔍 Extracted {name} (space-separated): {value}")
            return value
        prefix = f"{name}="
        if token.startswith(prefix):
            value = token[len(prefix):]
            logging.error(f"🔍 Extracted {name} (equals-style): {value}")
            return value
    logging.error(f"🔍 Failed to extract {name} from unparsed_args: {unparsed_args}")
    return None


def _validate_slowdown_cli_args():
    uses_step_window = (
        args.slowdown_start is not None or
        args.slowdown_end is not None or
        args.slowdown_duration is not None
    )
    uses_time_window = (
        args.slowdown_warmup_sec is not None or
        args.slowdown_duration_sec is not None
    )

    if uses_time_window:
        if args.slowdown_warmup_sec is None or args.slowdown_duration_sec is None:
            raise ValueError(
                "Both --slowdown-warmup-sec and --slowdown-duration-sec must be provided together."
            )
        if float(args.slowdown_warmup_sec) < 0.0:
            raise ValueError(
                f"--slowdown-warmup-sec must be >= 0, got {args.slowdown_warmup_sec}"
            )
        if float(args.slowdown_duration_sec) <= 0.0:
            raise ValueError(
                f"--slowdown-duration-sec must be > 0, got {args.slowdown_duration_sec}"
            )

    if uses_step_window and uses_time_window:
        raise ValueError(
            "Step-based slowdown (--slowdown-start/--slowdown-end/--slowdown-duration) "
            "cannot be combined with wall-clock slowdown "
            "(--slowdown-warmup-sec/--slowdown-duration-sec)."
        )

    if args.slowdown_mode == 'fixed':
        if args.slowdown_fixed_ms is None:
            raise ValueError(
                "--slowdown-mode=fixed requires --slowdown-fixed-ms to be set."
            )
        if float(args.slowdown_fixed_ms) <= 0.0:
            raise ValueError(
                f"--slowdown-fixed-ms must be > 0, got {args.slowdown_fixed_ms}"
            )
    elif args.slowdown_fixed_ms is not None and float(args.slowdown_fixed_ms) <= 0.0:
        raise ValueError(
            f"--slowdown-fixed-ms must be > 0 when provided, got {args.slowdown_fixed_ms}"
        )


def _configure_environment_injected_slowdown(total_steps: int) -> None:
    """Map FAILOVER_* env vars onto the existing TSPipe slowdown knobs."""
    scenario = os.environ.get("FAILOVER_INJECT_SCENARIO", "").strip().lower()
    if scenario != "slowdown":
        return

    target_gpu_raw = os.environ.get("FAILOVER_INJECT_GPU", "").strip()
    ratio_raw = os.environ.get("FAILOVER_INJECT_RATIO", "").strip()
    if not target_gpu_raw or not ratio_raw:
        logging.info(
            "🧪 FAILOVER slowdown env detected but FAILOVER_INJECT_GPU/FAILOVER_INJECT_RATIO is missing; "
            "skipping injection config"
        )
        return

    try:
        target_gpu = int(target_gpu_raw)
        ratio = float(ratio_raw)
    except ValueError as exc:
        logging.warning(f"⚠️ Invalid FAILOVER slowdown env values: gpu={target_gpu_raw}, ratio={ratio_raw} ({exc})")
        return

    if ratio <= 1.0:
        logging.info(f"🧪 FAILOVER_INJECT_RATIO={ratio:.3f} <= 1.0; skipping slowdown injection")
        return

    warmup_steps = 50
    total_steps = max(1, int(total_steps))
    if warmup_steps >= total_steps:
        logging.warning(
            f"⚠️ Warmup step ({warmup_steps}) is not smaller than total_steps ({total_steps}); "
            "skipping slowdown injection"
        )
        return

    args.failover_inject_scenario = "slowdown"
    args.inject_slowdown_gpu = target_gpu
    args.slowdown_mode = "ratio"
    args.slowdown_factor = ratio
    args.slowdown_start = warmup_steps
    args.slowdown_end = total_steps
    args.slowdown_duration = total_steps - warmup_steps
    args.slowdown_fixed_ms = None
    args.slowdown_warmup_sec = None
    args.slowdown_duration_sec = None
    args.slowdown_task_scope = "compute"

    logging.info(
        f"🧪 FAILOVER slowdown configured: gpu={target_gpu}, ratio={ratio:.2f}x, "
        f"warmup={warmup_steps} steps, active_window=[{warmup_steps}, {total_steps})"
    )

    _validate_slowdown_cli_args()


def _latest_failover_coeff_path(save_root: str) -> str:
    return os.path.join(save_root, "alpha_beta_latest.json")


def _write_json_atomic(path: str, payload: dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _persist_latest_failover_coefficients(
    save_root: str,
    alpha_comp,
    beta_comm,
    partition: Optional[PartitionConfig] = None,
    source: str = "runtime",
    step_id: Optional[int] = None,
):
    if not isinstance(alpha_comp, dict) or not isinstance(beta_comm, dict):
        return

    try:
        os.makedirs(save_root, exist_ok=True)
        payload = {
            "timestamp": time.time(),
            "source": str(source),
            "step_id": None if step_id is None else int(step_id),
            "alpha_comp": {int(k): float(v) for k, v in alpha_comp.items()},
            "beta_comm": {int(k): float(v) for k, v in beta_comm.items()},
        }

        if partition is not None:
            if isinstance(partition, PartitionConfig):
                payload["partition"] = {
                    "gpu_assignment": [int(v) for v in partition.gpu_assignment],
                    "snet_partition": [int(v) for v in partition.snet_partition],
                    "tnet_partition": [int(v) for v in partition.tnet_partition],
                }
            elif isinstance(partition, dict):
                payload["partition"] = partition

        _write_json_atomic(_latest_failover_coeff_path(save_root), payload)
    except Exception as e:
        logging.warning(f"⚠️ Failed to persist latest alpha/beta snapshot: {e}")


def _apply_restart_partition_to_tspipe_yaml(unparsed_args, restart_payload: dict):
    """Overwrite tspipe model_split in YAML so restarted process boots with new partition."""
    tspipe_config_path = _extract_cli_option(unparsed_args, "--tspipe-config")
    if not tspipe_config_path:
        logging.warning("Failover restart detected but --tspipe-config not found in CLI args")
        return

    partition = restart_payload.get("partition", {})
    snet_partition = partition.get("snet_partition")
    tnet_partition = partition.get("tnet_partition")
    if not isinstance(snet_partition, list) or not isinstance(tnet_partition, list):
        logging.warning("failover restart payload missing snet/tnet partition lists; skip YAML override")
        return
    # Build inline list strings (e.g., "[8, 4, 4, 14]")
    snet_inline = ", ".join(str(int(v)) for v in snet_partition)
    tnet_inline = ", ".join(str(int(v)) for v in tnet_partition)

    try:
        with open(tspipe_config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"❌ tspipe config YAML not found: {tspipe_config_path}")
        return

    new_lines = []
    inside_model_split = False
    model_split_indent = ""
    online_updated = False
    target_updated = False

    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        # Enter model_split block (under tspipe) and remember its indent
        if stripped.startswith("model_split:") and not inside_model_split:
            inside_model_split = True
            model_split_indent = indent
            new_lines.append(line)
            continue

        # If we were inside model_split and indentation goes back, we've exited the block
        if inside_model_split and stripped and not stripped.startswith("#"):
            if len(indent) <= len(model_split_indent):
                inside_model_split = False

        if inside_model_split:
            # Replace online/target lines only inside the model_split block
            if stripped.startswith("online:") and not online_updated:
                new_lines.append(f"{indent}online: [{snet_inline}]\n")
                online_updated = True
                continue
            if stripped.startswith("target:") and not target_updated:
                new_lines.append(f"{indent}target: [{tnet_inline}]\n")
                target_updated = True
                continue

        new_lines.append(line)

    if not (online_updated and target_updated):
        logging.error(
            f"❌ Failed to update model_split.online/target in YAML (online_updated={online_updated}, "
            f"target_updated={target_updated}): {tspipe_config_path}"
        )
        return

    with open(tspipe_config_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    logging.error(f"🔧 YAML partition updated in-place (model_split only): {tspipe_config_path}")
    logging.error(f"   New snet_partition (online): [{snet_inline}]")
    logging.error(f"   New tnet_partition (target): [{tnet_inline}]")

    if os.path.exists(tspipe_config_path):
        logging.error(f"✅ YAML file verified to exist: {tspipe_config_path}")
    else:
        logging.error(f"❌ ERROR: YAML file was not created after in-place update!")
        return


def _load_failover_bootstrap(save_root: str, unparsed_args, snet, tnet, optimizer, steps_per_epoch: int, skip_partition_yaml_apply=False, skip_checkpoint_restore=False):
    failover_restart_path = os.path.join(save_root, "failover_restart_config.json")
    emergency_restart_path = os.path.join(save_root, "emergency_restart_config.json")
    legacy_restart_path = os.path.join(save_root, "restart_config.json")

    # Fallback chain: soft → hard → legacy
    restart_config_path = failover_restart_path
    if not os.path.exists(restart_config_path):
        # Hard failure restart (K-1 reconfiguration generated by tspipe.py)
        restart_config_path = emergency_restart_path
    if not os.path.exists(restart_config_path):
        # Backward compatibility for older soft-failover runs.
        restart_config_path = legacy_restart_path

    if not os.path.exists(restart_config_path):
        return {
            "enabled": False,
            "resume_step": 0,
            "resume_epoch": 1,
            "resume_batch_offset": 0,
            "partition": None,
            "alpha_comp": None,
            "beta_comm": None,
            "restart_payload": None,
            "restart_transition": None,
            "slowdown_detector_state": None,
            "optimizer_runtime_state": None,
        }

    with open(restart_config_path, "r", encoding="utf-8") as f:
        restart_payload = json.load(f)

    # Both soft and hard failures can include partition metadata for K-1 reconfiguration
    if not isinstance(restart_payload.get("partition"), dict):
        logging.info(
            f"Ignoring restart payload without partition metadata: {os.path.basename(restart_config_path)}"
        )
        return {
            "enabled": False,
            "resume_step": 0,
            "resume_epoch": 1,
            "resume_batch_offset": 0,
            "partition": None,
            "alpha_comp": None,
            "beta_comm": None,
            "restart_payload": None,
            "restart_transition": None,
            "slowdown_detector_state": None,
            "optimizer_runtime_state": None,
        }

    def _infer_layer_count(model) -> int:
        try:
            if hasattr(model, "to_sequential"):
                seq = model.to_sequential()
                return len(list(seq.children()))
            if isinstance(model, torch.nn.Sequential):
                return len(list(model.children()))
            return len(list(model.children()))
        except Exception:
            return -1

    p = restart_payload.get("partition", {})
    sp = p.get("snet_partition")
    tp = p.get("tnet_partition")
    if isinstance(sp, list) and isinstance(tp, list):
        snet_layers = _infer_layer_count(snet)
        tnet_layers = _infer_layer_count(tnet)
        if snet_layers > 0 and tnet_layers > 0:
            if sum(int(v) for v in sp) != snet_layers or sum(int(v) for v in tp) != tnet_layers:
                logging.warning(
                    "Ignoring restart payload due to partition/model mismatch: "
                    f"snet sum={sum(int(v) for v in sp)} expected={snet_layers}, "
                    f"tnet sum={sum(int(v) for v in tp)} expected={tnet_layers}"
                )
                return {
                    "enabled": False,
                    "resume_step": 0,
                    "resume_epoch": 1,
                    "resume_batch_offset": 0,
                    "partition": None,
                    "alpha_comp": None,
                    "beta_comm": None,
                    "restart_payload": None,
                    "restart_transition": None,
                    "slowdown_detector_state": None,
                    "optimizer_runtime_state": None,
                }

    if restart_payload.get("partition") and not skip_partition_yaml_apply:
        logging.error("🔄 Failover restart detected. Applying new partition to YAML...")
        _apply_restart_partition_to_tspipe_yaml(unparsed_args, restart_payload)
        logging.error("✅ YAML partition application completed")
    elif restart_payload.get("partition") and skip_partition_yaml_apply:
        logging.info("↪️  Partition already applied in first load; skipping YAML update on second call")
    else:
        logging.warning("⚠️ Restart payload missing partition data. YAML not updated.")

    checkpoint_path = restart_payload.get("checkpoint_path") or os.path.join(save_root, "failover_checkpoint_latest.pth")
    resume_step = int(restart_payload.get("step_id", 0))

    # Early return if checkpoint restore is skipped (already done in first call)
    if skip_checkpoint_restore:
        partition = restart_payload.get("partition", {})
        partition_cfg = None
        if isinstance(partition.get("snet_partition"), list) and isinstance(partition.get("tnet_partition"), list):
            partition_cfg = PartitionConfig(
                snet_partition=partition["snet_partition"],
                tnet_partition=partition["tnet_partition"],
                gpu_assignment=partition.get("gpu_assignment", list(range(len(partition["snet_partition"])))),
            )
        resume_epoch = restart_payload.get("epoch_id")
        if resume_epoch is None:
            resume_epoch = (resume_step // max(1, steps_per_epoch)) + 1
        else:
            resume_epoch = int(resume_epoch)

        resume_batch_offset = restart_payload.get("batch_offset")
        if resume_batch_offset is None:
            resume_batch_offset = resume_step % max(1, steps_per_epoch)
        else:
            resume_batch_offset = int(resume_batch_offset)

        return {
            "enabled": True,
            "resume_step": resume_step,
            "resume_epoch": resume_epoch,
            "resume_batch_offset": resume_batch_offset,
            "partition": partition_cfg,
            "alpha_comp": None,  # Skip alpha/beta on second call
            "beta_comm": None,
            "restart_payload": restart_payload,
            "restart_transition": restart_payload.get("partition_transition"),
            "slowdown_detector_state": restart_payload.get("slowdown_detector_state"),
            "optimizer_runtime_state": restart_payload.get("optimizer_runtime_state"),
        }

    def _load_state_dict_compat(model, state_dict, model_name: str) -> bool:
        """Load checkpoint state dict across common prefix variants (module./resnet.)."""
        if not isinstance(state_dict, dict):
            logging.warning(f"{model_name} checkpoint is not a state_dict dict")
            return False

        def _add_prefix(sd, prefix):
            return {f"{prefix}{k}": v for k, v in sd.items()}

        def _strip_prefix(sd, prefix):
            out = {}
            for k, v in sd.items():
                out[k[len(prefix):] if k.startswith(prefix) else k] = v
            return out

        candidates = [state_dict]
        candidates.append(_strip_prefix(state_dict, "module."))
        candidates.append(_add_prefix(state_dict, "module."))
        candidates.append(_strip_prefix(state_dict, "resnet."))
        candidates.append(_add_prefix(state_dict, "resnet."))
        candidates.append(_strip_prefix(_strip_prefix(state_dict, "module."), "resnet."))
        candidates.append(_add_prefix(_strip_prefix(state_dict, "module."), "resnet."))
        candidates.append(_strip_prefix(_add_prefix(state_dict, "module."), "resnet."))
        candidates.append(_add_prefix(_add_prefix(state_dict, "module."), "resnet."))

        seen = set()
        unique_candidates = []
        for cand in candidates:
            sig = tuple(sorted(cand.keys()))
            if sig in seen:
                continue
            seen.add(sig)
            unique_candidates.append(cand)

        for cand in unique_candidates:
            try:
                model.load_state_dict(cand, strict=True)
                return True
            except RuntimeError:
                continue

        # Last resort: non-strict load so failover restart can continue.
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logging.warning(
                f"{model_name} loaded with strict=False (missing={len(missing)}, unexpected={len(unexpected)})"
            )
            return True
        except RuntimeError as e:
            logging.error(f"{model_name} checkpoint load failed after compatibility attempts: {e}")
            return False

    # ✅ NEW: Initialize variables for alpha/beta restoration
    alpha_comp_restored = None
    beta_comm_restored = None
    slowdown_detector_state = restart_payload.get("slowdown_detector_state")
    optimizer_runtime_state = restart_payload.get("optimizer_runtime_state")
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "student_state_dict" in ckpt:
            _load_state_dict_compat(snet, ckpt["student_state_dict"], "student")
        if "teacher_state_dict" in ckpt:
            _load_state_dict_compat(tnet, ckpt["teacher_state_dict"], "teacher")
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except RuntimeError as e:
                logging.warning(f"Optimizer state mismatch after partition change, skipping: {e}")
        resume_step = int(ckpt.get("global_step", resume_step))

        if slowdown_detector_state is None and "slowdown_detector_state" in ckpt:
            slowdown_detector_state = ckpt.get("slowdown_detector_state")
        if optimizer_runtime_state is None and "optimizer_runtime_state" in ckpt:
            optimizer_runtime_state = ckpt.get("optimizer_runtime_state")
        
        # ✅ NEW: Restore alpha/beta GPU performance coefficients
        if "alpha_comp" in ckpt:
            alpha_comp_restored = {int(k): float(v) for k, v in ckpt["alpha_comp"].items()}
            logging.info(f"✅ Restored alpha_comp from checkpoint: {alpha_comp_restored}")
        if "beta_comm" in ckpt:
            beta_comm_restored = {int(k): float(v) for k, v in ckpt["beta_comm"].items()}
            logging.info(f"✅ Restored beta_comm from checkpoint: {beta_comm_restored}")
        
        # 또한 emergency_restart_config.json에서도 alpha/beta를 로드
        emergency_config_path = os.path.join(save_root, 'emergency_restart_config.json')
        if os.path.exists(emergency_config_path):
            try:
                with open(emergency_config_path, 'r') as f:
                    emergency_config = json.load(f)
                if "alpha_comp" in emergency_config and alpha_comp_restored is None:
                    alpha_comp_restored = {int(k): float(v) for k, v in emergency_config["alpha_comp"].items()}
                    logging.info(f"✅ Restored alpha_comp from emergency config: {alpha_comp_restored}")
                if "beta_comm" in emergency_config and beta_comm_restored is None:
                    beta_comm_restored = {int(k): float(v) for k, v in emergency_config["beta_comm"].items()}
                    logging.info(f"✅ Restored beta_comm from emergency config: {beta_comm_restored}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to load emergency config: {e}")
    else:
        logging.warning(f"restart_config found but checkpoint missing: {checkpoint_path}")

    partition = restart_payload.get("partition", {})
    partition_cfg = None
    if isinstance(partition.get("snet_partition"), list) and isinstance(partition.get("tnet_partition"), list):
        partition_cfg = PartitionConfig(
            snet_partition=[int(v) for v in partition.get("snet_partition", [])],
            tnet_partition=[int(v) for v in partition.get("tnet_partition", [])],
            gpu_assignment=[int(v) for v in partition.get("gpu_assignment", [])],
        )

    resume_epoch = (resume_step // max(1, steps_per_epoch)) + 1
    resume_batch_offset = resume_step % max(1, steps_per_epoch)

    return {
        "enabled": True,
        "resume_step": resume_step,
        "resume_epoch": resume_epoch,
        "resume_batch_offset": resume_batch_offset,
        "partition": partition_cfg,
        "alpha_comp": alpha_comp_restored,
        "beta_comm": beta_comm_restored,
        "restart_payload": restart_payload,
        "restart_transition": restart_payload.get("partition_transition"),
        "slowdown_detector_state": slowdown_detector_state,
        "optimizer_runtime_state": optimizer_runtime_state,
    }


_resume_target_epoch = 0
_resume_batches_to_skip = 0


def _should_skip_resume_batch(epoch: int) -> bool:
    """Skip already completed batches when resuming from a mid-epoch checkpoint."""
    global _resume_target_epoch, _resume_batches_to_skip
    if epoch == _resume_target_epoch and _resume_batches_to_skip > 0:
        _resume_batches_to_skip -= 1
        return True
    return False


def _prepare_epoch_iterator(train_loader, epoch: int, max_steps_per_epoch: int = 0):
    """Create an epoch-local iterator that respects resume offset and max-steps budget."""
    global _resume_target_epoch, _resume_batches_to_skip

    effective_total = len(train_loader)
    if max_steps_per_epoch > 0:
        effective_total = min(effective_total, int(max_steps_per_epoch))

    resume_offset = 0
    if epoch == _resume_target_epoch and _resume_batches_to_skip > 0:
        resume_offset = min(int(_resume_batches_to_skip), effective_total)
        _resume_batches_to_skip = 0

    batch_sampler = _ResumeAwareBatchSampler(
        dataset_len=len(train_loader.dataset),
        batch_size=int(train_loader.batch_size),
        epoch=epoch,
        start_batch=resume_offset,
        max_batches=effective_total,
        shuffle=True,
        drop_last=bool(getattr(train_loader, 'drop_last', False)),
        seed=random_seed,
    )

    loader_kwargs = {
        'dataset': train_loader.dataset,
        'batch_sampler': batch_sampler,
        'num_workers': train_loader.num_workers,
        'pin_memory': train_loader.pin_memory,
        'collate_fn': train_loader.collate_fn,
    }
    if train_loader.num_workers > 0:
        loader_kwargs['persistent_workers'] = getattr(train_loader, 'persistent_workers', False)

    worker_init_fn = getattr(train_loader, 'worker_init_fn', None)
    if worker_init_fn is not None:
        loader_kwargs['worker_init_fn'] = worker_init_fn

    loader_generator = torch.Generator()
    loader_generator.manual_seed(random_seed + max(0, epoch - 1))
    loader_kwargs['generator'] = loader_generator

    if resume_offset > 0:
        logging.info(
            f"↪️ Fast resume enabled: seeking directly to batch {resume_offset}/{effective_total} "
            "without consuming skipped batches from the DataLoader front"
        )

    epoch_loader = torch.utils.data.DataLoader(**loader_kwargs)
    return epoch_loader, resume_offset, effective_total




def _partition_config_from_payload(partition_payload: Optional[Dict[str, Any]]) -> Optional[PartitionConfig]:
    if not isinstance(partition_payload, dict):
        return None
    sp = partition_payload.get("snet_partition")
    tp = partition_payload.get("tnet_partition")
    if not isinstance(sp, list) or not isinstance(tp, list):
        return None
    return PartitionConfig(
        snet_partition=[int(v) for v in sp],
        tnet_partition=[int(v) for v in tp],
        gpu_assignment=[int(v) for v in partition_payload.get("gpu_assignment", list(range(len(sp))))],
    )


def _estimate_restart_detector_baseline_ms(
    failover_optimizer: Optional[MathematicalFailoverOptimizer],
    detector_state: Optional[Dict[str, Any]],
    restart_transition: Optional[Dict[str, Any]],
) -> Optional[float]:
    if failover_optimizer is None or not isinstance(detector_state, dict):
        return None

    baseline_ms = detector_state.get("baseline_stage_time_ms")
    if baseline_ms is None:
        return None

    if not isinstance(restart_transition, dict):
        return float(baseline_ms)

    previous_partition = _partition_config_from_payload(restart_transition.get("previous_partition"))
    current_partition = getattr(failover_optimizer, "current_partition", None)
    if previous_partition is None or current_partition is None:
        return float(baseline_ms)

    previous_nominal = restart_transition.get("previous_nominal_step_time")
    if previous_nominal is None:
        try:
            previous_nominal = failover_optimizer.estimate_partition_nominal_step_time(previous_partition)
        except Exception:
            previous_nominal = None

    new_nominal = restart_transition.get("new_nominal_step_time")
    if new_nominal is None:
        try:
            new_nominal = failover_optimizer.estimate_partition_nominal_step_time(current_partition)
        except Exception:
            new_nominal = None

    try:
        previous_nominal = float(previous_nominal)
        new_nominal = float(new_nominal)
    except (TypeError, ValueError):
        return float(baseline_ms)

    if previous_nominal <= 0 or not np.isfinite(previous_nominal):
        return float(baseline_ms)
    if new_nominal <= 0 or not np.isfinite(new_nominal):
        return float(baseline_ms)

    scaled = float(baseline_ms) * (new_nominal / previous_nominal)
    if not np.isfinite(scaled) or scaled <= 0:
        return float(baseline_ms)
    return scaled


def _restore_post_restart_monitoring(
    slowdown_detector: Optional[SlowdownDetector],
    failover_optimizer: Optional[MathematicalFailoverOptimizer],
    detector_state: Optional[Dict[str, Any]],
    optimizer_state: Optional[Dict[str, Any]],
    restart_transition: Optional[Dict[str, Any]],
) -> None:
    if failover_optimizer is not None and isinstance(optimizer_state, dict):
        failover_optimizer.restore_restart_state(optimizer_state)

    if slowdown_detector is None or not isinstance(detector_state, dict):
        return

    slowdown_detector.restore_restart_state(
        detector_state,
        clear_recent_window=True,
        clear_trigger_state=True,
    )
    adjusted_baseline_ms = _estimate_restart_detector_baseline_ms(
        failover_optimizer,
        detector_state,
        restart_transition,
    )
    if adjusted_baseline_ms is not None:
        baseline_std_ms = detector_state.get("baseline_std_ms")
        slowdown_detector.set_baseline(
            baseline_stage_time_ms=adjusted_baseline_ms,
            baseline_std_ms=0.0 if baseline_std_ms is None else float(baseline_std_ms),
            clear_recent_window=True,
            clear_trigger_state=True,
        )
        logging.info(
            "🧭 Post-restart wall-clock baseline adjusted for new partition: "
            f"{adjusted_baseline_ms:.2f}ms"
        )


def _get_post_restart_observation_steps(
    restart_payload: Optional[Dict[str, Any]],
    slowdown_detector: Optional[SlowdownDetector],
) -> int:
    env_steps = os.environ.get("FAILOVER_POST_RESTART_OBSERVATION_STEPS", "").strip()
    if env_steps:
        try:
            return max(0, int(env_steps))
        except ValueError:
            pass

    if isinstance(restart_payload, dict):
        payload_steps = restart_payload.get("post_restart_observation_steps")
        if payload_steps is not None:
            try:
                return max(0, int(payload_steps))
            except (TypeError, ValueError):
                pass

    detector_steps = 0 if slowdown_detector is None else int(getattr(slowdown_detector, "detection_window", 0) or 0)
    return max(5, detector_steps * 2 if detector_steps > 0 else 10)

def _run_dryrun_failover_cycle():
    """Deterministic one-cycle failover dry-run for E2E launcher validation."""
    global niter
    logging.error("[DRYRUN] Starting E2E failover dry-run mode")

    # Tiny dummy models/optimizer for checkpoint serialization path validation.
    snet = torch.nn.Sequential(torch.nn.Linear(4, 4))
    tnet = torch.nn.Sequential(torch.nn.Linear(4, 4))
    optimizer = torch.optim.SGD(snet.parameters(), lr=0.01)

    bootstrap = _load_failover_bootstrap(
        save_root=args.save_root,
        unparsed_args=unparsed,
        snet=snet,
        tnet=tnet,
        optimizer=optimizer,
        steps_per_epoch=10,
    )
    niter = int(bootstrap["resume_step"])

    failover_optimizer = MathematicalFailoverOptimizer(total_epochs=1, steps_per_epoch=10)
    # Keep assignment single-GPU for deterministic launcher nproc sync in dry-run.
    failover_optimizer.current_partition = PartitionConfig(
        snet_partition=[failover_optimizer.snet_num_layers],
        tnet_partition=[failover_optimizer.tnet_num_layers],
        gpu_assignment=[0],
    )

    def _save_failover_checkpoint(meta):
        checkpoint_path = os.path.join(args.save_root, 'failover_checkpoint_latest.pth')
        state = {
            'global_step': niter,
            'policy': meta.get('trigger_policy'),
            'partition': meta.get('partition'),
            'alpha_comp': meta.get('alpha_comp'),
            'beta_comm': meta.get('beta_comm'),
            'student_state_dict': snet.state_dict(),
            'teacher_state_dict': tnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': time.time(),
        }
        torch.save(state, checkpoint_path)
        logging.error(f"[DRYRUN] Failover checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    failover_optimizer.configure_failover_restart(
        restart_config_path=os.path.join(args.save_root, 'failover_restart_config.json'),
        checkpoint_saver=_save_failover_checkpoint,
        auto_restart_on_failover=True,
    )

    if not bootstrap["enabled"]:
        # First boot: inject a deterministic failover event.
        for step in range(0, 6):
            niter = step
            failover_optimizer.update_training_progress(step, 0)
        logging.error("[DRYRUN] Injecting synthetic slowdown spike -> forcing REPLAN")
        failover_optimizer.execute_policy("REPLAN", gpu_id=0, current_slowdown=3.0)
        return

    # Restarted boot: verify resume and continue a few steps, then exit 0.
    logging.error(
        f"Failover recovery successful. Resuming training from step [{niter}] with new partition."
    )
    for _ in range(3):
        prev = niter
        niter += 1
        logging.error(f"[DRYRUN] Resume progress: step {prev} -> {niter}")
    logging.error("[DRYRUN] Dry-run cycle completed successfully")


def dummy_target_update(m, online_new_param: Optional[Iterable[torch.Tensor]], 
                        target_param: Optional[Iterable[torch.nn.Parameter]] = None):
    return target_param


def main():
    global niter, _resume_target_epoch, _resume_batches_to_skip
    _validate_slowdown_cli_args()
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    # [Port Setup] Use environment variable set by launcher (run_e2e_failover.sh)
    # CRITICAL: Do NOT call find_free_port() here - it would override the launcher's port!
    # The launcher sets PYTORCH_DISTRIBUTED_NCCL_START_PORT per restart iteration.
    # Port spacing: 31200, 31300, 31400... (increment of 100) to avoid TIME_WAIT conflicts
    try:
        import os
        nccl_port = os.environ.get('PYTORCH_DISTRIBUTED_NCCL_START_PORT')
        if nccl_port:
            logging.info(f"✅ [Port Setup] Using launcher-provided NCCL port: {nccl_port} (spacing=100, expected sequence: 31200, 31300, 31400...)")
        else:
            logging.warning(f"⚠️ [Port Setup] No NCCL port from launcher, tspipe.communicator will allocate default")
    except Exception as e:
        logging.warning(f"Port check (non-fatal): {e}")

    if args.dryrun_failover_cycle:
        _run_dryrun_failover_cycle()
        return

    logging.info('----------- Network Initialization --------------')
    image_size = 224 if 'cifar' not in args.data_name else 32
    snet = create_model(args.s_name, num_class=args.num_class, image_size=image_size)
    checkpoint = torch.load(args.s_init, 'cpu')
    load_pretrained_model(snet, checkpoint['net'])
    # logging.info('Student: %s', snet)
    logging.info('Student param size = %fMB, %d params', count_parameters_in_MB(snet), count_parameters(snet))

    tnet = create_model(args.t_name, num_class=args.num_class, image_size=image_size)
    checkpoint = torch.load(args.t_model, 'cpu')
    load_pretrained_model(tnet, checkpoint['net'])
    tnet.eval()
    for param in tnet.parameters():
        param.requires_grad = False
    # logging.info('Teacher: %s', tnet)
    logging.info('Teacher param size = %fMB, %d params', count_parameters_in_MB(tnet), count_parameters(tnet))
    logging.info('-----------------------------------------------')

    # define loss functions
    if args.kd_mode == 'logits':
        criterionKD = Logits()
    elif args.kd_mode == 'st':
        criterionKD = SoftTarget(args.T)
    else:
        raise Exception('Invalid kd mode...')
    if args.cuda:
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = torch.optim.SGD(snet.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # define transforms
    if args.data_name == 'cifar10':
        dataset = dst.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_dataset = partial(dataset, train=True, download=True)
        test_dataset = partial(dataset, train=False, download=True)
    elif args.data_name == 'cifar100':
        dataset = dst.CIFAR100
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
        train_dataset = partial(dataset, train=True, download=True)
        test_dataset = partial(dataset, train=False, download=True)
    elif args.data_name == 'imagenet100':
        dataset = small_datasets.ImageNet100
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_dataset, test_dataset = partial(dataset, split='train'), partial(dataset, split='val')
    else:
        raise Exception('Invalid dataset name...')

    if 'cifar' in args.data_name:
        train_transform = transforms.Compose([
                transforms.Pad(4, padding_mode='reflect'),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        test_transform = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    else:
        train_transform = transforms.Compose([
                transforms.Pad(4, padding_mode='reflect'),
                transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        test_transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    # define data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset(root=args.img_root, transform=train_transform),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset(root=args.img_root, transform=test_transform),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Bootstrap failover recovery state before pipeline initialization.
    # NOTE: This first call is the only place where we restore alpha_comp/beta_comm
    # from the failover checkpoint. We must preserve these values across the
    # second bootstrap call below (which skips checkpoint restore) so that
    # MathematicalFailoverOptimizer can reuse the learned GPU coefficients.
    bootstrap = _load_failover_bootstrap(
        save_root=args.save_root,
        unparsed_args=unparsed,
        snet=snet,
        tnet=tnet,
        optimizer=optimizer,
        steps_per_epoch=len(train_loader),
    )
    # Preserve restored alpha/beta from the first bootstrap call. The second
    # call with skip_checkpoint_restore=True will intentionally return None
    # for these fields, so we cache them here for later use when wiring up
    # failover_optimizer.
    bootstrap_alpha_comp = bootstrap["alpha_comp"]
    bootstrap_beta_comm = bootstrap["beta_comm"]
    bootstrap_restart_payload = bootstrap.get("restart_payload")
    bootstrap_restart_transition = bootstrap.get("restart_transition")
    bootstrap_detector_state = bootstrap.get("slowdown_detector_state")
    bootstrap_optimizer_state = bootstrap.get("optimizer_runtime_state")
    niter = int(bootstrap["resume_step"])
    start_epoch = int(bootstrap["resume_epoch"])
    _resume_target_epoch = start_epoch
    _resume_batches_to_skip = int(bootstrap["resume_batch_offset"])

    _configure_environment_injected_slowdown(total_steps=len(train_loader) * max(1, args.epochs))

    # initialize tspipe (if needed)
    slowdown_detector = None  # NEW: Initialize slowdown detector
    failover_optimizer = None # NEW: Initialize math-based optimizer
    timing_ingestor = None
    
    if args.tspipe:
        if not isinstance(snet, torch.nn.Sequential):
            snet = snet.to_sequential()
        if not isinstance(tnet, torch.nn.Sequential):
            tnet = tnet.to_sequential()
        
        optimizer = torch.optim.SGD(snet.parameters(),
                                    lr = args.lr,
                                    momentum = args.momentum, 
                                    weight_decay = args.weight_decay,
                                    nesterov = True)

        # Re-load bootstrap state on the final (sequential) model/optimizer objects.
        # ⚠️ Skip YAML update AND checkpoint restore: already applied in first call above.
        # This second call is used to validate partition compatibility against the
        # sequentialized models and to rebuild PartitionConfig, but it does NOT
        # carry alpha/beta (those were restored only in the first call above and
        # cached in bootstrap_alpha_comp/bootstrap_beta_comm).
        bootstrap = _load_failover_bootstrap(
            save_root=args.save_root,
            unparsed_args=unparsed,
            snet=snet,
            tnet=tnet,
            optimizer=optimizer,
            steps_per_epoch=len(train_loader),
            skip_partition_yaml_apply=True,
            skip_checkpoint_restore=True,
        )
        bootstrap_restart_payload = bootstrap.get("restart_payload") or bootstrap_restart_payload
        bootstrap_restart_transition = bootstrap.get("restart_transition") or bootstrap_restart_transition
        bootstrap_detector_state = bootstrap.get("slowdown_detector_state") or bootstrap_detector_state
        bootstrap_optimizer_state = bootstrap.get("optimizer_runtime_state") or bootstrap_optimizer_state
        niter = int(bootstrap["resume_step"])
        start_epoch = int(bootstrap["resume_epoch"])
        _resume_target_epoch = start_epoch
        _resume_batches_to_skip = int(bootstrap["resume_batch_offset"])

        # Keep worker-side slowdown injection aligned to the resumed global step.
        args.resume_step_offset = int(niter)

        tspipe_trainer = TSPipe(
            snet,
            tnet,
            None,
            optimizer,
            tspipe_loss,
            dummy_target_update,
            1,
            artifact_dir = args.save_root,
            tspipe_mode=TSPipeMode.SUPERVISED_MOMENTUM,
            target_train_mode=False,
            extra_args=args
        )
        assert args.kd_mode == 'logits' or args.kd_mode == 'st'

        if bootstrap["enabled"]:
            tspipe_trainer.batch_count = int(niter)
            if (
                tspipe_trainer.target_fail_gpu >= 0
                and tspipe_trainer.fail_after_batches > 0
                and tspipe_trainer.batch_count >= tspipe_trainer.fail_after_batches
            ):
                tspipe_trainer.failure_simulated = True
                logging.error(
                    "↪️ Resumed beyond scheduled hard-failure point; "
                    f"disabling re-simulation (resume_step={niter}, fail_after={tspipe_trainer.fail_after_batches})"
                )
        
        # NEW: Initialize slowdown detection and failover policy components
        slowdown_detector = SlowdownDetector(inject_scenario=args.failover_inject_scenario, slowdown_threshold=1.10,)
        
        # ✅ Step 1: Pass YAML partition config to optimizer to sync initial state
        # This ensures failover decisions use the actual runtime partition from start
        # Use tspipe_trainer.config which was loaded from YAML in TSPipe.__init__()
        yaml_partition_config = PartitionConfig(
            snet_partition=tspipe_trainer.config['model_split']['online'],
            tnet_partition=tspipe_trainer.config['model_split']['target'],
            gpu_assignment=list(range(len(tspipe_trainer.config['model_split']['online'])))
        )
        
        failover_optimizer = MathematicalFailoverOptimizer(
            total_epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            initial_partition_config=yaml_partition_config  # ← NEW: Pass YAML partition from tspipe
        )
        
        # If restarting from failover, override with bootstrap partition and
        # reapply previously restored alpha/beta coefficients.
        if failover_optimizer is not None and bootstrap["partition"] is not None:
            failover_optimizer.current_partition = bootstrap["partition"]
            _restore_post_restart_monitoring(
                slowdown_detector=slowdown_detector,
                failover_optimizer=failover_optimizer,
                detector_state=bootstrap_detector_state,
                optimizer_state=bootstrap_optimizer_state,
                restart_transition=bootstrap_restart_transition,
            )

            # ✅ Restore alpha/beta coefficients captured from the first
            # bootstrap call (which read them from the checkpoint or
            # emergency_restart_config). The second bootstrap call deliberately
            # returns None for these fields when skip_checkpoint_restore=True.
            if bootstrap_alpha_comp is not None:
                failover_optimizer.alpha_g.clear()
                failover_optimizer.alpha_g.update(bootstrap_alpha_comp)
                if getattr(failover_optimizer, "alpha_beta_estimator", None) is not None:
                    failover_optimizer.alpha_beta_estimator.alpha_g.clear()
                    failover_optimizer.alpha_beta_estimator.alpha_g.update(bootstrap_alpha_comp)
                logging.error(f"🔄 Restored alpha_g from checkpoint: {failover_optimizer.alpha_g}")

            if bootstrap_beta_comm is not None:
                failover_optimizer.beta_g.clear()
                failover_optimizer.beta_g.update(bootstrap_beta_comm)
                if getattr(failover_optimizer, "alpha_beta_estimator", None) is not None:
                    failover_optimizer.alpha_beta_estimator.beta_g.clear()
                    failover_optimizer.alpha_beta_estimator.beta_g.update(bootstrap_beta_comm)
                logging.error(f"🔄 Restored beta_g from checkpoint: {failover_optimizer.beta_g}")

        if failover_optimizer is not None:
            _persist_latest_failover_coefficients(
                save_root=args.save_root,
                alpha_comp=failover_optimizer.alpha_g,
                beta_comm=failover_optimizer.beta_g,
                partition=getattr(failover_optimizer, "current_partition", None),
                source="bootstrap_restore" if bootstrap["enabled"] else "initial_defaults",
                step_id=niter,
            )

        def _save_failover_checkpoint(meta):
            checkpoint_path = os.path.join(args.save_root, 'failover_checkpoint_latest.pth')
            detector_restart_state = None
            if slowdown_detector is not None and hasattr(slowdown_detector, 'export_restart_state'):
                detector_restart_state = slowdown_detector.export_restart_state()
                meta['slowdown_detector_state'] = detector_restart_state

            optimizer_restart_state = None
            if failover_optimizer is not None and hasattr(failover_optimizer, 'export_restart_state'):
                optimizer_restart_state = failover_optimizer.export_restart_state()
                meta['optimizer_runtime_state'] = optimizer_restart_state

            meta.setdefault('post_restart_observation_steps', _get_post_restart_observation_steps(None, slowdown_detector))

            state = {
                'global_step': niter,
                'policy': meta.get('trigger_policy'),
                'partition': meta.get('partition'),
                'alpha_comp': meta.get('alpha_comp'),
                'beta_comm': meta.get('beta_comm'),
                'slowdown_detector_state': detector_restart_state,
                'optimizer_runtime_state': optimizer_restart_state,
                'student_state_dict': snet.state_dict(),
                'teacher_state_dict': tnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'timestamp': time.time(),
            }
            torch.save(state, checkpoint_path)
            logging.error(f"💾 Failover checkpoint saved: {checkpoint_path}")
            return checkpoint_path

        if failover_optimizer is not None:
            failover_optimizer.configure_failover_restart(
                restart_config_path=os.path.join(args.save_root, 'failover_restart_config.json'),
                checkpoint_saver=_save_failover_checkpoint,
                auto_restart_on_failover=True,
            )
        timing_ingestor = RuntimeTimingIngestor(os.path.join(args.save_root, 'profiling_logs'))
        logging.info("✅ Failover monitoring components initialized")
    else:
        snet = snet.cuda()
        tnet = tnet.cuda()  

    post_restart_observation_steps = 0
    failover_resume_gate_step = niter
    if bootstrap["enabled"]:
        post_restart_observation_steps = _get_post_restart_observation_steps(
            bootstrap_restart_payload,
            slowdown_detector,
        )
        failover_resume_gate_step = niter + post_restart_observation_steps
        logging.error(
            f"Failover recovery successful. Resuming training from step [{niter}] with new partition."
        )
        logging.info(
            "🕒 Post-restart observation window enabled: "
            f"{post_restart_observation_steps} steps (policy reevaluation resumes at step {failover_resume_gate_step})"
        )

    writer = SummaryWriter()

    # warp nets and criterions for train and test
    nets = {'snet': snet, 'tnet': tnet}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD}

    # first initilizing the student nets
    if args.kd_mode in ['fsp', 'ab']:
        logging.info('The first stage, student initialization......')
        train_init(train_loader, nets, optimizer, criterions, 50)
        args.lambda_kd = 0.0
        logging.info('The second stage, softmax training......')

    best_top1 = 0
    best_top5 = 0
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_lr(optimizer, epoch)

        # train one epoch
        epoch_start_time = time.time()
        if args.tspipe:
            for param_group in optimizer.param_groups:
                tspipe_trainer.update_lr(param_group['lr'])
                break
            # NEW: Pass failover components to training function
            train_tspipe(tspipe_trainer, train_loader, nets, optimizer, criterions, epoch, writer,
                         slowdown_detector=slowdown_detector,
                         failover_optimizer=failover_optimizer,
                         timing_ingestor=timing_ingestor,
                         failover_resume_gate_step=failover_resume_gate_step,
                         max_steps_per_epoch=args.max_steps_per_epoch)
        else:
            train(train_loader, nets, optimizer, criterions, epoch, writer)

        continue
    if args.tspipe:        
        tspipe_trainer.stop()


def train_init(train_loader, nets, optimizer, criterions, total_epoch):
    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionKD  = criterions['criterionKD']

    snet.train()

    for epoch in range(1, total_epoch+1):
        adjust_lr_init(optimizer, epoch)

        batch_time = AverageMeter()
        data_time  = AverageMeter()
        cls_losses = AverageMeter()
        kd_losses  = AverageMeter()
        top1       = AverageMeter()
        top5       = AverageMeter()

        epoch_start_time = time.time()
        end = time.time()
        for i, (img, target) in enumerate(train_loader, start=1):
            data_time.update(time.time() - end)

            if args.cuda:
                img = img.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = snet(img)
            stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = tnet(img)

            cls_loss = criterionCls(out_s, target) * 0.0
            if args.kd_mode in ['fsp']:
                kd_loss = (criterionKD(stem_s[1], rb1_s[1], stem_t[1].detach(), rb1_t[1].detach()) +
                           criterionKD(rb1_s[1],  rb2_s[1], rb1_t[1].detach(),  rb2_t[1].detach()) +
                           criterionKD(rb2_s[1],  rb3_s[1], rb2_t[1].detach(),  rb3_t[1].detach())) / 3.0 * args.lambda_kd
            elif args.kd_mode in ['ab']:
                kd_loss = (criterionKD(rb1_s[0], rb1_t[0].detach()) +
                           criterionKD(rb2_s[0], rb2_t[0].detach()) +
                           criterionKD(rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
            else:
                raise Exception('Invalid kd mode...')
            loss = cls_loss + kd_loss

            prec1, prec5 = accuracy(out_s, target, topk=(1,5))
            cls_losses.update(cls_loss.item(), img.size(0))
            kd_losses.update(kd_loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
                           'Time:{batch_time.val:.4f} '
                           'Data:{data_time.val:.4f}  '
                           'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
                           'KD:{kd_losses.val:.4f}({kd_losses.avg:.4f})  '
                           'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                           'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                           epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                           cls_losses=cls_losses, kd_losses=kd_losses, top1=top1, top5=top5))
                logging.info(log_str)

        epoch_duration = time.time() - epoch_start_time
        logging.info('Epoch time: {}s'.format(int(epoch_duration)))


def auto_convert(output):
    if isinstance(output, torch.Tensor):
        return None, None, None, None, None, output
    return output


class RuntimeTimingIngestor:
    """Aggregate real per-GPU compute/comm timings from profiler trace files."""

    def __init__(self, profiling_dir: str, window: int = 64):
        self.profiling_dir = Path(profiling_dir)
        self.window = max(8, int(window))
        self._offsets = {}
        self._compute_hist = {}
        self._comm_hist = {}

    def _append_hist(self, store, gpu_id: int, value: float):
        buf = store.get(gpu_id)
        if buf is None:
            from collections import deque
            buf = deque(maxlen=self.window)
            store[gpu_id] = buf
        buf.append(float(value))

    def update(self):
        if not self.profiling_dir.exists():
            return

        for trace_path in self.profiling_dir.glob("gpu_task_summary_partition*.jsonl"):
            key = str(trace_path)
            last_offset = self._offsets.get(key, 0)
            try:
                with open(trace_path, "r") as f:
                    f.seek(last_offset)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        gpu_id = int(record.get("device", -1))
                        if gpu_id < 0:
                            continue

                        raw_exec_ms = record.get("exec_wall_ms", None)
                        if raw_exec_ms is None:
                            raw_exec_ms = record.get("time_ms", 0.0)
                        task_time_ms = float(raw_exec_ms or 0.0)
                        if task_time_ms <= 0:
                            continue

                        task_name = str(record.get("task_name", ""))
                        if task_name.startswith("compute"):
                            self._append_hist(self._compute_hist, gpu_id, task_time_ms)
                        elif task_name.startswith("copy"):
                            self._append_hist(self._comm_hist, gpu_id, task_time_ms)

                    self._offsets[key]=f.tell()
            except OSError:
                continue

    def build_timing_payload(self):
        payload = {}
        recent_n = 20
        all_gpu_ids = set(self._compute_hist.keys()) | set(self._comm_hist.keys())

        for gpu_id in all_gpu_ids:
            c_hist = list(self._compute_hist.get(gpu_id, []))[-recent_n:]
            m_hist = list(self._comm_hist.get(gpu_id, []))[-recent_n:]
            if not c_hist and not m_hist:
                continue

            compute_ms = float(np.median(c_hist)) if c_hist else 0.0
            comm_ms = float(np.median(m_hist)) if m_hist else 0.0

            payload[gpu_id] = {
                "compute_time": compute_ms / 1000.0,
                "comm_time": comm_ms / 1000.0,
            }
        return payload

def train(train_loader, nets, optimizer, criterions, epoch, writer):
    global niter

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    cls_losses = AverageMeter()
    kd_losses  = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionKD  = criterions['criterionKD']

    snet.train()
    if args.kd_mode in ['vid', 'ofd']:
        for i in range(1,4):
            criterionKD[i].train()

    end = time.time()
    global niter
    epoch_iter, resume_offset, effective_total = _prepare_epoch_iterator(
        train_loader,
        epoch,
        args.max_steps_per_epoch,
    )
    if resume_offset >= effective_total:
        logging.info(
            f"↪️ Epoch {epoch} already satisfied quick-run budget "
            f"(resume_offset={resume_offset}, total={effective_total}); skipping train loop"
        )
        return

    pbar = tqdm(epoch_iter, initial=resume_offset, total=effective_total)
    for epoch_step, (img, target) in enumerate(pbar, start=resume_offset + 1):

        data_time.update(time.time() - end)

        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        if args.kd_mode in ['sobolev', 'lwm']:
            img.requires_grad = True

        stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = auto_convert(snet(img))
        stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = auto_convert(tnet(img))

        cls_loss = criterionCls(out_s, target)
        if args.kd_mode in ['logits', 'st']:
            kd_loss = criterionKD(out_s, out_t.detach()) * args.lambda_kd
        else:
            raise Exception('Invalid kd mode...')
        loss = cls_loss + kd_loss
        # print(cls_loss.item(), kd_loss.item(), loss.item())

        writer.add_scalar('loss', loss, global_step=niter)
        pbar.set_postfix({'loss': loss, 'batch_id': niter})
        niter += 1

        prec1, prec5 = accuracy(out_s, target, topk=(1,5))
        cls_losses.update(cls_loss.item(), img.size(0))
        kd_losses.update(kd_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if epoch_step % args.print_freq == 0:
            log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
                       'Time:{batch_time.val:.4f} '
                       'Data:{data_time.val:.4f}  '
                       'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
                       'KD:{kd_losses.val:.4f}({kd_losses.avg:.4f})  '
                       'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                       'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                       epoch, epoch_step, effective_total, batch_time=batch_time, data_time=data_time,
                       cls_losses=cls_losses, kd_losses=kd_losses, top1=top1, top5=top5))
            logging.info(log_str)

def tspipe_loss(model_out: torch.Tensor, ema_model_out: torch.Tensor, label: torch.Tensor, tspipe_args: argparse.Namespace, args: argparse.Namespace, epoch: int):

    if args.kd_mode == 'logits':
        criterionKD = Logits()
    elif args.kd_mode == 'st':
        criterionKD = SoftTarget(args.T)
    else:
        assert False
    
    criterionCls = torch.nn.CrossEntropyLoss()
    
    cls_loss = criterionCls(model_out, label)
    kd_loss = criterionKD(model_out, ema_model_out.detach()) * args.lambda_kd

    return cls_loss + kd_loss

niter = 0
def train_tspipe(tspipe_trainer:TSPipe, train_loader, nets, optimizer, criterions, epoch, writer,
                 slowdown_detector=None, failover_optimizer=None, timing_ingestor=None,
                 failover_resume_gate_step: int = 0,
                 max_steps_per_epoch: int = 0):
    global niter

    epoch_iter, resume_offset, effective_total = _prepare_epoch_iterator(
        train_loader,
        epoch,
        max_steps_per_epoch,
    )
    if resume_offset >= effective_total:
        logging.info(
            f"↪️ Epoch {epoch} already satisfied quick-run budget "
            f"(resume_offset={resume_offset}, total={effective_total}); skipping tspipe loop"
        )
        return

    pbar = tqdm(epoch_iter, initial=resume_offset, total=effective_total)
    wallclock_sustain_sec = float(os.environ.get("FAILOVER_SLOWDOWN_THRESHOLD_SEC", "10.0"))
    trigger_threshold = 1.10

    for epoch_step, (img, target) in enumerate(pbar, start=resume_offset + 1):

        batch_start_time = time.time()
        loss = tspipe_trainer.feed(img, img, target)

        if loss is None:
            continue

        # 1) end-to-end wall-clock step time
        batch_elapsed_time_ms = (time.time() - batch_start_time) * 1000.0

        # 2) progress update
        if failover_optimizer is not None:
            failover_optimizer.update_training_progress(niter, epoch)

        # 3) timing ingestion for localization only
        timing_payload = {}
        if failover_optimizer is not None and timing_ingestor is not None:
            timing_ingestor.update()
            timing_payload = timing_ingestor.build_timing_payload()
            if timing_payload:
                failover_optimizer.ingest_runtime_timing_batch(timing_payload, ema=0.4)

        if failover_optimizer is not None:
            _persist_latest_failover_coefficients(
                save_root=args.save_root,
                alpha_comp=failover_optimizer.alpha_g,
                beta_comm=failover_optimizer.beta_g,
                partition=getattr(failover_optimizer, "current_partition", None),
                source="runtime_update",
                step_id=niter,
            )

        # 4) wall-clock trigger source
        trigger_state = None
        if slowdown_detector is not None:
            try:
                slowdown_detector.record_stage_time(
                    batch_elapsed_time_ms,
                    timestamp_sec=time.time(),
                    global_step=niter,
                )
            except TypeError:
                slowdown_detector.record_stage_time(batch_elapsed_time_ms)

            if hasattr(slowdown_detector, "get_trigger_state"):
                trigger_state = slowdown_detector.get_trigger_state(
                    sustain_sec=wallclock_sustain_sec
                )
            else:
                wall_ratio_fallback = slowdown_detector.get_slowdown_ratio()
                wall_sustained_fallback = float(
                    getattr(slowdown_detector, "wallclock_sustained_duration_sec", 0.0)
                )
                if hasattr(slowdown_detector, "wallclock_sustained_duration_sec"):
                    wall_triggered_fallback = (
                        wall_ratio_fallback > trigger_threshold
                        and wall_sustained_fallback >= wallclock_sustain_sec
                    )
                else:
                    wall_triggered_fallback = (wall_ratio_fallback > trigger_threshold)

                trigger_state = {
                    "current_slowdown_ratio": wall_ratio_fallback,
                    "sustained_duration_sec": wall_sustained_fallback,
                    "triggered": wall_triggered_fallback,
                }

        # 5) 주기적 failover 평가
        failover_gate_active = int(niter) < int(failover_resume_gate_step or 0)
        if (
            failover_optimizer is not None
            and slowdown_detector is not None
            and niter % 10 == 0
            and not failover_gate_active
        ):
            wall_ratio = float(trigger_state["current_slowdown_ratio"]) if trigger_state else 1.0
            wall_sustained = float(trigger_state["sustained_duration_sec"]) if trigger_state else 0.0
            wall_triggered = bool(trigger_state["triggered"]) if trigger_state else False

            if wall_triggered:
                preferred_gpu = None
                if args.inject_slowdown_gpu is not None:
                    preferred_gpu = int(args.inject_slowdown_gpu)

                try:
                    localized_gpu = failover_optimizer.identify_slow_gpu(
                        preferred_gpu=preferred_gpu
                    )
                except TypeError:
                    localized_gpu = failover_optimizer.identify_slow_gpu()

                if hasattr(failover_optimizer, "get_gpu_slowdown"):
                    localized_ratio = failover_optimizer.get_gpu_slowdown(localized_gpu)
                else:
                    localized_ratio = max(
                        failover_optimizer.alpha_g.get(localized_gpu, 1.0),
                        failover_optimizer.beta_g.get(localized_gpu, 1.0),
                    )

                effective_slowdown = max(wall_ratio, localized_ratio, trigger_threshold)

                logging.info(
                    f"⚠️ Wall-clock trigger confirmed at step {niter}: "
                    f"wall_ratio={wall_ratio:.2f}x, sustained={wall_sustained:.1f}s, "
                    f"localized_gpu={localized_gpu}, localized_ratio={localized_ratio:.2f}x, "
                    f"effective={effective_slowdown:.2f}x"
                )

                try:
                    policy = failover_optimizer.evaluate_slowdown_and_decide(
                        gpu_id=localized_gpu,
                        current_slowdown=effective_slowdown,
                        trigger_confirmed=True,
                    )
                except TypeError:
                    policy = failover_optimizer.evaluate_slowdown_and_decide(
                        gpu_id=localized_gpu,
                        current_slowdown=effective_slowdown,
                    )

                logging.info(
                    f"🎯 Failover Policy Decision: {policy} "
                    f"(step={niter}, gpu={localized_gpu}, "
                    f"wall_ratio={wall_ratio:.2f}x, localized_ratio={localized_ratio:.2f}x)"
                )

                if policy != "KEEP":
                    failover_optimizer.execute_policy(
                        policy,
                        gpu_id=localized_gpu,
                        current_slowdown=effective_slowdown,
                    )
            else:
                if wall_ratio > 1.05:
                    logging.info(
                        f"⏳ Wall-clock slowdown pending: "
                        f"ratio={wall_ratio:.2f}x, sustained={wall_sustained:.1f}s/"
                        f"{wallclock_sustain_sec:.1f}s, step={niter}"
                    )
        elif failover_gate_active and niter % 10 == 0:
            remaining_gate = int(failover_resume_gate_step) - int(niter)
            logging.info(
                "🕒 Post-restart observation active: "
                f"{max(0, remaining_gate)} steps until failover reevaluation resumes (step={niter})"
            )

        writer.add_scalar('loss', loss, global_step=niter)
        pbar.set_postfix({'loss': loss, 'batch_id': niter})

        niter += 1


def test(test_loader, nets, criterions, epoch):
    cls_losses = AverageMeter()
    kd_losses  = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionKD  = criterions['criterionKD']

    snet.eval()
    if args.kd_mode in ['vid', 'ofd']:
        for i in range(1,4):
            criterionKD[i].eval()

    end = time.time()
    for i, (img, target) in enumerate(test_loader, start=1):
        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        if args.kd_mode in ['sobolev', 'lwm']:
            img.requires_grad = True
            stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = snet(img)
            stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = tnet(img)
        else:
            with torch.no_grad():
                stem_s, rb1_s, rb2_s, rb3_s, feat_s, out_s = (None, None, None, None, None, snet(img))
                stem_t, rb1_t, rb2_t, rb3_t, feat_t, out_t = (None, None, None, None, None, tnet(img))

        cls_loss = criterionCls(out_s, target)
        if args.kd_mode in ['logits', 'st']:
            kd_loss  = criterionKD(out_s, out_t.detach()) * args.lambda_kd
        elif args.kd_mode in ['fitnet', 'nst']:
            kd_loss = criterionKD(rb3_s[1], rb3_t[1].detach()) * args.lambda_kd
        elif args.kd_mode in ['at', 'sp']:
            kd_loss = (criterionKD(rb1_s[1], rb1_t[1].detach()) +
                       criterionKD(rb2_s[1], rb2_t[1].detach()) +
                       criterionKD(rb3_s[1], rb3_t[1].detach())) / 3.0 * args.lambda_kd
        elif args.kd_mode in ['pkt', 'rkd', 'cc']:
            kd_loss = criterionKD(feat_s, feat_t.detach()) * args.lambda_kd
        elif args.kd_mode in ['fsp']:
            kd_loss = (criterionKD(stem_s[1], rb1_s[1], stem_t[1].detach(), rb1_t[1].detach()) +
                       criterionKD(rb1_s[1],  rb2_s[1], rb1_t[1].detach(),  rb2_t[1].detach()) +
                       criterionKD(rb2_s[1],  rb3_s[1], rb2_t[1].detach(),  rb3_t[1].detach())) / 3.0 * args.lambda_kd
        elif args.kd_mode in ['ab']:
            kd_loss = (criterionKD(rb1_s[0], rb1_t[0].detach()) +
                       criterionKD(rb2_s[0], rb2_t[0].detach()) +
                       criterionKD(rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
        elif args.kd_mode in ['sobolev']:
            kd_loss = criterionKD(out_s, out_t, img, target) * args.lambda_kd
        elif args.kd_mode in ['lwm']:
            kd_loss = criterionKD(out_s, rb2_s[1], out_t, rb2_t[1], target) * args.lambda_kd
        elif args.kd_mode in ['irg']:
            kd_loss = criterionKD([rb2_s[1], rb3_s[1], feat_s, out_s],
                                  [rb2_t[1].detach(),
                                   rb3_t[1].detach(),
                                   feat_t.detach(), 
                                   out_t.detach()]) * args.lambda_kd
        elif args.kd_mode in ['vid', 'afd']:
            kd_loss = (criterionKD[1](rb1_s[1], rb1_t[1].detach()) +
                       criterionKD[2](rb2_s[1], rb2_t[1].detach()) +
                       criterionKD[3](rb3_s[1], rb3_t[1].detach())) / 3.0 * args.lambda_kd
        elif args.kd_mode in ['ofd']:
            kd_loss = (criterionKD[1](rb1_s[0], rb1_t[0].detach()) +
                       criterionKD[2](rb2_s[0], rb2_t[0].detach()) +
                       criterionKD[3](rb3_s[0], rb3_t[0].detach())) / 3.0 * args.lambda_kd
        else:
            raise Exception('Invalid kd mode...')

        prec1, prec5 = accuracy(out_s, target, topk=(1,5))
        cls_losses.update(cls_loss.item(), img.size(0))
        kd_losses.update(kd_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    f_l = [cls_losses.avg, kd_losses.avg, top1.avg, top5.avg]
    logging.info('Cls: {:.4f}, KD: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

    return top1.avg, top5.avg


def adjust_lr_init(optimizer, epoch):
    scale   = 0.1
    lr_list = [args.lr*scale] * 30
    lr_list += [args.lr*scale*scale] * 10
    lr_list += [args.lr*scale*scale*scale] * 10

    lr = lr_list[epoch-1]
    logging.info('Epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr(optimizer, epoch):
    scale   = 0.1
    lr_list =  [args.lr] * 100
    lr_list += [args.lr*scale] * 50
    lr_list += [args.lr*scale*scale] * 50

    lr = lr_list[epoch-1]
    logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    try:
        main()

    except SystemExit as e:
        code = getattr(e, "code", 1)

        if code == 42:
            cleanup_start = time.monotonic()
            logging.error("Failover detected (exit code 42). Exiting for launcher restart.")

            try:
                import os
                import signal
                import multiprocessing as mp

                # 오버헤드 최소화를 위한 기본값
                # 런처가 포트를 매번 바꾸므로 기본 wait는 0으로 둠
                budget_sec = float(os.environ.get("FAILOVER_CLEANUP_BUDGET_SEC", "8"))
                rpc_timeout_sec = float(os.environ.get("FAILOVER_RPC_SHUTDOWN_TIMEOUT_SEC", "2"))
                join_timeout_sec = float(os.environ.get("FAILOVER_CHILD_JOIN_TIMEOUT_SEC", "1"))
                wait_sec = float(os.environ.get("FAILOVER_SOCKET_CLEANUP_WAIT_SEC", "0"))

                do_rpc = os.environ.get("FAILOVER_CLEANUP_DO_RPC", "0") == "1"
                do_pg = os.environ.get("FAILOVER_CLEANUP_DO_PG", "1") == "1"
                do_children = os.environ.get("FAILOVER_CLEANUP_DO_CHILDREN", "1") == "1"

                def _remaining():
                    return budget_sec - (time.monotonic() - cleanup_start)

                # 1) RPC shutdown은 기본적으로 스킵, 필요할 때만 짧게
                if do_rpc and _remaining() > 0:
                    try:
                        import torch.distributed.rpc as rpc
                        t = min(rpc_timeout_sec, max(0.0, _remaining()))
                        if t > 0:
                            rpc.shutdown(timeout=t)
                    except Exception as ex:
                        logging.warning(f"RPC shutdown (non-fatal): {ex}")

                # 2) Process group destroy는 보통 빠르니 best effort
                if do_pg and _remaining() > 0:
                    try:
                        import torch.distributed as dist
                        if dist.is_available() and dist.is_initialized():
                            dist.destroy_process_group()
                    except Exception as ex:
                        logging.warning(f"Process group destroy (non-fatal): {ex}")

                # 3) 남아있는 자식 프로세스 정리
                if do_children and _remaining() > 0:
                    try:
                        children = mp.active_children()
                        for p in children:
                            try:
                                p.terminate()
                            except Exception:
                                pass

                        # join은 짧게만
                        t_join = min(join_timeout_sec, max(0.0, _remaining()))
                        if t_join > 0:
                            for p in children:
                                try:
                                    p.join(timeout=t_join)
                                except Exception:
                                    pass

                        # 살아있으면 kill
                        for p in mp.active_children():
                            try:
                                if hasattr(p, "kill"):
                                    p.kill()
                                else:
                                    os.kill(p.pid, signal.SIGKILL)
                            except Exception:
                                pass
                    except Exception as ex:
                        logging.warning(f"Child cleanup (non-fatal): {ex}")

                if wait_sec > 0 and _remaining() > 0:
                    time.sleep(min(wait_sec, max(0.0, _remaining())))

                elapsed = time.monotonic() - cleanup_start
                logging.info(f"Cleanup finished in {elapsed:.3f}s. Forcing exit 42 for launcher restart.")

            except Exception as cleanup_error:
                logging.error(f"Cleanup error (continuing anyway): {cleanup_error}")

            os._exit(42)

        raise

    except Exception:
        logging.exception("Unhandled exception in main")
        raise
