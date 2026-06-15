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
from typing import Iterable, Optional

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
parser.add_argument('--slowdown-factor', type=float, default=None, help='Slowdown factor (e.g., 1.5 for 1.5x slower)')
parser.add_argument('--slowdown-duration', type=int, default=None, help='Number of steps to inject slowdown (default: scenario-specific)')
parser.add_argument('--slowdown-start', type=int, default=None, help='Start step for slowdown injection (default: scenario-specific)')
parser.add_argument('--slowdown-end', type=int, default=None, help='End step for slowdown injection (default: scenario-specific)')

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
    """Load restart_config/checkpoint and return resume info for failover restart.
    
    Args:
        skip_partition_yaml_apply: True면 YAML 파일 수정 건너뜀 (이미 첫 번째 호출에서 한 경우)
        skip_checkpoint_restore: True면 checkpoint 로드 건너뜀 (이미 첫 번째 호출에서 한 경우)
    """
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

    # Guard: restart partition must match current model layer counts.
    # Otherwise (e.g., stale config from another model family), skip restart payload.
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
                }

    # ✅ Apply new partition to YAML if restart payload has one (but skip if already applied in first call)
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
        return {
            "enabled": True,
            "resume_step": resume_step,
            "resume_epoch": int(restart_payload.get("epoch_id", 1)),
            "resume_batch_offset": int(restart_payload.get("batch_offset", 0)),
            "partition": partition_cfg,
            "alpha_comp": None,  # Skip alpha/beta on second call
            "beta_comm": None,
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
    niter = int(bootstrap["resume_step"])
    start_epoch = int(bootstrap["resume_epoch"])
    _resume_target_epoch = start_epoch
    _resume_batches_to_skip = int(bootstrap["resume_batch_offset"])

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
        niter = int(bootstrap["resume_step"])
        start_epoch = int(bootstrap["resume_epoch"])
        _resume_target_epoch = start_epoch
        _resume_batches_to_skip = int(bootstrap["resume_batch_offset"])

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
        
        # NEW: Initialize slowdown detection and failover policy components
        slowdown_detector = SlowdownDetector(inject_scenario=args.failover_inject_scenario)
        
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

            # ✅ Restore alpha/beta coefficients captured from the first
            # bootstrap call (which read them from the checkpoint or
            # emergency_restart_config). The second bootstrap call deliberately
            # returns None for these fields when skip_checkpoint_restore=True.
            if bootstrap_alpha_comp is not None:
                failover_optimizer.alpha_g = bootstrap_alpha_comp
                logging.error(f"🔄 Restored alpha_g from checkpoint: {failover_optimizer.alpha_g}")

            if bootstrap_beta_comm is not None:
                failover_optimizer.beta_g = bootstrap_beta_comm
                logging.error(f"🔄 Restored beta_g from checkpoint: {failover_optimizer.beta_g}")

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

    if bootstrap["enabled"]:
        logging.error(
            f"Failover recovery successful. Resuming training from step [{niter}] with new partition."
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
                        task_time_ms = float(record.get("time_ms", 0.0))
                        if task_time_ms <= 0:
                            continue

                        task_name = str(record.get("task_name", ""))
                        if task_name.startswith("compute"):
                            self._append_hist(self._compute_hist, gpu_id, task_time_ms)
                        elif task_name.startswith("copy"):
                            self._append_hist(self._comm_hist, gpu_id, task_time_ms)
                    self._offsets[key] = f.tell()
            except OSError:
                continue

    def build_timing_payload(self):
        payload = {}
        all_gpu_ids = set(self._compute_hist.keys()) | set(self._comm_hist.keys())
        for gpu_id in all_gpu_ids:
            c_hist = self._compute_hist.get(gpu_id, [])
            m_hist = self._comm_hist.get(gpu_id, [])
            if not c_hist and not m_hist:
                continue
            compute_ms = (sum(c_hist) / len(c_hist)) if c_hist else 0.0
            comm_ms = (sum(m_hist) / len(m_hist)) if m_hist else 0.0
            payload[gpu_id] = {
                "compute_time": compute_ms/1000.0,
                "comm_time": comm_ms/1000.0,
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
    # tqdm 진행바와 enumerate 시작 인덱스를 niter(복원 step)에 맞게 조정
    pbar = tqdm(train_loader, initial=niter)
    for i, (img, target) in enumerate(pbar, start=niter+1):
        if i <= niter:
            continue
        if _should_skip_resume_batch(epoch):
            continue

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
                 max_steps_per_epoch: int = 0):
    """TSPipe training with failover monitoring and policy decisions
    
    NEW: Integrated slowdown detection and dynamic policy selection
    """
    global niter

    global niter
    # tqdm 진행바와 enumerate 시작 인덱스를 niter(복원 step)에 맞게 조정
    pbar = tqdm(train_loader, initial=niter)
    for i, (img, target) in enumerate(pbar, start=niter+1):
        if i <= niter:
            continue
        if _should_skip_resume_batch(epoch):
            continue

        # Record batch start time
        batch_start_time = time.time()
        
        loss = tspipe_trainer.feed(img, img, target)

        if loss is None:
            continue

        # Record batch wall-clock for SlowdownDetector fallback
        batch_elapsed_time_ms = (time.time() - batch_start_time) * 1000

        # Generalized slowdown injection for failover experiments
        slowdown_gpu = args.inject_slowdown_gpu
        slowdown_factor = args.slowdown_factor
        slowdown_duration = args.slowdown_duration
        slowdown_start = args.slowdown_start
        slowdown_end = args.slowdown_end

        # Scenario-specific defaults
        inject_scenario = getattr(args, "failover_inject_scenario", "") or os.environ.get(
            "FAILOVER_INJECT_SCENARIO", ""
        ).strip()
        if inject_scenario == "KEEP_REPLAN_DEGRADE":
            if slowdown_factor is None:
                slowdown_factor = 1.05
            if slowdown_start is None:
                slowdown_start = 30
            if slowdown_end is None:
                slowdown_end = 80
        elif inject_scenario == "REPLAN_SLOWDOWN":
            if slowdown_factor is None:
                slowdown_factor = 1.6
            if slowdown_start is None:
                slowdown_start = 60
            total_steps = len(train_loader) * args.epochs
            if slowdown_duration is not None:
                slowdown_end = min(total_steps, slowdown_start + slowdown_duration)
            elif slowdown_end is None:
                slowdown_end = total_steps

        # Only inject slowdown if scenario and GPU match
        # (If slowdown_gpu is None, always inject; else only for matching rank)
        should_inject = False
        current_rank = getattr(tspipe_trainer, "rank", 0)
        if slowdown_gpu is None or current_rank == slowdown_gpu:
            if slowdown_factor is not None and slowdown_start is not None and slowdown_end is not None:
                if slowdown_start <= niter < slowdown_end:
                    should_inject = True

        if should_inject:
            baseline_ms = None
            if slowdown_detector is not None and slowdown_detector.baseline_stage_time is not None:
                baseline_ms = slowdown_detector.baseline_stage_time
            else:
                baseline_ms = batch_elapsed_time_ms

            extra_ms = max(baseline_ms * (slowdown_factor - 1.0), 0.0)
            extra_sec = extra_ms / 1000.0
            if extra_sec > 0:
                time.sleep(extra_sec)
                # sleep까지 포함한 실제 wall-clock을 다시 측정해 기록
                batch_elapsed_time_ms = (time.time() - batch_start_time) * 1000
                logging.info(
                    f"🧪 REAL slowdown injected (scenario={inject_scenario}, "
                    f"rank={current_rank}, gpu={slowdown_gpu}, "
                    f"range={slowdown_start}-{slowdown_end}, target≈{slowdown_factor:.2f}x, step={niter}, "
                    f"elapsed={batch_elapsed_time_ms:.2f}ms)"
                )

        if failover_optimizer is not None:
            # Keep progress tracker up to date every step for K_rem and phase transitions.
            failover_optimizer.update_training_progress(niter, epoch)

        timing_payload = {}
        if failover_optimizer is not None and timing_ingestor is not None:
            timing_ingestor.update()
            timing_payload = timing_ingestor.build_timing_payload()
            if timing_payload:
                failover_optimizer.ingest_runtime_timing_batch(timing_payload, ema=0.4)

        # Always feed wall-clock into SlowdownDetector for baseline tracking
        if slowdown_detector is not None:
            slowdown_detector.record_stage_time(batch_elapsed_time_ms)

        # Periodic failover evaluation (every 10 steps)
        if failover_optimizer is not None and niter % 10 == 0:
            # Prefer timing-based alpha_g (current/baseline ratio per GPU) over wall-clock.
            # alpha_g is updated by ingest_runtime_timing_batch above via EMA.
            # ✅ FIX: Consider both alpha (compute) and beta (communication) slowdown
            # ✅ FIX: Gate on phase-0 to allow wall-clock fallback during baseline collection
            use_timing_based = (
                timing_payload
                and failover_optimizer.alpha_g
                and not failover_optimizer._is_phase0_baseline_active()
            )
            
            if use_timing_based:
                slow_gpu_id = failover_optimizer.identify_slow_gpu()
                # ✅ Extract slowdown ratio for the identified slow GPU, not system-wide max
                # Ensures consistent (GPU id, slowdown ratio) pairing in logs
                slowdown_ratio = max(
                    failover_optimizer.alpha_g.get(slow_gpu_id, 1.0),
                    failover_optimizer.beta_g.get(slow_gpu_id, 1.0),
                )
            elif slowdown_detector is not None:
                slowdown_ratio = slowdown_detector.get_slowdown_ratio()
                slow_gpu_id = 0
            else:
                slowdown_ratio = 1.0
                slow_gpu_id = 0

            _SLOWDOWN_TRIGGER = 1.10  # 스펙 4절 1단계 임계값
            if slowdown_ratio > _SLOWDOWN_TRIGGER:
                logging.info(
                    f"⚠️ Slowdown detected: {slowdown_ratio:.2f}x "
                    f"(GPU {slow_gpu_id}) at step {niter}"
                )
                policy = failover_optimizer.evaluate_slowdown_and_decide(
                    gpu_id=slow_gpu_id,
                    current_slowdown=slowdown_ratio,
                )
                logging.info(
                    f"🎯 Failover Policy Decision: {policy} "
                    f"(step={niter}, slowdown={slowdown_ratio:.2f}x, gpu={slow_gpu_id})"
                )
                failover_optimizer.execute_policy(
                    policy, gpu_id=slow_gpu_id, current_slowdown=slowdown_ratio
                )

        writer.add_scalar('loss', loss, global_step=niter)
        pbar.set_postfix({'loss': loss, 'batch_id': niter})

        niter += 1

        # 한 epoch당 step 수를 제한해서 e2e 실험을 빠르게 종료
        if max_steps_per_epoch > 0 and i >= max_steps_per_epoch:
            break


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
                # 종료 지연의 주범이라 여기만은 기본적으로 수행
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

                # 4) 소켓 대기는 기본 0, 필요 시에만 짧게
                if wait_sec > 0 and _remaining() > 0:
                    time.sleep(min(wait_sec, max(0.0, _remaining())))

                elapsed = time.monotonic() - cleanup_start
                logging.info(f"Cleanup finished in {elapsed:.3f}s. Forcing exit 42 for launcher restart.")

            except Exception as cleanup_error:
                logging.error(f"Cleanup error (continuing anyway): {cleanup_error}")

            # 파이썬 종료 지연 방지용 강제 종료
            os._exit(42)

        raise

    except Exception:
        logging.exception("Unhandled exception in main")
        raise