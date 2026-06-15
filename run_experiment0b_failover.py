#!/usr/bin/env python3
"""
Experiment 0-B(min): Failure Injection Benchmark

For each checkpoint interval N (e.g. 50, 100):
  1. Start TSPipe KD training for a total wall-clock budget (default 20 min).
  2. Inject 2 failures (SIGKILL) at random times within the budget.
  3. After each kill, restart from the latest healthy checkpoint.
  4. Measure:
     - C_load      : time to load checkpoint and resume (seconds)
     - rollback_steps : batch_count at kill - batch_count in checkpoint
     - downtime    : wall-clock seconds between kill and first new training step
     - total_wallclock : end-to-end wall-clock including failures
     - steps_completed : total training steps across all segments
"""

import argparse
import json
import os
import random
import select
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ──────────────── helpers ────────────────

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def find_latest_checkpoint(artifact_dir: Path) -> Optional[Path]:
    """Find the healthy_checkpoint_latest.pth file."""
    ckpt = artifact_dir / "healthy_checkpoint_latest.pth"
    if ckpt.exists():
        return ckpt
    return None


def get_checkpoint_batch_count(ckpt_path: Path) -> int:
    """Read batch_count from a checkpoint file."""
    import torch
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    if isinstance(ckpt, dict) and 'batch_count' in ckpt:
        return int(ckpt['batch_count'])
    return 0


def count_step_rows(step_file: Path) -> int:
    """Count step timing rows in a step_metrics JSONL file."""
    rows = read_jsonl(step_file)
    return sum(1 for r in rows if r.get("event_type") == "step_timing")


# ──────────────── run one segment ────────────────

def build_command(args, interval: int, note: str, runtime_seconds: int,
                  resume_checkpoint: str = "") -> List[str]:
    """Build train_kd_profiling.py command."""
    cmd = [
        args.python_bin,
        "train_kd_profiling.py",
        f"--img_root={args.img_root}",
        f"--save_root={args.save_root}",
        f"--t_model={args.t_model}",
        f"--s_init={args.s_init}",
        f"--kd_mode={args.kd_mode}",
        f"--lambda_kd={args.lambda_kd}",
        f"--t_name={args.t_name}",
        f"--s_name={args.s_name}",
        f"--T={args.temperature}",
        "--data_name=imagenet100",
        "--num_class=100",
        f"--batch_size={args.batch_size}",
        f"--print_freq={args.print_freq}",
        f"--num_workers={args.num_workers}",
        "--tspipe-enable",
        f"--tspipe-config={args.tspipe_config}",
        "--num-nodes=1",
        "--rank=0",
        "--ip=localhost",
        "--epochs=1",
        "--max_step_profiling=1000000000",
        f"--max-runtime-seconds={runtime_seconds}",
        f"--note={note}",
        f"--healthy-checkpoint-interval={interval}",
        "--checkpoint-benchmark-enable",
        "--checkpoint-benchmark-prefix=exp0b_failover",
        "--soft-failover-enable",
    ]
    if resume_checkpoint:
        cmd.append(f"--resume-checkpoint={resume_checkpoint}")
    return cmd


def build_env(args, repo_root: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NCCL_P2P_DISABLE"] = "0"
    env["NCCL_IB_DISABLE"] = "1"
    env["NCCL_SOCKET_IFNAME"] = "lo"
    env["NCCL_DEBUG"] = "WARN"
    env.pop("MASTER_ADDR", None)
    env.pop("MASTER_PORT", None)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{existing}" if existing else str(repo_root)
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def run_segment(args, interval: int, note: str, runtime_seconds: int,
                resume_checkpoint: str = "") -> subprocess.Popen:
    """Start a training segment and return the Popen handle."""
    bench_dir = Path(args.benchmark_dir).resolve()
    repo_root = bench_dir.parent.parent
    cmd = build_command(args, interval, note, runtime_seconds, resume_checkpoint)
    env = build_env(args, repo_root)

    print(f"\n[EXP0B] Starting segment: note={note}, runtime={runtime_seconds}s, "
          f"resume={'yes' if resume_checkpoint else 'no'}")
    print(f"[EXP0B] cmd: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        cwd=str(bench_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env,
    )
    return process


def monitor_and_kill(process: subprocess.Popen, kill_after_sec: float,
                     label: str, max_wait_sec: float = 1800) -> float:
    """
    Monitor Popen output, then kill the process after kill_after_sec seconds.
    Returns the actual wall-clock time from process start to process death.
    """
    assert process.stdout is not None
    start = time.time()
    killed = False
    stdout_fd = process.stdout.fileno()

    while True:
        ready, _, _ = select.select([stdout_fd], [], [], 1.0)
        if ready:
            line = process.stdout.readline()
            if line:
                print(f"  [{label}] {line.rstrip()}")

        if process.poll() is not None:
            break

        elapsed = time.time() - start
        if not killed and elapsed >= kill_after_sec:
            print(f"\n[EXP0B] 💥 KILL injected at {elapsed:.1f}s for {label}")
            os.kill(process.pid, signal.SIGKILL)
            killed = True

        if (time.time() - start) > max_wait_sec:
            print(f"[EXP0B] ⏰ Max wait exceeded for {label}, killing...")
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break

    process.wait()
    return time.time() - start


def monitor_until_done(process: subprocess.Popen, label: str,
                       max_wait_sec: float = 1800) -> float:
    """Monitor a process until natural termination. Returns wall-clock duration."""
    assert process.stdout is not None
    start = time.time()
    stdout_fd = process.stdout.fileno()

    while True:
        ready, _, _ = select.select([stdout_fd], [], [], 1.0)
        if ready:
            line = process.stdout.readline()
            if line:
                print(f"  [{label}] {line.rstrip()}")

        if process.poll() is not None:
            break

        if (time.time() - start) > max_wait_sec:
            print(f"[EXP0B] ⏰ Max wait exceeded for {label}, killing...")
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break

    process.wait()
    return time.time() - start


# ──────────────── main experiment for one N ────────────────

def run_one_interval(args, interval: int, run_stamp: str) -> Dict[str, Any]:
    """
    Run the full failover experiment for one checkpoint interval N.
    Budget: args.total_minutes minutes, with args.num_kills kills injected.
    """
    total_budget_sec = args.total_minutes * 60
    num_kills = args.num_kills
    bench_dir = Path(args.benchmark_dir).resolve()

    # Decide kill times (wall-clock offsets from experiment start)
    # Place kills in the middle portion of the budget (20%-80% of total time)
    # to ensure enough training time before first kill and after last restart
    low = int(total_budget_sec * 0.15)
    high = int(total_budget_sec * 0.65)
    # Ensure at least 120s gap between kill times so the process can restart and run
    while True:
        kill_times = sorted(random.sample(range(low, high), num_kills))
        if all(kill_times[i+1] - kill_times[i] >= 120 for i in range(len(kill_times)-1)):
            break
    print(f"\n[EXP0B] ═══════════════════════════════════════════════")
    print(f"[EXP0B] N={interval}, budget={total_budget_sec}s, kills at offsets={kill_times}")
    print(f"[EXP0B] ═══════════════════════════════════════════════")

    experiment_start = time.time()
    segment_results: List[Dict[str, Any]] = []
    total_steps = 0
    accumulated_elapsed = 0.0  # wall-clock consumed so far

    for kill_idx in range(num_kills + 1):
        # Is this a kill segment or the final segment?
        is_kill_segment = kill_idx < num_kills

        # Determine how long this segment should run
        if is_kill_segment:
            # This segment runs until the next kill time
            kill_wall_offset = kill_times[kill_idx]
            segment_runtime = kill_wall_offset - accumulated_elapsed
            if segment_runtime < 30:
                # Too little time; clamp to minimum 30s
                segment_runtime = 30
        else:
            # Final segment: remaining budget
            segment_runtime = total_budget_sec - accumulated_elapsed
            if segment_runtime < 30:
                print(f"[EXP0B] Only {segment_runtime:.0f}s remaining, skipping final segment")
                break

        # Build note and determine resume checkpoint
        note = f"exp0b_n{interval}_{run_stamp}_seg{kill_idx}"
        artifact_dir = (bench_dir / args.save_root / note).resolve()
        resume_checkpoint = ""

        if kill_idx > 0:
            # Find checkpoint from any previous segment (search backwards)
            for prev_idx in range(kill_idx - 1, -1, -1):
                prev_note = f"exp0b_n{interval}_{run_stamp}_seg{prev_idx}"
                prev_artifact_dir = (bench_dir / args.save_root / prev_note).resolve()
                ckpt = find_latest_checkpoint(prev_artifact_dir)
                if ckpt:
                    resume_checkpoint = str(ckpt)
                    ckpt_batch = get_checkpoint_batch_count(ckpt)
                    print(f"[EXP0B] 📂 Resuming from {ckpt} (batch_count={ckpt_batch}, from seg{prev_idx})")
                    break
            else:
                print(f"[EXP0B] ⚠️  No checkpoint found in any previous segment, starting fresh")

        segment_start = time.time()

        # Give enough runtime so training actually runs past the kill point
        # (the training process gets max-runtime-seconds as its internal limit)
        internal_runtime = int(segment_runtime + 120) if is_kill_segment else int(segment_runtime)

        proc = run_segment(args, interval, note, internal_runtime, resume_checkpoint)

        if is_kill_segment:
            actual_wall = monitor_and_kill(proc, segment_runtime,
                                           f"N={interval}/seg{kill_idx}")
        else:
            actual_wall = monitor_until_done(proc, f"N={interval}/seg{kill_idx}",
                                              max_wait_sec=int(segment_runtime) + 300)

        segment_end = time.time()
        segment_duration = segment_end - segment_start

        # ---- Gather segment metrics ----
        step_file = artifact_dir / "exp0b_failover_step_metrics.jsonl"
        save_file = artifact_dir / "exp0b_failover_save_events.jsonl"

        seg_steps = count_step_rows(step_file)
        total_steps += seg_steps

        # Read resume_info.json for C_load
        c_load_sec = 0.0
        resumed_batch_count = 0
        resume_info_path = artifact_dir / "resume_info.json"
        if resume_info_path.exists():
            with resume_info_path.open() as f:
                ri = json.load(f)
            c_load_sec = ri.get("c_load_sec", 0.0)
            resumed_batch_count = ri.get("resumed_batch_count", 0)

        # Read last save event to get last checkpoint batch_count
        save_events = [r for r in read_jsonl(save_file)
                       if r.get("event_type") == "checkpoint_save"]
        last_ckpt_batch = save_events[-1]["batch_count"] if save_events else 0

        # Read step metrics to get last batch_id at kill time
        step_rows = [r for r in read_jsonl(step_file)
                     if r.get("event_type") == "step_timing"]
        last_step_batch = step_rows[-1].get("batch_count", 0) if step_rows else 0

        # rollback_steps = steps done after last checkpoint save
        rollback_steps = last_step_batch - last_ckpt_batch if is_kill_segment else 0

        seg_info = {
            "segment": kill_idx,
            "is_kill_segment": is_kill_segment,
            "note": note,
            "segment_runtime_planned_sec": segment_runtime,
            "segment_duration_actual_sec": segment_duration,
            "steps_in_segment": seg_steps,
            "resume_checkpoint": resume_checkpoint,
            "c_load_sec": c_load_sec,
            "resumed_batch_count": resumed_batch_count,
            "last_ckpt_batch_count": last_ckpt_batch,
            "last_step_batch": last_step_batch,
            "rollback_steps": rollback_steps,
            "num_saves": len(save_events),
        }
        segment_results.append(seg_info)
        print(f"\n[EXP0B] Segment {kill_idx} done: {seg_steps} steps, "
              f"rollback={rollback_steps}, C_load={c_load_sec:.3f}s, "
              f"wall={segment_duration:.1f}s")

        accumulated_elapsed += segment_duration
        # Small pause between segments for cleanup
        time.sleep(3)

    experiment_end = time.time()
    total_wallclock = experiment_end - experiment_start

    # Compute aggregate metrics
    kill_segments = [s for s in segment_results if s["is_kill_segment"]]
    resume_segments = [s for s in segment_results if s["c_load_sec"] > 0]

    avg_c_load = (sum(s["c_load_sec"] for s in resume_segments) / len(resume_segments)
                  if resume_segments else 0.0)
    avg_rollback = (sum(s["rollback_steps"] for s in kill_segments) / len(kill_segments)
                    if kill_segments else 0.0)
    total_downtime = sum(
        # Downtime ≈ time between kill and first training step of next segment
        # Approximate as: gap between end of kill segment and start of next segment + C_load
        3.0 + segment_results[i + 1]["c_load_sec"]  # 3s pause + C_load
        for i in range(len(kill_segments))
        if i + 1 < len(segment_results)
    ) if kill_segments else 0.0

    summary = {
        "checkpoint_interval": interval,
        "total_budget_sec": total_budget_sec,
        "num_kills": num_kills,
        "kill_time_offsets": kill_times,
        "total_wallclock_sec": total_wallclock,
        "total_steps_completed": total_steps,
        "avg_c_load_sec": avg_c_load,
        "avg_rollback_steps": avg_rollback,
        "total_downtime_sec_approx": total_downtime,
        "segments": segment_results,
        "run_stamp": run_stamp,
    }
    return summary


# ──────────────── main ────────────────

def main():
    parser = argparse.ArgumentParser(description="Experiment 0-B: Failure Injection Benchmark")
    parser.add_argument("--intervals", type=str, default="50,100",
                        help="Comma-separated checkpoint intervals to test")
    parser.add_argument("--total-minutes", type=int, default=20,
                        help="Total wall-clock budget per interval (minutes)")
    parser.add_argument("--num-kills", type=int, default=2,
                        help="Number of failure injections per interval")
    parser.add_argument("--output-dir", type=str, default="./exp0b_results",
                        help="Directory to write final results")

    parser.add_argument("--benchmark-dir", type=str, default="./benchmarks/soft_target")
    parser.add_argument("--save-root", type=str, default="./results/exp0b_failover")
    parser.add_argument("--img-root", type=str, default="/nas-ssd/datasets/imagenet2012/imagenet")
    parser.add_argument("--t-model", type=str,
                        default="./results/base/base-i100-vit-large/model_best.pth.tar")
    parser.add_argument("--s-init", type=str,
                        default="./results/base/base-i100-resnet152/initial_r152.pth.tar")
    parser.add_argument("--t-name", type=str, default="vit_large")
    parser.add_argument("--s-name", type=str, default="resnet152")
    parser.add_argument("--kd-mode", type=str, default="st")
    parser.add_argument("--lambda-kd", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument("--tspipe-config", type=str, default="tspipe.yaml")
    parser.add_argument("--cuda-visible-devices", type=str, default="")

    args = parser.parse_args()
    intervals = [int(x.strip()) for x in args.intervals.split(",") if x.strip()]
    if not intervals:
        raise ValueError("No valid intervals provided")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_summaries: List[Dict[str, Any]] = []

    for interval in intervals:
        summary = run_one_interval(args, interval, run_stamp)
        all_summaries.append(summary)
        print(f"\n[EXP0B] ══════ N={interval} RESULT ══════")
        print(f"  total_wallclock = {summary['total_wallclock_sec']:.1f}s")
        print(f"  total_steps     = {summary['total_steps_completed']}")
        print(f"  avg_C_load      = {summary['avg_c_load_sec']:.3f}s")
        print(f"  avg_rollback    = {summary['avg_rollback_steps']:.1f} steps")
        print(f"  downtime_approx = {summary['total_downtime_sec_approx']:.1f}s")

    # Write outputs
    out_dir = Path(args.output_dir).resolve() / f"exp0b_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "experiment0b_summary.json"
    with json_path.open("w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[EXP0B] Summary JSON: {json_path}")

    # CSV summary
    import csv
    csv_path = out_dir / "experiment0b_summary.csv"
    fields = [
        "checkpoint_interval", "total_wallclock_sec", "total_steps_completed",
        "avg_c_load_sec", "avg_rollback_steps", "total_downtime_sec_approx",
        "num_kills",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in all_summaries:
            writer.writerow({k: s.get(k) for k in fields})
    print(f"[EXP0B] Summary CSV : {csv_path}")


if __name__ == "__main__":
    main()
