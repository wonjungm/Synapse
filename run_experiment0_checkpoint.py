#!/usr/bin/env python3
"""
Experiment 0: Checkpoint interval impact benchmark

Measures, for each checkpoint interval N:
1) Average step time (ms/step) excluding warmup
2) Spike around checkpoint steps (pre-save vs post-save)
3) Per-save checkpoint duration C_save (sec)
4) Checkpoint file size (GB)
"""

import argparse
import csv
import json
import os
import select
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


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


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def summarize_run(interval: int, run_dir: Path, warmup_seconds: int) -> Dict[str, Any]:
    step_file = run_dir / "exp0_checkpoint_step_metrics.jsonl"
    save_file = run_dir / "exp0_checkpoint_save_events.jsonl"

    step_rows = [r for r in read_jsonl(step_file) if r.get("event_type") == "step_timing"]
    save_rows = read_jsonl(save_file)
    save_events = [r for r in save_rows if r.get("event_type") == "checkpoint_save"]
    spike_events = [r for r in save_rows if r.get("event_type") == "checkpoint_spike_observed"]

    if step_rows:
        t0 = min(float(r.get("timestamp_sec", 0.0)) for r in step_rows)
        warmup_cut = t0 + warmup_seconds
    else:
        warmup_cut = 0.0

    step_rows_warm = [r for r in step_rows if float(r.get("timestamp_sec", 0.0)) >= warmup_cut]
    save_events_warm = [r for r in save_events if float(r.get("timestamp_sec", 0.0)) >= warmup_cut]
    spike_events_warm = [r for r in spike_events if float(r.get("timestamp_sec", 0.0)) >= warmup_cut]

    step_ms = [float(r.get("step_time_ms", 0.0)) for r in step_rows_warm]
    pre_ms = [float(r.get("pre_step_time_ms")) for r in spike_events_warm if r.get("pre_step_time_ms") is not None]
    post_ms = [float(r.get("post_step_time_ms")) for r in spike_events_warm if r.get("post_step_time_ms") is not None]
    delta_ms = [float(r.get("delta_ms")) for r in spike_events_warm if r.get("delta_ms") is not None]

    save_durations = [float(r.get("save_duration_sec", 0.0)) for r in save_events_warm if r.get("save_duration_sec") is not None]
    size_gb = [float(r.get("file_size_bytes", 0.0)) / (1024 ** 3) for r in save_events_warm if r.get("file_size_bytes") is not None]

    return {
        "checkpoint_interval": interval,
        "run_dir": str(run_dir),
        "warmup_seconds": warmup_seconds,
        "num_steps_total": len(step_rows),
        "num_steps_after_warmup": len(step_rows_warm),
        "avg_step_time_ms": safe_mean(step_ms),
        "num_checkpoint_saves_after_warmup": len(save_events_warm),
        "c_save_sec_avg": safe_mean(save_durations),
        "checkpoint_file_size_gb_avg": safe_mean(size_gb),
        "num_spikes_after_warmup": len(spike_events_warm),
        "spike_pre_step_ms_avg": safe_mean(pre_ms),
        "spike_post_step_ms_avg": safe_mean(post_ms),
        "spike_delta_ms_avg": safe_mean(delta_ms),
    }


def run_one_interval(args, interval: int, run_stamp: str) -> Dict[str, Any]:
    bench_dir = Path(args.benchmark_dir).resolve()
    repo_root = bench_dir.parent.parent
    note = f"exp0_n{interval}_{run_stamp}"

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
        f"--max-runtime-seconds={args.run_minutes * 60}",
        f"--note={note}",
        f"--healthy-checkpoint-interval={interval}",
        "--checkpoint-benchmark-enable",
        "--checkpoint-benchmark-prefix=exp0_checkpoint",
        "--soft-failover-enable",
    ]

    print(f"\n[EXP0] Running N={interval} for {args.run_minutes} min")
    print(f"[EXP0] Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # NCCL stability settings for single-node multi-GPU
    env["NCCL_P2P_DISABLE"] = "0"
    env["NCCL_IB_DISABLE"] = "1"
    env["NCCL_SOCKET_IFNAME"] = "lo"
    env["NCCL_DEBUG"] = "WARN"
    # Ensure PyTorch doesn't interfere with our port management
    env.pop("MASTER_ADDR", None)
    env.pop("MASTER_PORT", None)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{existing_pythonpath}" if existing_pythonpath else str(repo_root)
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    process = subprocess.Popen(
        cmd,
        cwd=str(bench_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None
    timeout_sec = args.interval_timeout_seconds
    if timeout_sec <= 0:
        timeout_sec = args.run_minutes * 60 + 900

    start_time = time.time()
    last_output_time = start_time
    last_heartbeat_time = start_time
    stdout_fd = process.stdout.fileno()

    while True:
        ready, _, _ = select.select([stdout_fd], [], [], 1.0)
        if ready:
            line = process.stdout.readline()
            if line:
                print(f"[N={interval}] {line.rstrip()}")
                last_output_time = time.time()

        if process.poll() is not None:
            break

        now = time.time()
        if (now - last_heartbeat_time) >= args.heartbeat_seconds:
            elapsed = int(now - start_time)
            print(f"[N={interval}] [heartbeat] running... elapsed={elapsed}s")
            last_heartbeat_time = now

        if args.no_output_timeout_seconds > 0 and (now - last_output_time) > args.no_output_timeout_seconds:
            process.kill()
            raise RuntimeError(
                f"N={interval} produced no output for {args.no_output_timeout_seconds}s. "
                f"Killed to avoid indefinite loading."
            )

        if (now - start_time) > timeout_sec:
            process.kill()
            raise RuntimeError(
                f"N={interval} run exceeded timeout ({timeout_sec}s). "
                f"Process was killed to avoid indefinite hang."
            )

    ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"N={interval} run failed with exit code {ret}")

    run_dir = (bench_dir / args.save_root / note).resolve()
    if not run_dir.exists():
        raise RuntimeError(f"Expected run directory not found: {run_dir}")

    summary = summarize_run(interval, run_dir, args.warmup_seconds)
    return summary


def write_outputs(out_dir: Path, summaries: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "experiment0_summary.json"
    csv_path = out_dir / "experiment0_summary.csv"

    with json_path.open("w") as f:
        json.dump(summaries, f, indent=2)

    fieldnames = [
        "checkpoint_interval",
        "avg_step_time_ms",
        "spike_pre_step_ms_avg",
        "spike_post_step_ms_avg",
        "spike_delta_ms_avg",
        "c_save_sec_avg",
        "checkpoint_file_size_gb_avg",
        "num_steps_after_warmup",
        "num_checkpoint_saves_after_warmup",
        "num_spikes_after_warmup",
        "run_dir",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print("\n[EXP0] Summary JSON:", json_path)
    print("[EXP0] Summary CSV :", csv_path)


def main():
    parser = argparse.ArgumentParser(description="Experiment 0: checkpoint interval benchmark")
    parser.add_argument("--intervals", type=str, default="25,50,100,200,400")
    parser.add_argument("--run-minutes", type=int, default=12, help="Per-interval run time in minutes")
    parser.add_argument("--warmup-seconds", type=int, default=180, help="Warmup exclusion window in seconds")
    parser.add_argument("--interval-timeout-seconds", type=int, default=0,
                        help="Safety timeout per interval in seconds; 0 means run_minutes*60+900")
    parser.add_argument("--no-output-timeout-seconds", type=int, default=180,
                        help="Kill interval if no stdout is produced for this many seconds (0 disables)")
    parser.add_argument("--heartbeat-seconds", type=int, default=30,
                        help="Print heartbeat while interval process is running")

    parser.add_argument("--benchmark-dir", type=str, default="./benchmarks/soft_target")
    parser.add_argument("--save-root", type=str, default="./results/exp0_checkpoint")
    parser.add_argument("--output-dir", type=str, default="./exp0_results")

    parser.add_argument("--img-root", type=str, default="/nas-ssd/datasets/imagenet2012/imagenet")
    parser.add_argument("--t-model", type=str, default="./results/base/base-i100-vit-large/model_best.pth.tar")
    parser.add_argument("--s-init", type=str, default="./results/base/base-i100-resnet152/initial_r152.pth.tar")
    parser.add_argument("--t-name", type=str, default="vit_large")
    parser.add_argument("--s-name", type=str, default="resnet152")
    parser.add_argument("--kd-mode", type=str, default="st")
    parser.add_argument("--lambda-kd", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--python-bin", type=str, default="python",
                        help="Python executable used to launch train_kd_profiling.py (set this to your conda env python if needed)")
    parser.add_argument("--tspipe-config", type=str, default="tspipe.yaml",
                        help="TSPipe config filename relative to benchmark-dir")
    parser.add_argument("--cuda-visible-devices", type=str, default="",
                        help="Optional CUDA_VISIBLE_DEVICES override (e.g., '0,1,2')")

    args = parser.parse_args()

    intervals = [int(x.strip()) for x in args.intervals.split(",") if x.strip()]
    if not intervals:
        raise ValueError("No valid intervals provided")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries: List[Dict[str, Any]] = []

    for interval in intervals:
        summary = run_one_interval(args, interval, run_stamp)
        summaries.append(summary)
        print(
            f"[EXP0][N={interval}] avg_step={summary['avg_step_time_ms']:.3f} ms | "
            f"C_save={summary['c_save_sec_avg']:.3f} s | "
            f"size={summary['checkpoint_file_size_gb_avg']:.4f} GB | "
            f"spike_delta={summary['spike_delta_ms_avg']:.3f} ms"
        )

    out_dir = Path(args.output_dir).resolve() / f"exp0_{run_stamp}"
    write_outputs(out_dir, summaries)


if __name__ == "__main__":
    main()
