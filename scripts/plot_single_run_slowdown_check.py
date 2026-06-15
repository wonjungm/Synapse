#!/usr/bin/env python3
"""Plot a single experiment run with visible slowdown injection evidence.

This script is designed for result directories such as:
  results/e2e_tspipe_batch128_30_100_20260406_033250
  results/e2e_failover_...

It generates one PNG with three panels:
  1. Cumulative batch progress over wall-clock time
  2. Per-partition student compute time per batch
  3. Injected sleep on the slowed partition

Recommended interpreter:
  /venv/tspipe/bin/python scripts/plot_single_run_slowdown_check.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


COMPUTE_TASKS = {"compute_forward", "compute_backward", "compute_optimize"}
COLORS = ["#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0891b2"]


def rolling_mean(values: List[float], window: int) -> np.ndarray:
    if not values:
        return np.array([])
    if window <= 1:
        return np.asarray(values, dtype=float)
    arr = np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def parse_summary(summary_path: Path) -> Dict[str, str]:
    info: Dict[str, str] = {}
    if not summary_path.exists():
        return info
    for line in summary_path.read_text().splitlines():
        if ":" not in line:
            continue
        _, rest = line.split("] ", 1)
        key, value = rest.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def parse_log_metadata(log_path: Path) -> Dict[str, Optional[float]]:
    meta: Dict[str, Optional[float]] = {
        "inject_slowdown_gpu": None,
        "slowdown_fixed_ms": None,
        "slowdown_start": None,
        "slowdown_end": None,
        "batch_size": None,
        "epochs": None,
    }
    if not log_path.exists():
        return meta
    first_line = log_path.read_text().splitlines()[0]
    patterns = {
        "inject_slowdown_gpu": r"inject_slowdown_gpu=([^,\s)]+)",
        "slowdown_fixed_ms": r"slowdown_fixed_ms=([^,\s)]+)",
        "slowdown_start": r"slowdown_start=([^,\s)]+)",
        "slowdown_end": r"slowdown_end=([^,\s)]+)",
        "batch_size": r"batch_size=(\d+)",
        "epochs": r"epochs=(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, first_line)
        if not match:
            continue
        raw = match.group(1)
        if raw in {"None", "null"}:
            meta[key] = None
        else:
            meta[key] = float(raw)
    return meta


def load_partition_data(run_dir: Path) -> Tuple[
    Dict[int, Dict[int, float]],
    Dict[int, Dict[int, float]],
    Dict[int, Dict[int, float]],
]:
    profiling_dir = run_dir / "profiling_logs"
    student_compute_ms: Dict[int, Dict[int, float]] = {}
    injected_sleep_ms: Dict[int, Dict[int, float]] = {}
    completion_time_sec: Dict[int, Dict[int, float]] = {}

    for jsonl_path in sorted(profiling_dir.glob("gpu_task_summary_partition*.jsonl")):
        part = int(jsonl_path.stem.replace("gpu_task_summary_partition", ""))
        per_batch_compute: Dict[int, float] = defaultdict(float)
        per_batch_sleep: Dict[int, float] = defaultdict(float)
        per_batch_completion: Dict[int, float] = {}

        with jsonl_path.open() as f:
            for line in f:
                row = json.loads(line)
                batch_id = int(row["batch_id"])
                task_name = row["task_name"]
                target = bool(row["target"])
                wall_ms = float(row["wall_ms"])
                sleep_ms = float(row.get("injected_sleep_ms", 0.0))
                start_time = float(row.get("start_time", 0.0))

                if not target and task_name in COMPUTE_TASKS:
                    per_batch_compute[batch_id] += wall_ms
                    per_batch_sleep[batch_id] = max(per_batch_sleep[batch_id], sleep_ms)

                if not target and task_name == "compute_optimize":
                    per_batch_completion[batch_id] = start_time + wall_ms / 1000.0

        student_compute_ms[part] = dict(sorted(per_batch_compute.items()))
        injected_sleep_ms[part] = dict(sorted(per_batch_sleep.items()))
        completion_time_sec[part] = dict(sorted(per_batch_completion.items()))

    return student_compute_ms, injected_sleep_ms, completion_time_sec


def detect_slow_window(injected_sleep_ms: Dict[int, Dict[int, float]]) -> Tuple[Optional[int], Optional[int], Optional[int], float]:
    best_part = None
    best_total = -1.0
    for part, batch_map in injected_sleep_ms.items():
        total = sum(batch_map.values())
        if total > best_total:
            best_total = total
            best_part = part

    if best_part is None:
        return None, None, None, 0.0

    active_batches = [batch for batch, value in injected_sleep_ms[best_part].items() if value > 0]
    if not active_batches:
        return best_part, None, None, 0.0
    sleep_values = [injected_sleep_ms[best_part][batch] for batch in active_batches]
    return best_part, min(active_batches), max(active_batches), float(np.mean(sleep_values))


def compute_progress_series(completion_time_sec: Dict[int, Dict[int, float]]) -> Tuple[List[int], List[float], List[float]]:
    if not completion_time_sec:
        return [], [], []
    last_partition = max(completion_time_sec.keys())
    completion = completion_time_sec[last_partition]
    if not completion:
        return [], [], []

    batches = sorted(completion.keys())
    first_end = completion[batches[0]]
    elapsed_min = [(completion[b] - first_end) / 60.0 for b in batches]

    interval_ms = []
    for prev_b, next_b in zip(batches, batches[1:]):
        interval_ms.append((completion[next_b] - completion[prev_b]) * 1000.0)
    interval_ms = [interval_ms[0] if interval_ms else 0.0] + interval_ms
    return batches, elapsed_min, interval_ms


def region_mean(batch_to_value: Dict[int, float], start_batch: int, end_batch: int) -> Optional[float]:
    values = [value for batch, value in batch_to_value.items() if start_batch <= batch <= end_batch]
    if not values:
        return None
    return float(np.mean(values))


def plot_run(run_dir: Path, output_path: Path, rolling_window: int) -> None:
    summary = parse_summary(run_dir / "e2e_summary.log")
    meta = parse_log_metadata(run_dir / "log.txt")
    student_compute_ms, injected_sleep_ms, completion_time_sec = load_partition_data(run_dir)
    slowed_part, slow_start, slow_end, avg_sleep_ms = detect_slow_window(injected_sleep_ms)
    progress_batches, progress_elapsed_min, progress_interval_ms = compute_progress_series(completion_time_sec)

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True)

    title = summary.get("Run note", run_dir.name)
    total = summary.get("Total wall-clock time", "unknown")
    status = summary.get("Status", "UNKNOWN")
    batch_size = int(meta["batch_size"]) if meta["batch_size"] is not None else "?"
    epochs = int(meta["epochs"]) if meta["epochs"] is not None else "?"
    fig.suptitle(
        f"{title}\nstatus={status} | total={total} | batch={batch_size} | epochs={epochs}",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(progress_batches, progress_elapsed_min, color="#1d4ed8", linewidth=2.2)
    if slow_start is not None and slow_end is not None:
        ax.axvspan(slow_start, slow_end, color="#fecaca", alpha=0.5, label="slowdown active")
    ax.set_title("Batch Progress Over Wall-Clock Time")
    ax.set_xlabel("Batch ID")
    ax.set_ylabel("Elapsed Minutes")
    ax.grid(alpha=0.25)

    if progress_batches and slow_start is not None:
        batch_to_interval = dict(zip(progress_batches, progress_interval_ms))
        pre_mean = region_mean(batch_to_interval, max(1, slow_start - 200), max(1, slow_start - 20))
        post_mean = region_mean(batch_to_interval, min(slow_start + 50, progress_batches[-1]), min(slow_start + 250, progress_batches[-1]))
        text_lines = []
        if pre_mean is not None:
            text_lines.append(f"pre interval: {pre_mean:.0f} ms")
        if post_mean is not None:
            text_lines.append(f"post interval: {post_mean:.0f} ms")
        if pre_mean and post_mean:
            text_lines.append(f"slowdown ratio: {post_mean / pre_mean:.2f}x")
        ax.text(
            0.015,
            0.97,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cbd5e1"},
        )

    ax = axes[1]
    for idx, part in enumerate(sorted(student_compute_ms.keys())):
        batch_map = student_compute_ms[part]
        batches = sorted(batch_map.keys())
        values = [batch_map[b] for b in batches]
        smoothed = rolling_mean(values, rolling_window)
        ax.plot(
            batches,
            smoothed,
            linewidth=2.0,
            color=COLORS[idx % len(COLORS)],
            label=f"partition {part}",
        )
    if slow_start is not None and slow_end is not None:
        ax.axvspan(slow_start, slow_end, color="#fecaca", alpha=0.5)
    ax.set_title(f"Student Compute Time Per Batch (rolling window={rolling_window})")
    ax.set_xlabel("Batch ID")
    ax.set_ylabel("Compute Wall Time (ms)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    if slowed_part is not None and slowed_part in student_compute_ms and slow_start is not None:
        batch_map = student_compute_ms[slowed_part]
        pre_mean = region_mean(batch_map, max(1, slow_start - 200), max(1, slow_start - 20))
        post_mean = region_mean(batch_map, min(slow_start + 50, max(batch_map)), min(slow_start + 250, max(batch_map)))
        if pre_mean is not None and post_mean is not None:
            ax.text(
                0.015,
                0.97,
                f"partition {slowed_part}: {pre_mean:.0f} -> {post_mean:.0f} ms ({post_mean / pre_mean:.2f}x)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cbd5e1"},
            )

    ax = axes[2]
    if slowed_part is not None and slowed_part in injected_sleep_ms:
        batch_map = injected_sleep_ms[slowed_part]
        batches = sorted(batch_map.keys())
        values = [batch_map[b] for b in batches]
        ax.step(batches, values, where="mid", color="#b91c1c", linewidth=2.2)
        ax.fill_between(batches, values, step="mid", alpha=0.25, color="#f87171")
        if slow_start is not None and slow_end is not None:
            ax.axvspan(slow_start, slow_end, color="#fecaca", alpha=0.5)
        config_text = (
            f"slowed partition={slowed_part} | detected window={slow_start}-{slow_end} | "
            f"avg injected sleep={avg_sleep_ms:.0f} ms"
        )
        if meta.get("inject_slowdown_gpu") is not None:
            config_text += f" | configured gpu={int(meta['inject_slowdown_gpu'])}"
        if meta.get("slowdown_fixed_ms") is not None:
            config_text += f" | configured fixed={meta['slowdown_fixed_ms']:.0f} ms"
        ax.text(
            0.015,
            0.97,
            config_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cbd5e1"},
        )
    ax.set_title("Injected Sleep on the Slowed Partition")
    ax.set_xlabel("Batch ID")
    ax.set_ylabel("Injected Sleep (ms)")
    ax.grid(alpha=0.25)

    fig.savefig(output_path, dpi=180)
    print(f"saved plot to {output_path}")
    if slow_start is not None and slow_end is not None:
        print(
            f"detected slowdown on partition {slowed_part}: batches {slow_start}-{slow_end}, "
            f"mean injected sleep {avg_sleep_ms:.1f} ms"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Experiment result directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <run_dir>/slowdown_check.png)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=15,
        help="Rolling mean window for per-partition compute plot",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_path = args.output.resolve() if args.output else run_dir / "slowdown_check.png"
    plot_run(run_dir, output_path, max(1, args.rolling_window))


if __name__ == "__main__":
    main()
