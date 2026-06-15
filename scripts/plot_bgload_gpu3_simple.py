#!/usr/bin/env python3
"""Create a simple, intuitive figure for the GPU-3 background-load comparison."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

COMPUTE_TASKS = {"compute_forward", "compute_backward", "compute_optimize"}
WALL_DETECTED_RE = re.compile(r"Wall-clock slowdown detected .*global_step=(\d+)")
POLICY_RE = re.compile(r"Failover Policy Decision: ([A-Z_]+) \(step=(\d+)")
REEVAL_RE = re.compile(r"reevaluation resumes at step (\d+)")


@dataclass
class RunSeries:
    label: str
    run_dir: Path
    total_seconds: int
    restart_count: int
    steps: np.ndarray
    values: np.ndarray
    smooth: np.ndarray
    onset_step: Optional[int]
    wall_detected_step: Optional[int]
    replan_step: Optional[int]
    reeval_step: Optional[int]
    color: str


def rolling_mean(values: Sequence[float], window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if arr.size == 1 or window <= 1:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def parse_summary(path: Path) -> Tuple[int, int]:
    total_seconds = None
    restart_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Total wall-clock time:" in line:
            m = re.search(r"Total wall-clock time:\s+(\d+)s", line)
            if m:
                total_seconds = int(m.group(1))
        elif "Restart count:" in line:
            m = re.search(r"Restart count:\s+(\d+)", line)
            if m:
                restart_count = int(m.group(1))
    if total_seconds is None:
        raise ValueError(f"Could not parse total time from {path}")
    return total_seconds, restart_count


def parse_events(log_path: Path) -> Dict[str, Optional[int]]:
    events: Dict[str, Optional[int]] = {
        "wall_detected": None,
        "replan": None,
        "reeval": None,
    }
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if events["wall_detected"] is None:
            m = WALL_DETECTED_RE.search(line)
            if m:
                events["wall_detected"] = int(m.group(1))
        if events["reeval"] is None:
            m = REEVAL_RE.search(line)
            if m:
                events["reeval"] = int(m.group(1))
        m = POLICY_RE.search(line)
        if m and m.group(1) == "REPLAN":
            events["replan"] = int(m.group(2))
    return events


def infer_onset(steps: Sequence[int], values: Sequence[float], threshold_ratio: float = 1.10) -> Optional[int]:
    arr = np.asarray(values, dtype=float)
    step_arr = np.asarray(steps, dtype=int)
    if arr.size < 8:
        return None
    baseline = float(np.mean(arr[: min(30, arr.size)]))
    smooth = rolling_mean(arr, 5)
    threshold = baseline * threshold_ratio
    for idx, step in enumerate(step_arr):
        if step < 20:
            continue
        end = min(idx + 3, smooth.size)
        if end - idx < 3:
            break
        if np.all(smooth[idx:end] > threshold):
            return int(step)
    return None


def load_partition3_compute(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    profiling_dir = run_dir / "profiling_logs"
    values = defaultdict(float)
    for path in sorted(profiling_dir.glob("gpu_task_summary_partition*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if bool(row.get("target")):
                    continue
                if int(row.get("partition", -1)) != 3:
                    continue
                if row.get("task_name") not in COMPUTE_TASKS:
                    continue
                gs = row.get("global_step")
                if gs is None:
                    continue
                values[int(gs)] += float(row.get("exec_wall_ms", row.get("time_ms", 0.0)) or 0.0)
    steps = np.asarray(sorted(values.keys()), dtype=int)
    series = np.asarray([values[int(step)] for step in steps], dtype=float)
    return steps, series


def load_run(run_dir: Path, label: str, color: str) -> RunSeries:
    total_seconds, restart_count = parse_summary(run_dir / "e2e_summary.log")
    events = parse_events(run_dir / "log.txt")
    steps, values = load_partition3_compute(run_dir)
    smooth = rolling_mean(values, 5)
    onset_step = infer_onset(steps, values)
    return RunSeries(
        label=label,
        run_dir=run_dir,
        total_seconds=total_seconds,
        restart_count=restart_count,
        steps=steps,
        values=values,
        smooth=smooth,
        onset_step=onset_step,
        wall_detected_step=events.get("wall_detected"),
        replan_step=events.get("replan"),
        reeval_step=events.get("reeval"),
        color=color,
    )


def save_csv(path: Path, runs: Iterable[RunSeries]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "run_dir",
                "total_seconds",
                "restart_count",
                "onset_step",
                "wall_detected_step",
                "replan_step",
                "reeval_step",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "label": run.label,
                    "run_dir": str(run.run_dir),
                    "total_seconds": run.total_seconds,
                    "restart_count": run.restart_count,
                    "onset_step": run.onset_step,
                    "wall_detected_step": run.wall_detected_step,
                    "replan_step": run.replan_step,
                    "reeval_step": run.reeval_step,
                }
            )


def annotate_line(ax, x: float, y: float, text: str, color: str) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + 2.0, y + 35.0),
        textcoords="data",
        fontsize=10,
        color=color,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
    )


def plot_simple(tspipe: RunSeries, failover: RunSeries, output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 160,
        }
    )

    focus_points = [x for x in [tspipe.onset_step, failover.onset_step, failover.replan_step, failover.reeval_step] if x is not None]
    x_min = max(35, min(focus_points) - 10) if focus_points else 40
    x_max = max(focus_points) + 15 if focus_points else 100

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.8, 8.2),
        gridspec_kw={"height_ratios": [1.7, 1.0]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#FCFAF5")
    for ax in axes:
        ax.set_facecolor("#FCFAF5")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    ax = axes[0]
    for run in [tspipe, failover]:
        mask = (run.steps >= x_min) & (run.steps <= x_max)
        ax.plot(run.steps[mask], run.values[mask], color=run.color, alpha=0.15, linewidth=1.0)
        ax.plot(run.steps[mask], run.smooth[mask], color=run.color, linewidth=3.0, label=run.label)

    if failover.onset_step is not None:
        ax.axvline(failover.onset_step, color="#E67E22", linestyle="--", linewidth=1.6)
        ax.text(failover.onset_step, ax.get_ylim()[1] * 0.98, f"Overload visible\nstep {failover.onset_step}", ha="center", va="top", fontsize=10, color="#E67E22")
    if failover.replan_step is not None:
        ax.axvline(failover.replan_step, color="#059669", linestyle="-", linewidth=2.0)
        ax.text(failover.replan_step + 1.2, ax.get_ylim()[1] * 0.80, f"REPLAN\nstep {failover.replan_step}", ha="left", va="top", fontsize=11, color="#059669", fontweight="bold")
    if failover.replan_step is not None and failover.reeval_step is not None:
        ax.axvspan(failover.replan_step, failover.reeval_step, color="#F3E3B5", alpha=0.45)
        ax.text((failover.replan_step + failover.reeval_step) / 2, ax.get_ylim()[0] + 18, "restart + observation", ha="center", va="bottom", fontsize=10, color="#6B5B22")
    if failover.reeval_step is not None:
        ax.axvline(failover.reeval_step, color="#B45309", linestyle="--", linewidth=1.6)
        ax.text(failover.reeval_step + 0.8, ax.get_ylim()[1] * 0.60, f"reeval\nstep {failover.reeval_step}", ha="left", va="top", fontsize=10, color="#B45309")

    fail_mask = (failover.steps >= x_min) & (failover.steps <= x_max)
    fail_x = failover.steps[fail_mask]
    fail_y = failover.smooth[fail_mask]
    if fail_x.size:
        pre_idx = int(np.argmax(fail_y))
        annotate_line(ax, float(fail_x[pre_idx]), float(fail_y[pre_idx]), "slowdown builds up", failover.color)
        if failover.replan_step is not None:
            post_candidates = np.where(fail_x >= failover.replan_step + 3)[0]
            if post_candidates.size:
                idx = int(post_candidates[np.argmin(fail_y[post_candidates])])
                annotate_line(ax, float(fail_x[idx]), float(fail_y[idx]), "drops after REPLAN", failover.color)

    ax.set_title("GPU-3 Partition Compute Time: REPLAN Reduces the Slow Region")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Per-step compute time on partition 3 (ms)")
    ax.set_xlim(x_min, x_max)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    runs = [tspipe, failover]
    labels = [run.label for run in runs]
    minutes = [run.total_seconds / 60.0 for run in runs]
    bars = ax.bar(labels, minutes, color=[run.color for run in runs], width=0.56, edgecolor="#2F2A24", linewidth=1.0)
    for bar, run in zip(bars, runs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.18,
            f"{run.total_seconds // 60}m {run.total_seconds % 60}s",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#2F2A24",
        )
    saved_sec = tspipe.total_seconds - failover.total_seconds
    saved_pct = saved_sec / tspipe.total_seconds * 100.0 if tspipe.total_seconds else 0.0
    ax.text(0.5, 0.98, f"Failover finishes {saved_sec}s faster ({saved_pct:.1f}%)", transform=ax.transAxes, ha="center", va="top", fontsize=11, color="#2F2A24")
    ax.set_title("Total Training Completion Time")
    ax.set_ylabel("Minutes")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_ylim(0, max(minutes) + 3.0)

    fig.suptitle("Persistent GPU-3 Background Load: Simple Before/After REPLAN View", fontsize=16, fontweight="bold")

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_base.with_suffix(".csv")
    save_csv(csv_path, runs)

    print(png_path)
    print(pdf_path)
    print(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tspipe-run", required=True, type=Path)
    parser.add_argument("--failover-run", required=True, type=Path)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("/workspace/Synapse/Synapse/results/figures/bgload_gpu3_b128_simple_compare"),
    )
    args = parser.parse_args()

    tspipe = load_run(args.tspipe_run, "TSPipe baseline", "#4C6A92")
    failover = load_run(args.failover_run, "Failover + REPLAN", "#C46A2D")
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    plot_simple(tspipe, failover, args.output_base)


if __name__ == "__main__":
    main()
