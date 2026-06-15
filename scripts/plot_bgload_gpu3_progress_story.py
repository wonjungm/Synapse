#!/usr/bin/env python3
"""Create a more intuitive comparison figure for persistent GPU-3 bgload runs.

This figure emphasizes three things on one page:
1. Overall training progress over wall-clock time.
2. Local step-interval slowdown and recovery around the control event.
3. Final end-to-end completion time.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

WALL_DETECTED_RE = re.compile(r"Wall-clock slowdown detected .*global_step=(\d+)")
TRIGGER_RE = re.compile(r"Wall-clock trigger confirmed at step (\d+)")
POLICY_RE = re.compile(r"Failover Policy Decision: ([A-Z_]+) \(step=(\d+)")
RESUME_RE = re.compile(r"Resuming training from step \[(\d+)\]")
REEVAL_RE = re.compile(r"reevaluation resumes at step (\d+)")

EVENT_STYLE = {
    "wall_detected": {"color": "#7C3AED", "marker": "o", "label": "Detected"},
    "trigger": {"color": "#DC2626", "marker": "s", "label": "Trigger"},
    "replan": {"color": "#059669", "marker": "*", "label": "REPLAN"},
    "resume": {"color": "#2563EB", "marker": "D", "label": "Resume"},
    "reeval": {"color": "#B45309", "marker": "^", "label": "Reeval"},
}


@dataclass
class RunMetrics:
    label: str
    run_dir: Path
    total_seconds: int
    restart_count: int
    step_completion_steps: np.ndarray
    elapsed_minutes: np.ndarray
    interval_steps: np.ndarray
    interval_ms: np.ndarray
    interval_ma: np.ndarray
    events: Dict[str, Optional[int]]
    color: str


def rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if window <= 1 or arr.size == 1:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def parse_summary(path: Path) -> Tuple[int, int]:
    total_seconds = None
    restart_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Total wall-clock time:" in line:
            match = re.search(r"Total wall-clock time:\s+(\d+)s", line)
            if match:
                total_seconds = int(match.group(1))
        elif "Restart count:" in line:
            match = re.search(r"Restart count:\s+(\d+)", line)
            if match:
                restart_count = int(match.group(1))
    if total_seconds is None:
        raise ValueError(f"Could not parse total wall-clock time from {path}")
    return total_seconds, restart_count


def parse_events(log_path: Path) -> Dict[str, Optional[int]]:
    events: Dict[str, Optional[int]] = {
        "wall_detected": None,
        "trigger": None,
        "replan": None,
        "resume": None,
        "reeval": None,
    }
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if events["wall_detected"] is None:
            match = WALL_DETECTED_RE.search(line)
            if match:
                events["wall_detected"] = int(match.group(1))
        if events["trigger"] is None:
            match = TRIGGER_RE.search(line)
            if match:
                events["trigger"] = int(match.group(1))
        if events["resume"] is None:
            match = RESUME_RE.search(line)
            if match:
                events["resume"] = int(match.group(1))
        if events["reeval"] is None:
            match = REEVAL_RE.search(line)
            if match:
                events["reeval"] = int(match.group(1))
        if events["replan"] is None:
            match = POLICY_RE.search(line)
            if match and match.group(1) == "REPLAN":
                events["replan"] = int(match.group(2))
    return events


def load_step_timing(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step_completion: Dict[int, float] = {}
    profiling_dir = run_dir / "profiling_logs"

    for trace_path in sorted(profiling_dir.glob("gpu_task_summary_partition*.jsonl")):
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if bool(row.get("target")):
                    continue
                if row.get("task_name") != "compute_optimize":
                    continue
                global_step = row.get("global_step")
                start_time = row.get("start_time")
                if global_step is None or start_time is None:
                    continue
                global_step = int(global_step)
                end_time = float(start_time) + float(row.get("wall_ms", row.get("time_ms", 0.0)) or 0.0) / 1000.0
                prev = step_completion.get(global_step)
                if prev is None or end_time > prev:
                    step_completion[global_step] = end_time

    step_keys = sorted(step_completion)
    if not step_keys:
        raise ValueError(f"No step completion data found under {profiling_dir}")

    first_end = step_completion[step_keys[0]]
    elapsed_minutes = [(step_completion[step] - first_end) / 60.0 for step in step_keys]

    interval_steps: List[int] = []
    interval_ms: List[float] = []
    for prev_step, next_step in zip(step_keys, step_keys[1:]):
        interval_steps.append(next_step)
        interval_ms.append((step_completion[next_step] - step_completion[prev_step]) * 1000.0)

    return (
        np.asarray(step_keys, dtype=int),
        np.asarray(elapsed_minutes, dtype=float),
        np.asarray(interval_steps, dtype=int),
        np.asarray(interval_ms, dtype=float),
    )


def load_run(run_dir: Path, label: str, color: str) -> RunMetrics:
    total_seconds, restart_count = parse_summary(run_dir / "e2e_summary.log")
    events = parse_events(run_dir / "log.txt")
    step_completion_steps, elapsed_minutes, interval_steps, interval_ms = load_step_timing(run_dir)
    interval_ma = rolling_mean(interval_ms, 7)
    return RunMetrics(
        label=label,
        run_dir=run_dir,
        total_seconds=total_seconds,
        restart_count=restart_count,
        step_completion_steps=step_completion_steps,
        elapsed_minutes=elapsed_minutes,
        interval_steps=interval_steps,
        interval_ms=interval_ms,
        interval_ma=interval_ma,
        events=events,
        color=color,
    )


def interpolate_y(xs: np.ndarray, ys: np.ndarray, step: int) -> Optional[float]:
    if xs.size == 0 or ys.size == 0:
        return None
    if step < xs[0] or step > xs[-1]:
        return None
    return float(np.interp(step, xs.astype(float), ys.astype(float)))


def compute_focus_window(runs: Sequence[RunMetrics]) -> Tuple[int, int]:
    points: List[int] = []
    for run in runs:
        for key in ["wall_detected", "trigger", "replan", "resume", "reeval"]:
            value = run.events.get(key)
            if value is not None:
                points.append(int(value))
    if not points:
        return 40, 120
    return max(0, min(points) - 20), max(points) + 25


def draw_shared_event_region(ax, failover_run: RunMetrics, x_min: int, x_max: int) -> None:
    replan = failover_run.events.get("replan")
    reeval = failover_run.events.get("reeval")
    if replan is not None and reeval is not None:
        ax.axvspan(replan, reeval, color="#F3E3B5", alpha=0.35, zorder=0)
    for key in ["wall_detected", "trigger", "replan", "resume", "reeval"]:
        step = failover_run.events.get(key)
        if step is None:
            continue
        style = EVENT_STYLE[key]
        ax.axvline(step, color=style["color"], linestyle="--" if key != "replan" else "-", linewidth=1.4, alpha=0.9)
    ax.set_xlim(x_min, x_max)


def plot_progress_panel(ax, tspipe_run: RunMetrics, failover_run: RunMetrics) -> None:
    runs = [tspipe_run, failover_run]
    for run in runs:
        ax.plot(run.step_completion_steps, run.elapsed_minutes, color=run.color, linewidth=2.8, label=run.label)

    replan = failover_run.events.get("replan")
    reeval = failover_run.events.get("reeval")
    if replan is not None and reeval is not None:
        ax.axvspan(replan, reeval, color="#F3E3B5", alpha=0.35, zorder=0)

    label_offsets = {
        "wall_detected": (7, 0.18),
        "trigger": (10, 0.52),
        "replan": (10, 0.22),
        "reeval": (10, 0.42),
    }
    for key in ["wall_detected", "trigger", "replan", "reeval"]:
        step = failover_run.events.get(key)
        if step is None:
            continue
        y = interpolate_y(failover_run.step_completion_steps, failover_run.elapsed_minutes, step)
        if y is None:
            continue
        style = EVENT_STYLE[key]
        ax.scatter(step, y, color=style["color"], marker=style["marker"], s=110 if key == "replan" else 65, zorder=5, edgecolor="#2F2A24", linewidth=0.6)
        dx, dy = label_offsets[key]
        ax.text(step + dx, y + dy, f"{style['label']}\nstep {step}", fontsize=9, color=style["color"], ha="left", va="bottom", fontweight="bold")

    tspipe_end = tspipe_run.elapsed_minutes[-1]
    failover_end = failover_run.elapsed_minutes[-1]
    ax.text(
        tspipe_run.step_completion_steps[-1] - 14,
        tspipe_end + 0.18,
        f"{tspipe_run.total_seconds // 60}m {tspipe_run.total_seconds % 60}s",
        color=tspipe_run.color,
        fontsize=10,
        ha="right",
        va="bottom",
        fontweight="bold",
    )
    ax.text(
        failover_run.step_completion_steps[-1] - 14,
        failover_end - 0.18,
        f"{failover_run.total_seconds // 60}m {failover_run.total_seconds % 60}s",
        color=failover_run.color,
        fontsize=10,
        ha="right",
        va="top",
        fontweight="bold",
    )

    y_top = max(float(tspipe_end), float(failover_end))
    if replan is not None and reeval is not None:
        ax.text((replan + reeval) / 2.0, y_top * 0.16, "restart + observation", ha="center", va="bottom", fontsize=10, color="#6B5B22")

    ax.set_title("A. Progress Over Wall-Clock Time")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Elapsed minutes")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_interval_panel(ax, tspipe_run: RunMetrics, failover_run: RunMetrics, x_min: int, x_max: int) -> None:
    runs = [tspipe_run, failover_run]
    for run in runs:
        mask = (run.interval_steps >= x_min) & (run.interval_steps <= x_max)
        if not np.any(mask):
            continue
        ax.plot(run.interval_steps[mask], run.interval_ms[mask], color=run.color, alpha=0.14, linewidth=0.9)
        ax.plot(run.interval_steps[mask], run.interval_ma[mask], color=run.color, linewidth=2.8, label=run.label)

    draw_shared_event_region(ax, failover_run, x_min, x_max)
    ymin, ymax = ax.get_ylim()
    label_pos = {
        "wall_detected": 0.96,
        "trigger": 0.87,
        "replan": 0.78,
        "resume": 0.69,
        "reeval": 0.60,
    }
    for key, frac in label_pos.items():
        step = failover_run.events.get(key)
        if step is None:
            continue
        style = EVENT_STYLE[key]
        y = ymin + (ymax - ymin) * frac
        ax.text(step + 1.2, y, f"{style['label']}\n{step}", fontsize=9, color=style["color"], ha="left", va="top")

    ax.set_title("B. Step Interval Near the Control Event")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Per-step interval (ms)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_completion_panel(ax, tspipe_run: RunMetrics, failover_run: RunMetrics) -> None:
    runs = [tspipe_run, failover_run]
    labels = [run.label for run in runs]
    minutes = [run.total_seconds / 60.0 for run in runs]
    bars = ax.bar(labels, minutes, color=[run.color for run in runs], width=0.58, edgecolor="#2F2A24", linewidth=1.0)

    for bar, run in zip(bars, runs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{run.total_seconds // 60}m {run.total_seconds % 60}s",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#2F2A24",
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(0.25, bar.get_height() * 0.08),
            f"restarts={run.restart_count}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2F2A24",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F2E8D5", ec="none", alpha=0.9),
        )

    saved_sec = tspipe_run.total_seconds - failover_run.total_seconds
    saved_pct = (saved_sec / tspipe_run.total_seconds * 100.0) if tspipe_run.total_seconds else 0.0
    ax.text(
        0.5,
        0.98,
        f"Failover finishes {saved_sec}s faster ({saved_pct:.1f}%)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#2F2A24",
    )
    ax.set_title("C. End-to-End Completion Time")
    ax.set_ylabel("Minutes")
    ax.set_ylim(0, max(minutes) + 3.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_csv(path: Path, runs: Iterable[RunMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "run_dir",
                "total_seconds",
                "restart_count",
                "wall_detected_step",
                "trigger_step",
                "replan_step",
                "resume_step",
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
                    "wall_detected_step": run.events.get("wall_detected"),
                    "trigger_step": run.events.get("trigger"),
                    "replan_step": run.events.get("replan"),
                    "resume_step": run.events.get("resume"),
                    "reeval_step": run.events.get("reeval"),
                }
            )


def plot_story(tspipe_run: RunMetrics, failover_run: RunMetrics, output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 180,
        }
    )

    x_min, x_max = compute_focus_window([tspipe_run, failover_run])
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.2, 11.2),
        gridspec_kw={"height_ratios": [1.45, 1.2, 0.9]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#FCFAF5")
    for ax in axes:
        ax.set_facecolor("#FCFAF5")

    plot_progress_panel(axes[0], tspipe_run, failover_run)
    plot_interval_panel(axes[1], tspipe_run, failover_run, x_min, x_max)
    plot_completion_panel(axes[2], tspipe_run, failover_run)

    fig.suptitle(
        "Persistent GPU-3 Background Load: Failover Recovers and Finishes Earlier",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.004,
        "Both runs face the same persistent GPU-3 external load. The failover run detects slowdown, triggers control, executes REPLAN, then resumes ETA reevaluation.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4B4034",
    )

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_base.with_suffix(".csv")
    save_csv(csv_path, [tspipe_run, failover_run])

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
        default=Path("/workspace/Synapse/Synapse/results/figures/bgload_gpu3_b128_progress_story_compare"),
    )
    args = parser.parse_args()

    tspipe_run = load_run(args.tspipe_run, "TSPipe only", "#4C6A92")
    failover_run = load_run(args.failover_run, "Failover + REPLAN", "#C46A2D")
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    plot_story(tspipe_run, failover_run, args.output_base)


if __name__ == "__main__":
    main()
