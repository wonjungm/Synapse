#!/usr/bin/env python3
"""Create a gpu_task_summary-based comparison figure for persistent GPU-3 background load.

The figure emphasizes the control story clearly:
- when overload first appears in gpu_task_summary
- when wall-clock slowdown is detected and confirmed
- when REPLAN happens
- when post-restart reevaluation resumes
"""

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
from matplotlib.lines import Line2D

COMPUTE_TASKS = {"compute_forward", "compute_backward", "compute_optimize"}

WALL_DETECTED_RE = re.compile(r"Wall-clock slowdown detected .*global_step=(\d+)")
TRIGGER_RE = re.compile(r"Wall-clock trigger confirmed at step (\d+)")
POLICY_RE = re.compile(r"Failover Policy Decision: ([A-Z_]+) \(step=(\d+)")
REEVAL_RE = re.compile(r"reevaluation resumes at step (\d+)")
RESUME_RE = re.compile(r"Resuming training from step \[(\d+)\]")

EVENT_STYLES = {
    "onset": {"color": "#E67E22", "marker": "D", "label": "GPU3 overload onset (inferred)"},
    "wall_detected": {"color": "#7C3AED", "marker": "o", "label": "Wall-clock slowdown detected"},
    "trigger": {"color": "#DC2626", "marker": "s", "label": "Trigger confirmed"},
    "replan": {"color": "#059669", "marker": "*", "label": "REPLAN executed"},
    "reeval": {"color": "#B45309", "marker": "^", "label": "ETA reevaluation resumes"},
}


@dataclass
class RunMetrics:
    label: str
    run_dir: Path
    total_seconds: int
    restart_count: int
    step_interval_steps: np.ndarray
    step_interval_ms: np.ndarray
    step_interval_ma: np.ndarray
    gpu3_steps: np.ndarray
    gpu3_compute_ms: np.ndarray
    gpu3_compute_ma: np.ndarray
    onset_step_interval: Optional[int]
    onset_step_gpu3: Optional[int]
    baseline_interval_ms: float
    baseline_gpu3_ms: float
    events: Dict[str, Optional[int]]
    color: str

    @property
    def onset_step(self) -> Optional[int]:
        candidates = [x for x in [self.onset_step_gpu3, self.onset_step_interval] if x is not None]
        return min(candidates) if candidates else None


def rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if window <= 1 or arr.size == 1:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def parse_summary(summary_path: Path) -> Tuple[int, int]:
    total_seconds = None
    restart_count = 0
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if "Total wall-clock time:" in line:
            match = re.search(r"Total wall-clock time:\s+(\d+)s", line)
            if match:
                total_seconds = int(match.group(1))
        elif "Restart count:" in line:
            match = re.search(r"Restart count:\s+(\d+)", line)
            if match:
                restart_count = int(match.group(1))
    if total_seconds is None:
        raise ValueError(f"Could not parse total wall-clock time from {summary_path}")
    return total_seconds, restart_count


def parse_log_events(log_path: Path) -> Dict[str, Optional[int]]:
    events: Dict[str, Optional[int]] = {
        "wall_detected": None,
        "trigger": None,
        "replan": None,
        "resume": None,
        "reeval": None,
    }
    if not log_path.exists():
        return events

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
        match = POLICY_RE.search(line)
        if match and match.group(1) == "REPLAN":
            events["replan"] = int(match.group(2))
    return events


def infer_onset(
    steps: Sequence[int],
    values: Sequence[float],
    *,
    threshold_ratio: float = 1.10,
    baseline_points: int = 30,
    smooth_window: int = 5,
    min_step: int = 20,
    consecutive: int = 3,
) -> Tuple[Optional[int], float, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    step_arr = np.asarray(steps, dtype=int)
    if arr.size == 0:
        return None, 0.0, arr

    baseline_count = min(max(5, baseline_points), arr.size)
    baseline = float(np.mean(arr[:baseline_count]))
    smoothed = rolling_mean(arr, smooth_window)
    threshold = baseline * float(threshold_ratio)

    for idx, step in enumerate(step_arr):
        if step < min_step:
            continue
        end = min(idx + consecutive, smoothed.size)
        if end - idx < consecutive:
            break
        if np.all(smoothed[idx:end] > threshold):
            return int(step), baseline, smoothed

    return None, baseline, smoothed


def load_run_metrics(run_dir: Path, label: str, color: str) -> RunMetrics:
    profiling_dir = run_dir / "profiling_logs"
    total_seconds, restart_count = parse_summary(run_dir / "e2e_summary.log")
    events = parse_log_events(run_dir / "log.txt")

    gpu3_compute = defaultdict(float)
    step_completion = {}

    for trace_path in sorted(profiling_dir.glob("gpu_task_summary_partition*.jsonl")):
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if bool(row.get("target")):
                    continue

                global_step = row.get("global_step")
                if global_step is None:
                    continue
                global_step = int(global_step)
                task_name = str(row.get("task_name", ""))
                partition_id = int(row.get("partition", -1))
                duration_ms = float(row.get("exec_wall_ms", row.get("time_ms", 0.0)) or 0.0)

                if partition_id == 3 and task_name in COMPUTE_TASKS:
                    gpu3_compute[global_step] += duration_ms

                if task_name == "compute_optimize":
                    start_time = row.get("start_time")
                    if start_time is None:
                        continue
                    end_time = float(start_time) + float(row.get("wall_ms", row.get("time_ms", 0.0)) or 0.0) / 1000.0
                    prev = step_completion.get(global_step)
                    if prev is None or end_time > prev:
                        step_completion[global_step] = end_time

    step_keys = sorted(step_completion)
    interval_steps: List[int] = []
    interval_ms: List[float] = []
    for prev_step, next_step in zip(step_keys, step_keys[1:]):
        interval_steps.append(int(next_step))
        interval_ms.append((step_completion[next_step] - step_completion[prev_step]) * 1000.0)

    gpu3_steps = sorted(gpu3_compute)
    gpu3_values = [gpu3_compute[step] for step in gpu3_steps]

    onset_interval, baseline_interval, step_interval_ma = infer_onset(interval_steps, interval_ms)
    onset_gpu3, baseline_gpu3, gpu3_compute_ma = infer_onset(gpu3_steps, gpu3_values)

    return RunMetrics(
        label=label,
        run_dir=run_dir,
        total_seconds=total_seconds,
        restart_count=restart_count,
        step_interval_steps=np.asarray(interval_steps, dtype=int),
        step_interval_ms=np.asarray(interval_ms, dtype=float),
        step_interval_ma=np.asarray(step_interval_ma, dtype=float),
        gpu3_steps=np.asarray(gpu3_steps, dtype=int),
        gpu3_compute_ms=np.asarray(gpu3_values, dtype=float),
        gpu3_compute_ma=np.asarray(gpu3_compute_ma, dtype=float),
        onset_step_interval=onset_interval,
        onset_step_gpu3=onset_gpu3,
        baseline_interval_ms=baseline_interval,
        baseline_gpu3_ms=baseline_gpu3,
        events=events,
        color=color,
    )


def save_event_csv(path: Path, runs: Iterable[RunMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "run_dir",
                "total_seconds",
                "restart_count",
                "interval_onset_step",
                "gpu3_compute_onset_step",
                "wall_detected_step",
                "trigger_step",
                "replan_step",
                "resume_step",
                "reeval_step",
                "baseline_interval_ms",
                "baseline_gpu3_ms",
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
                    "interval_onset_step": run.onset_step_interval,
                    "gpu3_compute_onset_step": run.onset_step_gpu3,
                    "wall_detected_step": run.events.get("wall_detected"),
                    "trigger_step": run.events.get("trigger"),
                    "replan_step": run.events.get("replan"),
                    "resume_step": run.events.get("resume"),
                    "reeval_step": run.events.get("reeval"),
                    "baseline_interval_ms": f"{run.baseline_interval_ms:.4f}",
                    "baseline_gpu3_ms": f"{run.baseline_gpu3_ms:.4f}",
                }
            )


def compute_focus_window(tspipe_run: RunMetrics, failover_run: RunMetrics) -> Tuple[int, int]:
    points = [
        tspipe_run.onset_step,
        tspipe_run.events.get("wall_detected"),
        failover_run.onset_step,
        failover_run.events.get("wall_detected"),
        failover_run.events.get("trigger"),
        failover_run.events.get("replan"),
        failover_run.events.get("reeval"),
    ]
    points = [int(p) for p in points if p is not None]
    if not points:
        return 0, 100
    x_min = max(0, min(points) - 15)
    x_max = max(points) + 15
    return x_min, x_max


def draw_shared_event_lines(ax, failover_run: RunMetrics, x_min: int, x_max: int) -> None:
    if failover_run.events.get("replan") is not None and failover_run.events.get("reeval") is not None:
        ax.axvspan(failover_run.events["replan"], failover_run.events["reeval"], color="#F3E3B5", alpha=0.35)
    for key in ["onset", "wall_detected", "trigger", "replan", "reeval"]:
        if key == "onset":
            step = failover_run.onset_step
        else:
            step = failover_run.events.get(key)
        if step is None:
            continue
        style = EVENT_STYLES[key]
        ax.axvline(step, color=style["color"], linestyle="--" if key != "replan" else "-", linewidth=1.5, alpha=0.9)
    ax.set_xlim(x_min, x_max)


def annotate_zoom_panel(ax, failover_run: RunMetrics) -> None:
    markers = [
        (failover_run.onset_step, "onset", 0.96),
        (failover_run.events.get("wall_detected"), "detected", 0.88),
        (failover_run.events.get("trigger"), "trigger", 0.80),
        (failover_run.events.get("replan"), "REPLAN", 0.72),
        (failover_run.events.get("reeval"), "reeval", 0.64),
    ]
    ymin, ymax = ax.get_ylim()
    for step, label, frac in markers:
        if step is None:
            continue
        y = ymin + (ymax - ymin) * frac
        ax.text(step + 1.2, y, f"{label}\nstep {step}", fontsize=9, color="#2F2A24", ha="left", va="top")


def plot_timeline_panel(ax, tspipe_run: RunMetrics, failover_run: RunMetrics, x_min: int, x_max: int) -> None:
    rows = [(tspipe_run, 1.0), (failover_run, 0.0)]
    ax.set_title("A. Event Timeline: When Overload Appears and When REPLAN Happens")
    for run, y in rows:
        ax.hlines(y, x_min, x_max, color=run.color, linewidth=3.0, alpha=0.9)
        events = [(run.onset_step, "onset")]
        if run.events.get("wall_detected") is not None:
            events.append((run.events.get("wall_detected"), "wall_detected"))
        if run is failover_run:
            for key in ["trigger", "replan", "reeval"]:
                if run.events.get(key) is not None:
                    events.append((run.events.get(key), key))
        for idx, (step, key) in enumerate(events):
            if step is None:
                continue
            style = EVENT_STYLES[key]
            ax.scatter(step, y, s=110 if key == "replan" else 70, color=style["color"], marker=style["marker"], zorder=5, edgecolor="#2F2A24", linewidth=0.6)
            offset = 0.19 if idx % 2 == 0 else -0.23
            if run is tspipe_run:
                offset = 0.18 if idx == 0 else -0.20
            va = "bottom" if offset > 0 else "top"
            short_label = {
                "onset": f"onset\n{step}",
                "wall_detected": f"detected\n{step}",
                "trigger": f"trigger\n{step}",
                "replan": f"REPLAN\n{step}",
                "reeval": f"reeval\n{step}",
            }[key]
            ax.text(step, y + offset, short_label, fontsize=9, ha="center", va=va, color=style["color"], fontweight="bold")

    if failover_run.events.get("replan") is not None and failover_run.events.get("reeval") is not None:
        ax.fill_betweenx([-0.28, 0.28], failover_run.events["replan"], failover_run.events["reeval"], color="#F3E3B5", alpha=0.55)
        ax.text(
            (failover_run.events["replan"] + failover_run.events["reeval"]) / 2,
            -0.34,
            "restart + observation window",
            ha="center",
            va="top",
            fontsize=9,
            color="#5B4B23",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.55, 1.45)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels([tspipe_run.label, failover_run.label])
    ax.set_xlabel("Global step")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker=EVENT_STYLES[key]["marker"], color="none", markerfacecolor=EVENT_STYLES[key]["color"], markeredgecolor="#2F2A24", markersize=8 if key != "replan" else 11, label=EVENT_STYLES[key]["label"])
        for key in ["onset", "wall_detected", "trigger", "replan", "reeval"]
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncol=2, frameon=False, fontsize=9)


def plot_zoom_panel(ax, runs: Sequence[RunMetrics], *, series: str, x_min: int, x_max: int, title: str, ylabel: str) -> None:
    for run in runs:
        if series == "interval":
            x = run.step_interval_steps
            raw = run.step_interval_ms
            smooth = run.step_interval_ma
        else:
            x = run.gpu3_steps
            raw = run.gpu3_compute_ms
            smooth = run.gpu3_compute_ma
        mask = (x >= x_min) & (x <= x_max)
        if not np.any(mask):
            continue
        ax.plot(x[mask], raw[mask], color=run.color, alpha=0.18, linewidth=1.0)
        ax.plot(x[mask], smooth[mask], color=run.color, linewidth=2.6, label=run.label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_completion_panel(ax, tspipe_run: RunMetrics, failover_run: RunMetrics) -> None:
    runs = [tspipe_run, failover_run]
    labels = [run.label for run in runs]
    minutes = [run.total_seconds / 60.0 for run in runs]
    colors = [run.color for run in runs]
    bars = ax.bar(labels, minutes, color=colors, width=0.58, edgecolor="#2F2A24", linewidth=1.0)
    ax.set_title("D. Total Training Completion Time")
    ax.set_ylabel("Minutes")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for bar, run in zip(bars, runs):
        sec = run.total_seconds
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.18, f"{sec//60}m {sec%60}s", ha="center", va="bottom", fontsize=11, color="#2F2A24", fontweight="bold")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(0.2, bar.get_height() * 0.08),
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
        f"Failover finishes {saved_sec}s faster ({saved_pct:.1f}%) under persistent GPU-3 background load",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#2F2A24",
    )
    ax.set_ylim(0, max(minutes) + 3.0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_comparison(tspipe_run: RunMetrics, failover_run: RunMetrics, output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 160,
        }
    )

    x_min, x_max = compute_focus_window(tspipe_run, failover_run)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.8, 12.8),
        gridspec_kw={"height_ratios": [0.95, 1.25, 1.25, 0.95]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#FCFAF5")
    for ax in axes:
        ax.set_facecolor("#FCFAF5")

    plot_timeline_panel(axes[0], tspipe_run, failover_run, x_min, x_max)

    plot_zoom_panel(
        axes[1],
        [tspipe_run, failover_run],
        series="interval",
        x_min=x_min,
        x_max=x_max,
        title="B. Zoomed End-to-End Step Interval Around the Control Event",
        ylabel="Interval per global step (ms)",
    )
    draw_shared_event_lines(axes[1], failover_run, x_min, x_max)
    annotate_zoom_panel(axes[1], failover_run)
    axes[1].legend(loc="upper right", frameon=False)

    plot_zoom_panel(
        axes[2],
        [tspipe_run, failover_run],
        series="gpu3",
        x_min=x_min,
        x_max=x_max,
        title="C. Zoomed GPU-3 Partition Compute Time Around the Control Event",
        ylabel="Per-step compute time on partition 3 (ms)",
    )
    draw_shared_event_lines(axes[2], failover_run, x_min, x_max)
    annotate_zoom_panel(axes[2], failover_run)
    axes[2].set_xlabel("Global step")

    plot_completion_panel(axes[3], tspipe_run, failover_run)

    fig.suptitle(
        "Persistent GPU-3 Background Load: Clear Timeline of Overload Detection and REPLAN",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.003,
        "Overload onset is inferred from the first sustained >10% increase in a 5-step moving average over the first-30-step gpu_task_summary baseline.",
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
    save_event_csv(csv_path, [tspipe_run, failover_run])

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
        default=Path("/workspace/Synapse/Synapse/results/figures/bgload_gpu3_b128_timeline_compare"),
    )
    args = parser.parse_args()

    tspipe_run = load_run_metrics(args.tspipe_run, "TSPipe baseline", "#4C6A92")
    failover_run = load_run_metrics(args.failover_run, "Failover + REPLAN", "#C46A2D")
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(tspipe_run, failover_run, args.output_base)


if __name__ == "__main__":
    main()
