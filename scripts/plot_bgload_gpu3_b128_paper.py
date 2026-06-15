#!/usr/bin/env python3
"""Paper-style comparison figure for persistent GPU-3 bgload at batch size 128."""

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
from matplotlib.lines import Line2D

WALL_DETECTED_RE = re.compile(r"Wall-clock slowdown detected .*global_step=(\d+)")
TRIGGER_RE = re.compile(r"Wall-clock trigger confirmed at step (\d+)")
POLICY_RE = re.compile(r"Failover Policy Decision: ([A-Z_]+) \(step=(\d+)")
RESUME_RE = re.compile(r"Resuming training from step \[(\d+)\]")
REEVAL_RE = re.compile(r"reevaluation resumes at step (\d+)")

EVENTS = {
    "wall_detected": {"color": "#7C3AED", "linestyle": "--", "label": "Detected"},
    "trigger": {"color": "#DC2626", "linestyle": "--", "label": "Trigger"},
    "replan": {"color": "#059669", "linestyle": "-", "label": "REPLAN"},
    "resume": {"color": "#2563EB", "linestyle": "--", "label": "Resume"},
    "reeval": {"color": "#B45309", "linestyle": "--", "label": "Reeval"},
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


def mean_in_window(run: RunMetrics, lo: int, hi: int) -> float:
    vals = run.interval_ms[(run.interval_steps >= lo) & (run.interval_steps <= hi)]
    if vals.size == 0:
        raise ValueError(f"No interval samples for {run.label} in [{lo}, {hi}]")
    return float(np.mean(vals))


def interpolate_y(xs: np.ndarray, ys: np.ndarray, x: int) -> Optional[float]:
    if xs.size == 0 or ys.size == 0 or x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs.astype(float), ys.astype(float)))


def save_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_progress(ax, tspipe_run: RunMetrics, failover_run: RunMetrics) -> None:
    for run in [tspipe_run, failover_run]:
        ax.plot(run.step_completion_steps, run.elapsed_minutes, color=run.color, linewidth=2.8, label=run.label)

    replan = failover_run.events.get("replan")
    reeval = failover_run.events.get("reeval")
    if replan is not None and reeval is not None:
        ax.axvspan(replan, reeval, color="#F3E3B5", alpha=0.38, zorder=0)
        ax.text((replan + reeval) / 2.0, 3.2, "restart +\nobservation", ha="center", va="bottom", fontsize=10, color="#6B5B22")

    if replan is not None:
        y = interpolate_y(failover_run.step_completion_steps, failover_run.elapsed_minutes, replan)
        if y is not None:
            ax.scatter(replan, y, s=120, color=EVENTS["replan"]["color"], marker="*", edgecolor="#2F2A24", linewidth=0.7, zorder=5)
            ax.annotate(
                "REPLAN at step 80",
                xy=(replan, y),
                xytext=(135, y + 4.5),
                fontsize=11,
                color=EVENTS["replan"]["color"],
                fontweight="bold",
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", lw=1.5, color=EVENTS["replan"]["color"]),
            )

    for run, dy, va in [(tspipe_run, 0.25, "bottom"), (failover_run, -0.25, "top")]:
        ax.text(
            run.step_completion_steps[-1] - 18,
            run.elapsed_minutes[-1] + dy,
            f"{run.total_seconds // 60}m {run.total_seconds % 60}s",
            color=run.color,
            fontsize=11,
            fontweight="bold",
            ha="right",
            va=va,
        )

    ax.set_title("A. Global Progress: Failover Pulls Ahead After REPLAN")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Elapsed time (min)")
    ax.set_xlim(0, max(tspipe_run.step_completion_steps[-1], failover_run.step_completion_steps[-1]) + 15)
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(loc="upper left", frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_interval_recovery(ax, tspipe_run: RunMetrics, failover_run: RunMetrics) -> Dict[str, float]:
    x_min, x_max = 48, 132
    pre_lo, pre_hi = 60, 79
    post_lo, post_hi = 90, 129

    for run in [tspipe_run, failover_run]:
        mask = (run.interval_steps >= x_min) & (run.interval_steps <= x_max)
        ax.plot(run.interval_steps[mask], run.interval_ms[mask], color=run.color, alpha=0.12, linewidth=1.0)
        ax.plot(run.interval_steps[mask], run.interval_ma[mask], color=run.color, linewidth=3.0, label=run.label)

    replan = failover_run.events.get("replan")
    reeval = failover_run.events.get("reeval")
    if replan is not None and reeval is not None:
        ax.axvspan(replan, reeval, color="#F3E3B5", alpha=0.38, zorder=0)

    for key in ["wall_detected", "trigger", "replan", "reeval"]:
        step = failover_run.events.get(key)
        if step is None:
            continue
        spec = EVENTS[key]
        ax.axvline(step, color=spec["color"], linestyle=spec["linestyle"], linewidth=1.5)

    fail_pre = mean_in_window(failover_run, pre_lo, pre_hi)
    fail_post = mean_in_window(failover_run, post_lo, post_hi)
    base_post = mean_in_window(tspipe_run, post_lo, post_hi)
    drop_pct = (fail_pre - fail_post) / fail_pre * 100.0
    gain_vs_base_pct = (base_post - fail_post) / base_post * 100.0

    ax.hlines(fail_pre, pre_lo, pre_hi, color=failover_run.color, linewidth=2.4, linestyle=(0, (4, 2)))
    ax.hlines(fail_post, post_lo, post_hi, color=failover_run.color, linewidth=2.8)
    ax.hlines(base_post, post_lo, post_hi, color=tspipe_run.color, linewidth=2.2, linestyle=(0, (1.4, 2.4)))

    ax.annotate(
        f"Failover before REPLAN\n{fail_pre:.0f} ms",
        xy=(pre_hi - 1, fail_pre),
        xytext=(50, fail_pre + 380),
        fontsize=10,
        color=failover_run.color,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", lw=1.3, color=failover_run.color),
    )
    ax.annotate(
        f"Failover after REPLAN\n{fail_post:.0f} ms\n({drop_pct:.1f}% lower)",
        xy=(post_hi - 4, fail_post),
        xytext=(106, fail_post - 470),
        fontsize=10,
        color=failover_run.color,
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", lw=1.3, color=failover_run.color),
    )
    ax.annotate(
        f"TSPipe under same load\n{base_post:.0f} ms",
        xy=(121, base_post),
        xytext=(95, base_post + 430),
        fontsize=10,
        color=tspipe_run.color,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", lw=1.2, color=tspipe_run.color),
    )

    label_y = ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] else 1.0
    text_specs = [
        (failover_run.events.get("wall_detected"), "Detected\n57", EVENTS["wall_detected"]["color"]),
        (failover_run.events.get("trigger"), "Trigger\n60", EVENTS["trigger"]["color"]),
        (failover_run.events.get("replan"), "REPLAN\n80", EVENTS["replan"]["color"]),
        (failover_run.events.get("reeval"), "Reeval\n90", EVENTS["reeval"]["color"]),
    ]
    for step, label, color in text_specs:
        if step is None:
            continue
        ax.text(step + 1.2, label_y, label, color=color, fontsize=9.5, ha="left", va="top")

    ax.text(
        0.99,
        0.05,
        f"Post-REPLAN failover is {gain_vs_base_pct:.1f}% lower than TSPipe\nin the matched post-event window.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#2F2A24",
        bbox=dict(boxstyle="round,pad=0.35", fc="#F6EBD8", ec="none", alpha=0.95),
    )

    ax.set_title("B. Local Recovery: Failover Step Time Drops After REPLAN")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Per-step interval (ms)")
    ax.set_xlim(x_min, x_max)
    ax.grid(alpha=0.22, linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=tspipe_run.color, lw=3.0, label=tspipe_run.label),
        Line2D([0], [0], color=failover_run.color, lw=3.0, label=failover_run.label),
        Line2D([0], [0], color=EVENTS["replan"]["color"], lw=1.8, linestyle='-', label='REPLAN / restart window'),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)

    return {
        "failover_pre_ms": round(fail_pre, 3),
        "failover_post_ms": round(fail_post, 3),
        "tspipe_post_ms": round(base_post, 3),
        "failover_drop_pct": round(drop_pct, 3),
        "post_window_gain_vs_tspipe_pct": round(gain_vs_base_pct, 3),
    }


def plot_completion(ax, tspipe_run: RunMetrics, failover_run: RunMetrics) -> Dict[str, float]:
    runs = [tspipe_run, failover_run]
    minutes = [run.total_seconds / 60.0 for run in runs]
    bars = ax.bar(
        [run.label for run in runs],
        minutes,
        color=[run.color for run in runs],
        width=0.62,
        edgecolor="#2F2A24",
        linewidth=1.0,
    )

    for bar, run in zip(bars, runs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.24,
            f"{run.total_seconds // 60}m {run.total_seconds % 60}s",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#2F2A24",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(0.28, bar.get_height() * 0.08),
            f"restarts={run.restart_count}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#2F2A24",
            bbox=dict(boxstyle="round,pad=0.28", fc="#F2E8D5", ec="none", alpha=0.95),
        )

    saved_sec = tspipe_run.total_seconds - failover_run.total_seconds
    saved_pct = saved_sec / tspipe_run.total_seconds * 100.0 if tspipe_run.total_seconds else 0.0
    ax.annotate(
        f"135 s faster\n({saved_pct:.1f}% reduction)",
        xy=(1, minutes[1]),
        xytext=(0.5, max(minutes) + 1.35),
        textcoords="data",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#2F2A24",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-[,widthB=6.0,lengthB=0.9", lw=1.5, color="#2F2A24"),
    )

    ax.set_title("C. End-to-End Completion Time")
    ax.set_ylabel("Minutes")
    ax.set_ylim(0, max(minutes) + 3.3)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    return {
        "tspipe_total_seconds": tspipe_run.total_seconds,
        "failover_total_seconds": failover_run.total_seconds,
        "saved_seconds": saved_sec,
        "saved_pct": round(saved_pct, 3),
    }


def plot_figure(tspipe_run: RunMetrics, failover_run: RunMetrics, output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 220,
        }
    )

    fig = plt.figure(figsize=(12.4, 10.2), constrained_layout=True)
    fig.patch.set_facecolor("#FCFAF5")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], width_ratios=[1.7, 1.0])
    ax_progress = fig.add_subplot(grid[0, :])
    ax_interval = fig.add_subplot(grid[1, 0])
    ax_completion = fig.add_subplot(grid[1, 1])

    for ax in [ax_progress, ax_interval, ax_completion]:
        ax.set_facecolor("#FCFAF5")

    plot_progress(ax_progress, tspipe_run, failover_run)
    interval_stats = plot_interval_recovery(ax_interval, tspipe_run, failover_run)
    completion_stats = plot_completion(ax_completion, tspipe_run, failover_run)

    fig.suptitle(
        "Persistent GPU-3 Background Load: REPLAN Lowers Step Time and Shortens Training",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "Both runs face the same persistent external load on GPU 3. The failover run first detects slowdown, then triggers REPLAN at step 80, reduces step interval in the matched post-event window, and finishes earlier overall.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#4B4034",
    )

    png_path = output_base.with_suffix('.png')
    pdf_path = output_base.with_suffix('.pdf')
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    csv_rows = [
        {
            "label": tspipe_run.label,
            "run_dir": str(tspipe_run.run_dir),
            "total_seconds": tspipe_run.total_seconds,
            "restart_count": tspipe_run.restart_count,
            "wall_detected_step": tspipe_run.events.get('wall_detected'),
            "trigger_step": tspipe_run.events.get('trigger'),
            "replan_step": tspipe_run.events.get('replan'),
            "resume_step": tspipe_run.events.get('resume'),
            "reeval_step": tspipe_run.events.get('reeval'),
        },
        {
            "label": failover_run.label,
            "run_dir": str(failover_run.run_dir),
            "total_seconds": failover_run.total_seconds,
            "restart_count": failover_run.restart_count,
            "wall_detected_step": failover_run.events.get('wall_detected'),
            "trigger_step": failover_run.events.get('trigger'),
            "replan_step": failover_run.events.get('replan'),
            "resume_step": failover_run.events.get('resume'),
            "reeval_step": failover_run.events.get('reeval'),
        },
        {"label": "interval_stats", **interval_stats},
        {"label": "completion_stats", **completion_stats},
    ]
    save_csv(output_base.with_suffix('.csv'), csv_rows)

    print(png_path)
    print(pdf_path)
    print(output_base.with_suffix('.csv'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tspipe-run', required=True, type=Path)
    parser.add_argument('--failover-run', required=True, type=Path)
    parser.add_argument(
        '--output-base',
        type=Path,
        default=Path('/workspace/Synapse/Synapse/results/figures/bgload_gpu3_b128_paper_compare'),
    )
    args = parser.parse_args()

    tspipe_run = load_run(args.tspipe_run, 'TSPipe only', '#4C6A92')
    failover_run = load_run(args.failover_run, 'Failover + REPLAN', '#C46A2D')
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    plot_figure(tspipe_run, failover_run, args.output_base)


if __name__ == '__main__':
    main()
