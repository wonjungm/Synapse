#!/usr/bin/env python3
"""Create a run-start aligned throughput comparison figure for GPU-3 background load.

Typical usage:
  python scripts/plot_bgload_gpu3_onset_aligned.py --batch-size 128

You can still pass explicit run directories with `--tspipe-run` and
`--failover-run`, but if you omit them the script uses the shared batch mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_bgload_batch_bar_comparison import RUNS as BGLOAD_RUNS

COMPUTE_TASKS = {"compute_forward", "compute_backward", "compute_optimize"}
WALL_DETECTED_RE = re.compile(r"Wall-clock slowdown detected .*global_step=(\d+)")
POLICY_RE = re.compile(r"Failover Policy Decision: ([A-Z_]+) \(step=(\d+)")
REEVAL_RE = re.compile(r"reevaluation resumes at step (\d+)")
SLOWDOWN_BASELINE_RE = re.compile(r"Baseline set after skip=(\d+):")
RUN_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")
BATCH_SIZE_RE = re.compile(r"(?:^|[_-])b(\d+)(?:[_-]|$)")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SLOWDOWN_BASELINE_WINDOW = 10
DEFAULT_SLOWDOWN_BASELINE_SKIP_STEPS = 20

BGLOAD_INJECT_DELAY_SEC = {
    64: 120.0,
    128: 120.0,
    256: 120.0,
    512: 300.0,
}


@dataclass
class RunSeries:
    label: str
    batch_size: int
    inject_start_sec: float
    run_dir: Path
    total_seconds: int
    restart_count: int
    steps: np.ndarray
    values_ms: np.ndarray
    smooth_ms: np.ndarray
    slowdown_baseline_step: Optional[int]
    slowdown_baseline_elapsed_sec: Optional[float]
    wall_detected_step: Optional[int]
    wall_detected_elapsed_sec: Optional[float]
    replan_step: Optional[int]
    replan_elapsed_sec: Optional[float]
    reeval_step: Optional[int]
    reeval_elapsed_sec: Optional[float]
    baseline_ms: float
    baseline_speed_ratio: float
    elapsed_seconds: np.ndarray
    speed_ratio: np.ndarray
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


def rolling_median(values: Sequence[float], window: int = 9) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if arr.size == 1 or window <= 1:
        return arr.copy()
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.asarray([np.median(padded[idx : idx + window]) for idx in range(arr.size)], dtype=float)


def smooth_series(values: Sequence[float]) -> np.ndarray:
    return rolling_mean(rolling_median(values, 9), 5)


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
        raise ValueError(f"Could not parse total time from {path}")
    return total_seconds, restart_count


def parse_log_metadata(log_path: Path) -> Dict[str, Optional[int]]:
    metadata: Dict[str, Optional[int]] = {
        "wall_detected": None,
        "replan": None,
        "reeval": None,
        "slowdown_baseline_skip_steps": None,
    }
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if metadata["wall_detected"] is None:
            match = WALL_DETECTED_RE.search(line)
            if match:
                metadata["wall_detected"] = int(match.group(1))
        if metadata["reeval"] is None:
            match = REEVAL_RE.search(line)
            if match:
                metadata["reeval"] = int(match.group(1))
        if metadata["slowdown_baseline_skip_steps"] is None:
            match = SLOWDOWN_BASELINE_RE.search(line)
            if match:
                metadata["slowdown_baseline_skip_steps"] = int(match.group(1))
        match = POLICY_RE.search(line)
        if match and match.group(1) == "REPLAN":
            metadata["replan"] = int(match.group(2))
    return metadata


def infer_batch_size_from_name(name: str) -> Optional[int]:
    match = BATCH_SIZE_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def parse_run_start_epoch(run_dir: Path) -> float:
    match = RUN_TIMESTAMP_RE.search(run_dir.name)
    if match is None:
        raise ValueError(f"Could not parse run timestamp from {run_dir.name}")
    dt = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def resolve_batch_size(args: argparse.Namespace, tspipe_run: Path, failover_run: Path) -> int:
    if args.batch_size is not None:
        return int(args.batch_size)

    inferred = {
        batch_size
        for batch_size in (
            infer_batch_size_from_name(tspipe_run.name),
            infer_batch_size_from_name(failover_run.name),
        )
        if batch_size is not None
    }
    if len(inferred) == 1:
        return inferred.pop()
    raise SystemExit(
        "Could not infer a unique batch size from the run directory names. "
        "Provide --batch-size explicitly."
    )


def resolve_inject_delay_sec(args: argparse.Namespace, batch_size: int) -> float:
    if args.inject_delay_sec is not None:
        return float(args.inject_delay_sec)
    inject_delay_sec = BGLOAD_INJECT_DELAY_SEC.get(int(batch_size))
    if inject_delay_sec is None:
        raise SystemExit(
            f"No default bgload injection delay is known for batch size {batch_size}. "
            "Provide --inject-delay-sec explicitly."
        )
    return float(inject_delay_sec)


def compute_pre_injection_baseline_ms(
    values_ms: np.ndarray,
    elapsed_seconds: np.ndarray,
    inject_start_sec: float,
) -> float:
    baseline_values = values_ms[elapsed_seconds < inject_start_sec]
    if baseline_values.size == 0:
        baseline_values = values_ms[: min(30, len(values_ms))]
    if baseline_values.size == 0:
        return 1.0
    baseline_ms = float(np.median(baseline_values))
    if baseline_ms <= 0:
        baseline_ms = float(np.mean(baseline_values))
    return baseline_ms if baseline_ms > 0 else 1.0


def load_partition3_compute(run_dir: Path, run_start_epoch: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    profiling_dir = run_dir / "profiling_logs"
    values = defaultdict(float)
    step_completion = {}

    for path in sorted(profiling_dir.glob("gpu_task_summary_partition*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if bool(row.get("target")):
                    continue

                global_step = row.get("global_step")
                if global_step is None:
                    continue
                step = int(global_step)
                task_name = str(row.get("task_name", ""))
                start_time = row.get("start_time")
                if start_time is not None:
                    start_time = float(start_time)

                if int(row.get("partition", -1)) == 3 and task_name in COMPUTE_TASKS:
                    values[step] += float(row.get("exec_wall_ms", row.get("time_ms", 0.0)) or 0.0)

                if task_name == "compute_optimize" and start_time is not None:
                    end_time = start_time + float(row.get("wall_ms", row.get("time_ms", 0.0)) or 0.0) / 1000.0
                    prev = step_completion.get(step)
                    if prev is None or end_time > prev:
                        step_completion[step] = end_time

    step_keys = sorted(set(values.keys()) & set(step_completion.keys()))
    if not step_keys:
        step_keys = sorted(values.keys())

    steps = np.asarray(step_keys, dtype=int)
    series = np.asarray([values[int(step)] for step in steps], dtype=float)
    elapsed_seconds = np.asarray(
        [max(0.0, step_completion[int(step)] - run_start_epoch) for step in steps],
        dtype=float,
    )
    return steps, series, elapsed_seconds


def event_elapsed_seconds(run_steps: np.ndarray, elapsed_seconds: np.ndarray, step: Optional[int]) -> Optional[float]:
    if step is None:
        return None
    matches = np.where(run_steps == int(step))[0]
    if matches.size == 0:
        return None
    return float(elapsed_seconds[int(matches[0])])


def format_optional_float(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.3f}"


def build_run(
    run_dir: Path,
    label: str,
    color: str,
    batch_size: int,
    inject_start_sec: float,
) -> RunSeries:
    run_start_epoch = parse_run_start_epoch(run_dir)
    total_seconds, restart_count = parse_summary(run_dir / "e2e_summary.log")
    metadata = parse_log_metadata(run_dir / "log.txt")
    steps, values_ms, elapsed_seconds = load_partition3_compute(run_dir, run_start_epoch)
    smooth_ms = smooth_series(values_ms)
    baseline_ms = compute_pre_injection_baseline_ms(values_ms, elapsed_seconds, inject_start_sec)
    baseline_speed_ratio = 1.0
    speed_ratio = baseline_ms / np.maximum(smooth_ms, 1e-9)

    slowdown_baseline_skip_steps = metadata.get("slowdown_baseline_skip_steps")
    if slowdown_baseline_skip_steps is None:
        slowdown_baseline_skip_steps = DEFAULT_SLOWDOWN_BASELINE_SKIP_STEPS
    slowdown_baseline_index = int(slowdown_baseline_skip_steps) + SLOWDOWN_BASELINE_WINDOW - 1
    slowdown_baseline_step = None
    if 0 <= slowdown_baseline_index < len(steps):
        slowdown_baseline_step = int(steps[slowdown_baseline_index])

    return RunSeries(
        label=label,
        batch_size=batch_size,
        inject_start_sec=float(inject_start_sec),
        run_dir=run_dir,
        total_seconds=total_seconds,
        restart_count=restart_count,
        steps=steps,
        values_ms=values_ms,
        smooth_ms=smooth_ms,
        slowdown_baseline_step=slowdown_baseline_step,
        slowdown_baseline_elapsed_sec=event_elapsed_seconds(steps, elapsed_seconds, slowdown_baseline_step),
        wall_detected_step=metadata.get("wall_detected"),
        wall_detected_elapsed_sec=event_elapsed_seconds(steps, elapsed_seconds, metadata.get("wall_detected")),
        replan_step=metadata.get("replan"),
        replan_elapsed_sec=event_elapsed_seconds(steps, elapsed_seconds, metadata.get("replan")),
        reeval_step=metadata.get("reeval"),
        reeval_elapsed_sec=event_elapsed_seconds(steps, elapsed_seconds, metadata.get("reeval")),
        baseline_ms=baseline_ms,
        baseline_speed_ratio=baseline_speed_ratio,
        elapsed_seconds=elapsed_seconds,
        speed_ratio=speed_ratio,
        color=color,
    )


def save_csv(path: Path, runs: Iterable[RunSeries]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "run_dir",
                "batch_size",
                "inject_start_sec",
                "total_seconds",
                "restart_count",
                "baseline_ms",
                "baseline_speed_ratio",
                "slowdown_baseline_step",
                "slowdown_baseline_elapsed_sec",
                "wall_detected_step",
                "wall_detected_elapsed_sec",
                "replan_step",
                "replan_elapsed_sec",
                "reeval_step",
                "reeval_elapsed_sec",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "label": run.label,
                    "run_dir": str(run.run_dir),
                    "batch_size": run.batch_size,
                    "inject_start_sec": f"{run.inject_start_sec:.3f}",
                    "total_seconds": run.total_seconds,
                    "restart_count": run.restart_count,
                    "baseline_ms": f"{run.baseline_ms:.4f}",
                    "baseline_speed_ratio": f"{run.baseline_speed_ratio:.4f}",
                    "slowdown_baseline_step": run.slowdown_baseline_step,
                    "slowdown_baseline_elapsed_sec": format_optional_float(run.slowdown_baseline_elapsed_sec),
                    "wall_detected_step": run.wall_detected_step,
                    "wall_detected_elapsed_sec": format_optional_float(run.wall_detected_elapsed_sec),
                    "replan_step": run.replan_step,
                    "replan_elapsed_sec": format_optional_float(run.replan_elapsed_sec),
                    "reeval_step": run.reeval_step,
                    "reeval_elapsed_sec": format_optional_float(run.reeval_elapsed_sec),
                }
            )


def visible_max(values: np.ndarray, mask: np.ndarray) -> float:
    if np.any(mask):
        return float(np.nanmax(values[mask]))
    return float(np.nanmax(values))


def anchored_plot_series(run: RunSeries, x_min: float, x_max: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = (run.elapsed_seconds >= x_min) & (run.elapsed_seconds <= x_max)
    xs = run.elapsed_seconds[mask]
    ys = run.speed_ratio[mask]
    if xs.size == 0:
        return xs, ys
    if xs[0] > 0.0:
        xs = np.concatenate(([0.0], xs))
        ys = np.concatenate(([0.0], ys))
    return xs, ys


def format_time_label(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}s"


def plot_aligned(tspipe: RunSeries, failover: RunSeries, output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 160,
        }
    )

    marker_times = [tspipe.inject_start_sec, failover.inject_start_sec]
    for run in (tspipe, failover):
        for value in (
            run.slowdown_baseline_elapsed_sec,
            run.wall_detected_elapsed_sec,
            run.replan_elapsed_sec,
            run.reeval_elapsed_sec,
        ):
            if value is not None:
                marker_times.append(value)

    max_total_seconds = max(tspipe.total_seconds, failover.total_seconds)
    x_max = max(max(marker_times) + 60.0, max(tspipe.inject_start_sec, failover.inject_start_sec) + 180.0)
    x_max = min(float(max_total_seconds), x_max)
    x_min = 0.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.2, 8.2),
        gridspec_kw={"height_ratios": [1.7, 1.0]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#FCFAF5")
    for ax in axes:
        ax.set_facecolor("#FCFAF5")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    ax = axes[0]
    for run in (tspipe, failover):
        xs, ys = anchored_plot_series(run, x_min, x_max)
        ax.plot(
            xs,
            ys,
            color=run.color,
            linewidth=3.0,
            label=run.label,
        )

    if tspipe.slowdown_baseline_elapsed_sec is not None:
        ax.axvline(
            tspipe.slowdown_baseline_elapsed_sec,
            color=tspipe.color,
            linestyle=":",
            linewidth=1.9,
            alpha=0.95,
        )
    if failover.slowdown_baseline_elapsed_sec is not None:
        ax.axvline(
            failover.slowdown_baseline_elapsed_sec,
            color=failover.color,
            linestyle=":",
            linewidth=1.9,
            alpha=0.95,
        )

    inject_marker_sec = float(np.mean([tspipe.inject_start_sec, failover.inject_start_sec]))
    ax.axvline(inject_marker_sec, color="#E67E22", linestyle="--", linewidth=1.9, alpha=0.9)

    tspipe_mask = (tspipe.elapsed_seconds >= x_min) & (tspipe.elapsed_seconds <= x_max)
    failover_mask = (failover.elapsed_seconds >= x_min) & (failover.elapsed_seconds <= x_max)
    y_max = max(
        visible_max(tspipe.speed_ratio, tspipe_mask),
        visible_max(failover.speed_ratio, failover_mask),
    )

    ax.text(
        inject_marker_sec + 3.0,
        y_max * 0.98,
        f"external bgload starts ({inject_marker_sec:.0f}s)",
        ha="left",
        va="top",
        fontsize=10,
        color="#E67E22",
    )

    if failover.replan_elapsed_sec is not None:
        ax.axvline(failover.replan_elapsed_sec, color="#059669", linestyle="-", linewidth=2.0)
        ax.text(
            failover.replan_elapsed_sec + 3.0,
            y_max * 0.90,
            "REPLAN",
            ha="left",
            va="top",
            fontsize=11,
            color="#059669",
            fontweight="bold",
        )
    if failover.replan_elapsed_sec is not None and failover.reeval_elapsed_sec is not None:
        ax.axvspan(failover.replan_elapsed_sec, failover.reeval_elapsed_sec, color="#F3E3B5", alpha=0.45)

    ax.set_title("GPU-3 compute speed from run start")
    ax.set_xlabel("Seconds from run start")
    ax.set_ylabel("Speed (x baseline)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, y_max * 1.08)
    ax.grid(alpha=0.25, linestyle="--")

    run_legend = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.03), frameon=False)
    marker_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="#E67E22", linestyle="--", linewidth=1.9, label="External bgload start"),
            Line2D([0], [0], color=tspipe.color, linestyle=":", linewidth=1.9, label="TSPipe slowdown baseline"),
            Line2D([0], [0], color=failover.color, linestyle=":", linewidth=1.9, label="Failover slowdown baseline"),
            Line2D([0], [0], color="#059669", linestyle="-", linewidth=2.0, label="REPLAN"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.03),
        frameon=False,
    )
    ax.add_artist(run_legend)
    ax.add_artist(marker_legend)

    summary_lines = [
        f"bgload start: {inject_marker_sec:.1f}s",
        f"TSPipe slowdown baseline: {format_time_label(tspipe.slowdown_baseline_elapsed_sec)}",
        f"Failover slowdown baseline: {format_time_label(failover.slowdown_baseline_elapsed_sec)}",
        "speed = pre-injection baseline compute time / current compute time",
    ]
    ax.text(
        0.01,
        0.02,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        color="#5B534A",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F7F1E5", "edgecolor": "#D9CBAA", "alpha": 0.92},
    )

    ax = axes[1]
    runs = [tspipe, failover]
    bars = ax.bar(
        [run.label for run in runs],
        [run.total_seconds / 60.0 for run in runs],
        color=[run.color for run in runs],
        width=0.56,
        edgecolor="#2F2A24",
        linewidth=1.0,
    )
    for bar, run in zip(bars, runs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.18,
            f"{run.total_seconds / 60.0:.1f}m",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#2F2A24",
        )
    ax.set_title("Completion time")
    ax.set_ylabel("Minutes")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_ylim(0, max(run.total_seconds / 60.0 for run in runs) + 3.0)

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


def resolve_run_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.tspipe_run is not None and args.failover_run is not None:
        return args.tspipe_run.resolve(), args.failover_run.resolve()

    if args.tspipe_run is not None or args.failover_run is not None:
        raise SystemExit("Provide both --tspipe-run and --failover-run, or omit both and use --batch-size.")

    if args.batch_size is None:
        raise SystemExit("Provide --batch-size, or pass both --tspipe-run and --failover-run.")

    batch_runs = BGLOAD_RUNS.get(args.batch_size)
    if batch_runs is None:
        known = ", ".join(str(batch) for batch in sorted(BGLOAD_RUNS))
        raise SystemExit(
            f"No default run mapping for batch size {args.batch_size}. Known batch sizes: {known}"
        )

    return (
        (RESULTS_DIR / batch_runs["tspipe"]).resolve(),
        (RESULTS_DIR / batch_runs["failover"]).resolve(),
    )


def resolve_output_base(args: argparse.Namespace) -> Path:
    if args.output_base is not None:
        return args.output_base.resolve()
    if args.batch_size is None:
        raise SystemExit("Provide --batch-size, or pass --output-base explicitly.")
    return (FIGURES_DIR / f"bgload_gpu3_b{args.batch_size}_onset_aligned_compare").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tspipe-run", type=Path)
    parser.add_argument("--failover-run", type=Path)
    parser.add_argument("--output-base", type=Path)
    parser.add_argument(
        "--inject-delay-sec",
        type=float,
        help="Optional explicit external bgload start delay in seconds from run start.",
    )
    args = parser.parse_args()

    tspipe_run, failover_run = resolve_run_dirs(args)
    output_base = resolve_output_base(args)
    batch_size = resolve_batch_size(args, tspipe_run, failover_run)
    inject_delay_sec = resolve_inject_delay_sec(args, batch_size)

    tspipe = build_run(tspipe_run, "TSPipe baseline", "#4C6A92", batch_size, inject_delay_sec)
    failover = build_run(failover_run, "Failover + REPLAN", "#C46A2D", batch_size, inject_delay_sec)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    plot_aligned(tspipe, failover, output_base)


if __name__ == "__main__":
    main()
