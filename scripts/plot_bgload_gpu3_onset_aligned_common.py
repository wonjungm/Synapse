#!/usr/bin/env python3
"""Create an injection-aligned comparison figure using a shared pre-injection baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from plot_bgload_gpu3_onset_aligned import (
    RunSeries,
    build_run,
    format_optional_float,
    infer_batch_size_from_name,
    relative_seconds,
    resolve_inject_delay_sec,
    step_elapsed_seconds,
    visible_max,
)


def resolve_batch_size(
    batch_size: Optional[int],
    tspipe_run: Path,
    failover_run: Path,
) -> int:
    if batch_size is not None:
        return int(batch_size)

    inferred = {
        size
        for size in (
            infer_batch_size_from_name(tspipe_run.name),
            infer_batch_size_from_name(failover_run.name),
        )
        if size is not None
    }
    if len(inferred) == 1:
        return inferred.pop()
    raise SystemExit(
        "Could not infer a unique batch size from the run directory names. "
        "Provide --batch-size explicitly."
    )


def choose_shared_baseline(tspipe: RunSeries, failover: RunSeries, source: str) -> float:
    if source == "tspipe":
        return tspipe.baseline_ms
    if source == "failover":
        return failover.baseline_ms
    if source == "mean":
        return (tspipe.baseline_ms + failover.baseline_ms) / 2.0
    raise ValueError(f"Unsupported baseline source: {source}")


def save_csv(path: Path, runs: Iterable[RunSeries], baseline_source: str, shared_baseline_ms: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "run_dir",
                "batch_size",
                "inject_delay_sec",
                "total_seconds",
                "restart_count",
                "individual_baseline_ms",
                "shared_baseline_source",
                "shared_baseline_ms",
                "wall_detected_step",
                "wall_detected_elapsed_sec",
                "wall_detected_relative_sec",
                "replan_step",
                "replan_elapsed_sec",
                "replan_relative_sec",
                "reeval_step",
                "reeval_elapsed_sec",
                "reeval_relative_sec",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "label": run.label,
                    "run_dir": str(run.run_dir),
                    "batch_size": run.batch_size,
                    "inject_delay_sec": f"{run.inject_delay_sec:.1f}",
                    "total_seconds": run.total_seconds,
                    "restart_count": run.restart_count,
                    "individual_baseline_ms": f"{run.baseline_ms:.4f}",
                    "shared_baseline_source": baseline_source,
                    "shared_baseline_ms": f"{shared_baseline_ms:.4f}",
                    "wall_detected_step": run.wall_detected_step,
                    "wall_detected_elapsed_sec": format_optional_float(
                        step_elapsed_seconds(run, run.wall_detected_step)
                    ),
                    "wall_detected_relative_sec": format_optional_float(
                        relative_seconds(run, run.wall_detected_step)
                    ),
                    "replan_step": run.replan_step,
                    "replan_elapsed_sec": format_optional_float(
                        step_elapsed_seconds(run, run.replan_step)
                    ),
                    "replan_relative_sec": format_optional_float(
                        relative_seconds(run, run.replan_step)
                    ),
                    "reeval_step": run.reeval_step,
                    "reeval_elapsed_sec": format_optional_float(
                        step_elapsed_seconds(run, run.reeval_step)
                    ),
                    "reeval_relative_sec": format_optional_float(
                        relative_seconds(run, run.reeval_step)
                    ),
                }
            )


def plot_aligned(tspipe: RunSeries, failover: RunSeries, output_base: Path, baseline_source: str) -> None:
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

    shared_baseline_ms = choose_shared_baseline(tspipe, failover, baseline_source)
    tspipe_speed = shared_baseline_ms / np.maximum(tspipe.smooth_ms, 1e-9)
    failover_speed = shared_baseline_ms / np.maximum(failover.smooth_ms, 1e-9)

    rel_points = [0.0]
    for run, step in (
        (failover, failover.replan_step),
        (failover, failover.reeval_step),
        (failover, failover.wall_detected_step),
        (tspipe, tspipe.wall_detected_step),
    ):
        rel = relative_seconds(run, step)
        if rel is not None:
            rel_points.append(rel)
    x_min = min(-max(30.0, failover.inject_delay_sec * 0.35), min(rel_points) - 20.0)
    x_max = max(120.0, max(rel_points) + 40.0)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.6, 7.9),
        gridspec_kw={"height_ratios": [1.65, 1.0]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#FCFAF5")
    for ax in axes:
        ax.set_facecolor("#FCFAF5")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    ax = axes[0]
    for run, speed in ((tspipe, tspipe_speed), (failover, failover_speed)):
        mask = (run.aligned_seconds >= x_min) & (run.aligned_seconds <= x_max)
        ax.plot(run.aligned_seconds[mask], speed[mask], color=run.color, linewidth=3.0, label=run.label)

    tspipe_mask = (tspipe.aligned_seconds >= x_min) & (tspipe.aligned_seconds <= x_max)
    failover_mask = (failover.aligned_seconds >= x_min) & (failover.aligned_seconds <= x_max)
    y_max = max(
        visible_max(tspipe_speed, tspipe_mask),
        visible_max(failover_speed, failover_mask),
    )

    ax.axvline(0, color="#E67E22", linestyle="--", linewidth=1.4, alpha=0.85)
    ax.text(
        2.0,
        y_max * 0.98,
        f"bgload injected (+{failover.inject_delay_sec:.0f}s)",
        ha="left",
        va="top",
        fontsize=10,
        color="#E67E22",
    )

    replan_rel = relative_seconds(failover, failover.replan_step)
    reeval_rel = relative_seconds(failover, failover.reeval_step)
    if replan_rel is not None:
        ax.axvline(replan_rel, color="#059669", linestyle="-", linewidth=2.0)
        ax.text(
            replan_rel + 3.0,
            y_max * 0.90,
            "REPLAN",
            ha="left",
            va="top",
            fontsize=11,
            color="#059669",
            fontweight="bold",
        )
    if replan_rel is not None and reeval_rel is not None:
        ax.axvspan(replan_rel, reeval_rel, color="#F3E3B5", alpha=0.45)

    title_suffix = {
        "tspipe": "shared TSPipe baseline",
        "failover": "shared failover baseline",
        "mean": "shared mean baseline",
    }[baseline_source]
    ax.set_title(f"GPU-3 compute speed ({title_suffix})")
    ax.set_xlabel("Seconds from bgload injection")
    ax.set_ylabel("Relative speed (x)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, y_max * 1.08)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.03), frameon=False)
    ax.text(
        0.01,
        0.02,
        f"Speed = shared pre-injection baseline ({shared_baseline_ms:.1f} ms) / smoothed compute time",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        color="#5B534A",
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
    save_csv(csv_path, runs, baseline_source, shared_baseline_ms)

    print(png_path)
    print(pdf_path)
    print(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tspipe-run", required=True, type=Path)
    parser.add_argument("--failover-run", required=True, type=Path)
    parser.add_argument("--output-base", required=True, type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--inject-delay-sec",
        type=float,
        help="Optional explicit bgload injection delay in seconds. Defaults to batch-specific metadata.",
    )
    parser.add_argument(
        "--common-baseline-source",
        choices=["tspipe", "failover", "mean"],
        default="tspipe",
    )
    args = parser.parse_args()

    tspipe_run = args.tspipe_run.resolve()
    failover_run = args.failover_run.resolve()
    batch_size = resolve_batch_size(args.batch_size, tspipe_run, failover_run)
    inject_delay_sec = resolve_inject_delay_sec(args, batch_size)

    tspipe = build_run(tspipe_run, "TSPipe baseline", "#4C6A92", batch_size, inject_delay_sec)
    failover = build_run(failover_run, "Failover + REPLAN", "#C46A2D", batch_size, inject_delay_sec)
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    plot_aligned(tspipe, failover, args.output_base.resolve(), args.common_baseline_source)


if __name__ == "__main__":
    main()
