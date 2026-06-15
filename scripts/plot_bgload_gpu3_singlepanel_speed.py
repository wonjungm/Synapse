#!/usr/bin/env python3
"""Create a white-background, single-panel speed figure for GPU-3 bgload runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_bgload_gpu3_onset_aligned import (
    FIGURES_DIR,
    RunSeries,
    anchored_plot_series,
    build_run,
    format_time_label,
    resolve_batch_size,
    resolve_inject_delay_sec,
    resolve_run_dirs,
    save_csv,
    visible_max,
)


def resolve_output_base(args: argparse.Namespace) -> Path:
    if args.output_base is not None:
        return args.output_base.resolve()
    if args.batch_size is None:
        raise SystemExit("Provide --batch-size, or pass --output-base explicitly.")
    return (FIGURES_DIR / f"bgload_gpu3_b{args.batch_size}_runspeed_singlepanel").resolve()


def plot_single_panel(tspipe: RunSeries, failover: RunSeries, output_base: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(10.2, 4.9), constrained_layout=True)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for run in (tspipe, failover):
        xs, ys = anchored_plot_series(run, x_min, x_max)
        ax.plot(xs, ys, color=run.color, linewidth=3.0, label=run.label)

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
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FFFFFF", "edgecolor": "#D9CBAA", "alpha": 0.96},
    )

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_base.with_suffix(".csv")
    save_csv(csv_path, [tspipe, failover])

    print(png_path)
    print(pdf_path)
    print(csv_path)


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
    plot_single_panel(tspipe, failover, output_base)


if __name__ == "__main__":
    main()
