#!/usr/bin/env python3
"""Generate one per-GPU step-time plot for a selected batch size.

Typical usage:
  python scripts/plot_bgload_gpu3_step_time_by_batch.py --batch-size 128

By default the graph zooms to a short window around the first REPLAN event so
that the post-REPLAN behavior is easier to read. Use `--full-range` to plot
all steps instead.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_bgload_batch_bar_comparison import RUNS as BGLOAD_RUNS
from plot_gpu_step_timeline import (
    GPU_COLORS,
    GPU_LINESTYLES,
    POLICY_COLORS,
    POLICY_MARKERS,
    ExperimentData,
    PolicyEvent,
    StepPoint,
    build_point_lookup,
    choose_gpu_ids,
    load_experiment,
    point_axis_value,
    resolve_run_step,
    select_marker_point,
    step_to_axis_value,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DEFAULT_BATCH_SIZE = 128
DEFAULT_PRE_STEPS = 8
DEFAULT_POST_STEPS = 60
RUN_KIND_CHOICES = ("failover", "tspipe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size to plot. Typical values: 64, 128, 256, 512.",
    )
    parser.add_argument(
        "--run-kind",
        choices=RUN_KIND_CHOICES,
        default="failover",
        help="Which mapped run to use for the selected batch size.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        help="Optional explicit experiment directory. Overrides --batch-size mapping.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        help="Optional explicit output base path without extension.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3],
        help="GPU ids to plot. Missing GPUs are skipped automatically.",
    )
    parser.add_argument(
        "--pre-steps",
        type=int,
        default=DEFAULT_PRE_STEPS,
        help="How many steps before REPLAN to include in the zoomed view.",
    )
    parser.add_argument(
        "--post-steps",
        type=int,
        default=DEFAULT_POST_STEPS,
        help="How many steps after REPLAN to include in the zoomed view.",
    )
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Plot the full step range instead of zooming around REPLAN.",
    )
    return parser.parse_args()


def resolve_experiment_dir(args: argparse.Namespace) -> Path:
    if args.experiment_dir is not None:
        path = args.experiment_dir.resolve()
        if path.is_dir() and path.name == "profiling_logs":
            return path.parent
        return path

    batch_runs = BGLOAD_RUNS.get(args.batch_size)
    if batch_runs is None:
        known = ", ".join(str(batch) for batch in sorted(BGLOAD_RUNS))
        raise SystemExit(
            f"No default run mapping for batch size {args.batch_size}. Known batch sizes: {known}"
        )

    run_name = batch_runs.get(args.run_kind)
    if not run_name:
        raise SystemExit(
            f"No '{args.run_kind}' run mapping available for batch size {args.batch_size}."
        )
    return (RESULTS_DIR / run_name).resolve()


def resolve_output_base(args: argparse.Namespace, experiment: ExperimentData) -> Path:
    if args.output_base is not None:
        return args.output_base.resolve()

    batch_label = (
        f"b{experiment.batch_size}" if experiment.batch_size is not None else "batch_unknown"
    )
    suffix = "policy_full" if args.full_range else "policy"
    if not experiment.policies:
        suffix = "timeline_full" if args.full_range else "timeline"
    return (FIGURES_DIR / f"bgload_gpu3_{batch_label}_per_gpu_step_{suffix}").resolve()


def collect_all_step_x(experiment: ExperimentData, selected_gpus: Sequence[int]) -> List[float]:
    all_x: List[float] = []
    for gpu_id in selected_gpus:
        for point in experiment.step_points_by_gpu[gpu_id]:
            x_value = point_axis_value(point, experiment, "step")
            if x_value is not None:
                all_x.append(x_value)
    return all_x


def resolve_replan_focus_anchor(experiment: ExperimentData) -> Optional[float]:
    anchors: List[float] = []
    for event in experiment.policies:
        if event.policy != "REPLAN":
            continue
        event_x = step_to_axis_value(experiment, event.run_index, event.step, "step")
        if event_x is not None:
            anchors.append(float(event_x))
    if not anchors:
        return None
    return min(anchors)


def resolve_focus_window(
    experiment: ExperimentData,
    selected_gpus: Sequence[int],
    pre_steps: int,
    post_steps: int,
    full_range: bool,
) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    all_x = collect_all_step_x(experiment, selected_gpus)
    if not all_x or full_range:
        return None, None

    anchor_x = resolve_replan_focus_anchor(experiment)
    if anchor_x is None:
        return None, None

    x_min = max(min(all_x), anchor_x - max(pre_steps, 0))
    x_max = min(max(all_x), anchor_x + max(post_steps, 1))
    if x_max <= x_min:
        return None, anchor_x
    return (x_min, x_max), anchor_x


def is_visible_x(x_value: float, focus_window: Optional[Tuple[float, float]]) -> bool:
    if focus_window is None:
        return True
    return focus_window[0] <= x_value <= focus_window[1]


def plot_gpu_lines(
    axis: plt.Axes,
    experiment: ExperimentData,
    selected_gpus: Sequence[int],
    focus_window: Optional[Tuple[float, float]],
) -> Tuple[List[float], List[float], Dict[int, Dict[Tuple[int, int], List[StepPoint]]]]:
    all_x: List[float] = []
    all_y: List[float] = []
    point_lookup_by_gpu: Dict[int, Dict[Tuple[int, int], List[StepPoint]]] = {}

    color_index_by_gpu = {gpu_id: idx for idx, gpu_id in enumerate(selected_gpus)}
    draw_order = sorted(selected_gpus, key=lambda gpu_id: (gpu_id != 3, gpu_id))

    for gpu_id in draw_order:
        points = experiment.step_points_by_gpu[gpu_id]
        point_lookup_by_gpu[gpu_id] = build_point_lookup(points)

        run_groups: Dict[int, List[StepPoint]] = {}
        for point in points:
            run_groups.setdefault(point.run_index, []).append(point)

        first_segment = True
        for run_index in sorted(run_groups.keys()):
            xs: List[float] = []
            ys: List[float] = []

            for point in run_groups[run_index]:
                x_value = point_axis_value(point, experiment, "step")
                if x_value is None or not is_visible_x(x_value, focus_window):
                    continue
                xs.append(x_value)
                ys.append(point.step_time_ms)
                all_x.append(x_value)
                all_y.append(point.step_time_ms)

            if not xs:
                continue

            axis.plot(
                xs,
                ys,
                label=f"GPU {gpu_id}" if first_segment else None,
                color=GPU_COLORS[color_index_by_gpu[gpu_id] % len(GPU_COLORS)],
                linestyle=GPU_LINESTYLES.get(gpu_id, "-"),
                linewidth=2.2 if gpu_id == 3 else 1.8,
                alpha=0.78 if gpu_id == 3 else 0.92,
                zorder=2 if gpu_id == 3 else 3,
            )
            first_segment = False

    return all_x, all_y, point_lookup_by_gpu


def marker_y_value(
    event: PolicyEvent,
    event_x: float,
    axis_top: float,
    experiment: ExperimentData,
    point_lookup_by_gpu: Dict[int, Dict[Tuple[int, int], List[StepPoint]]],
) -> float:
    if event.gpu_id is None or event.gpu_id not in point_lookup_by_gpu:
        return axis_top * 0.92

    resolved_step = resolve_run_step(experiment, event.run_index, event.step)
    step_key = (event.run_index, resolved_step if resolved_step is not None else event.step)
    matched = select_marker_point(
        point_lookup_by_gpu[event.gpu_id].get(step_key, []),
        experiment,
        "step",
        event_x,
    )
    if matched is None:
        return axis_top * 0.92
    return min(max(matched.step_time_ms, axis_top * 0.05), axis_top * 0.94)


def annotate_policy_event(axis: plt.Axes, event_x: float, event_y: float, event: PolicyEvent) -> None:
    label = f"{event.policy}\nstep {event.step}"
    if event.gpu_id is not None:
        label = f"{label} / GPU {event.gpu_id}"

    axis.annotate(
        label,
        xy=(event_x, event_y),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color="#2F2A24",
        ha="left",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#D6D3D1",
            "alpha": 0.88,
        },
    )


def draw_policy_markers(
    axis: plt.Axes,
    experiment: ExperimentData,
    axis_top: float,
    point_lookup_by_gpu: Dict[int, Dict[Tuple[int, int], List[StepPoint]]],
    focus_window: Optional[Tuple[float, float]],
) -> Set[str]:
    seen_policies: Set[str] = set()
    for event in experiment.policies:
        event_x = step_to_axis_value(experiment, event.run_index, event.step, "step")
        if event_x is None or not is_visible_x(event_x, focus_window):
            continue

        marker = POLICY_MARKERS.get(event.policy, "X")
        color = POLICY_COLORS.get(event.policy, "#666666")
        event_y = marker_y_value(event, event_x, axis_top, experiment, point_lookup_by_gpu)

        axis.scatter(
            [event_x],
            [event_y],
            marker=marker,
            s=96,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
        )
        annotate_policy_event(axis, event_x, event_y, event)
        seen_policies.add(event.policy)

    return seen_policies


def build_policy_legend_handles(seen_policies: Iterable[str]) -> List[Line2D]:
    handles: List[Line2D] = []
    for policy in ("KEEP", "REPLAN", "DEGRADE"):
        if policy not in seen_policies:
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                marker=POLICY_MARKERS.get(policy, "X"),
                color="none",
                markerfacecolor=POLICY_COLORS.get(policy, "#666666"),
                markeredgecolor="black",
                markeredgewidth=0.7,
                markersize=8.5,
                label=f"Policy: {policy}",
            )
        )
    return handles


def write_csv(
    output_path: Path,
    experiment: ExperimentData,
    selected_gpus: Sequence[int],
    focus_window: Optional[Tuple[float, float]],
) -> None:
    fieldnames = [
        "row_type",
        "batch_size",
        "experiment_dir",
        "gpu_id",
        "run_index",
        "raw_step",
        "plot_step",
        "step_time_ms",
        "sum_time_ms",
        "start_time_sec",
        "policy",
        "slowdown_ratio",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for gpu_id in selected_gpus:
            for point in experiment.step_points_by_gpu[gpu_id]:
                x_value = point_axis_value(point, experiment, "step")
                if x_value is None or not is_visible_x(x_value, focus_window):
                    continue
                writer.writerow(
                    {
                        "row_type": "step_point",
                        "batch_size": experiment.batch_size,
                        "experiment_dir": str(experiment.path),
                        "gpu_id": gpu_id,
                        "run_index": point.run_index,
                        "raw_step": point.raw_step,
                        "plot_step": point.plot_step,
                        "step_time_ms": f"{point.step_time_ms:.6f}",
                        "sum_time_ms": f"{point.sum_time_ms:.6f}",
                        "start_time_sec": (
                            f"{point.start_time_sec:.6f}"
                            if point.start_time_sec is not None
                            else ""
                        ),
                        "policy": "",
                        "slowdown_ratio": "",
                    }
                )

        for event in experiment.policies:
            event_x = step_to_axis_value(experiment, event.run_index, event.step, "step")
            if event_x is None or not is_visible_x(event_x, focus_window):
                continue
            writer.writerow(
                {
                    "row_type": "policy_event",
                    "batch_size": experiment.batch_size,
                    "experiment_dir": str(experiment.path),
                    "gpu_id": event.gpu_id if event.gpu_id is not None else "",
                    "run_index": event.run_index,
                    "raw_step": event.step,
                    "plot_step": event_x,
                    "step_time_ms": "",
                    "sum_time_ms": "",
                    "start_time_sec": "",
                    "policy": event.policy,
                    "slowdown_ratio": (
                        f"{event.slowdown_ratio:.6f}"
                        if event.slowdown_ratio is not None
                        else ""
                    ),
                }
            )


def plot_experiment(
    experiment: ExperimentData,
    output_base: Path,
    gpu_ids: Optional[Sequence[int]],
    pre_steps: int,
    post_steps: int,
    full_range: bool,
) -> List[Path]:
    selected_gpus = choose_gpu_ids(experiment, gpu_ids)
    if not selected_gpus:
        raise ValueError(f"No matching GPUs found in {experiment.path}")

    focus_window, anchor_x = resolve_focus_window(
        experiment,
        selected_gpus,
        pre_steps=pre_steps,
        post_steps=post_steps,
        full_range=full_range,
    )

    fig, axis = plt.subplots(figsize=(15.0, 7.0))
    all_x, all_y, point_lookup_by_gpu = plot_gpu_lines(axis, experiment, selected_gpus, focus_window)
    if not all_x or not all_y:
        plt.close(fig)
        raise ValueError(f"No plottable step-time points found in {experiment.path}")

    y_max = max(all_y)
    y_padding = max(y_max * 0.12, 12.0)
    axis_top = y_max + y_padding
    seen_policies = draw_policy_markers(axis, experiment, axis_top, point_lookup_by_gpu, focus_window)

    if focus_window is not None and anchor_x is not None and is_visible_x(anchor_x, focus_window):
        axis.axvline(
            anchor_x,
            color=POLICY_COLORS.get("REPLAN", "#059669"),
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            zorder=1,
        )

    axis.set_xlim(min(all_x), max(all_x))
    axis.set_ylim(0.0, axis_top)
    axis.set_xlabel("Step")
    axis.set_ylabel("Step Time (ms)")
    axis.grid(True, axis="y", linestyle="--", alpha=0.28)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    batch_label = f"Batch Size {experiment.batch_size}" if experiment.batch_size is not None else "Batch Size Unknown"
    if focus_window is not None and anchor_x is not None:
        title_suffix = f"REPLAN zoom ({pre_steps} before, {post_steps} after)"
    elif full_range:
        title_suffix = "Full range"
    else:
        title_suffix = "Focused range"
    axis.set_title(
        f"{batch_label}: Per-GPU Step Time During Slowdown and Recovery\n"
        f"{experiment.path.name} | {title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    line_handles, line_labels = axis.get_legend_handles_labels()
    policy_handles = build_policy_legend_handles(seen_policies)
    handles = line_handles + policy_handles
    labels = line_labels + [handle.get_label() for handle in policy_handles]
    dedup_handles: List[object] = []
    dedup_labels: List[str] = []
    seen_labels: Set[str] = set()
    for handle, label in zip(handles, labels):
        if not label or label in seen_labels:
            continue
        dedup_handles.append(handle)
        dedup_labels.append(label)
        seen_labels.add(label)

    if dedup_handles:
        axis.legend(
            dedup_handles,
            dedup_labels,
            loc="upper right",
            frameon=True,
            framealpha=0.92,
            ncol=2,
            fontsize=9,
        )

    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    csv_path = output_base.with_suffix(".csv")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    write_csv(csv_path, experiment, selected_gpus, focus_window)
    return [png_path, pdf_path, csv_path]


def main() -> int:
    args = parse_args()
    exp_dir = resolve_experiment_dir(args)
    experiment = load_experiment(exp_dir)
    output_base = resolve_output_base(args, experiment)
    outputs = plot_experiment(
        experiment,
        output_base,
        args.gpu_ids,
        pre_steps=args.pre_steps,
        post_steps=args.post_steps,
        full_range=args.full_range,
    )

    print(f"[OK] {exp_dir}")
    for output_path in outputs:
        print(f"  -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
