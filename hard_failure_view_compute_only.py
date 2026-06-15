import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from hard_failure_view import (
    BASE_RUN,
    HARD_RUN,
    build_hard_series_with_pause,
    build_progress_series,
    estimate_recovery_overhead,
    get_total_minutes,
    load_checkpoint_events,
    load_hard_failure_info,
    smooth_series,
    style_axes,
    zero_fill_pause_bins,
)


OUTPUT_PATH = Path("analysis/hard_failure_recovery_figure_compute_only.png")
LEGACY_OUTPUT_PATH = Path("analysis/hard_failure_progress_vs_time_compute_only.png")


def load_compute_utilization_series(run_dir: Path, t0: float, bin_sec: float = 10.0):
    profiling_dir = run_dir / "profiling_logs"
    if not profiling_dir.exists():
        return [], []

    # Match the original view, but only aggregate compute_* task records.
    bin_partition_values = defaultdict(list)

    for path in sorted(profiling_dir.glob("gpu_task_summary_partition*.jsonl")):
        partition_match = re.search(r"partition(\d+)", path.name)
        partition_id = int(partition_match.group(1)) if partition_match else -1

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                task_name = rec.get("task_name", "")
                if not task_name.startswith("compute_"):
                    continue
                if "gpu_util" not in rec or "start_time" not in rec:
                    continue
                rel_sec = float(rec["start_time"]) - t0
                if rel_sec < 0:
                    continue
                bin_idx = int(rel_sec // bin_sec)
                bin_partition_values[(bin_idx, partition_id)].append(float(rec["gpu_util"]))

    if not bin_partition_values:
        return [], []

    partition_medians_by_bin = defaultdict(list)
    for (bin_idx, _partition_id), values in bin_partition_values.items():
        partition_medians_by_bin[bin_idx].append(statistics.median(values))

    bin_means = {}
    for bin_idx, values in partition_medians_by_bin.items():
        bin_means[bin_idx] = statistics.mean(values)

    xs = []
    ys = []
    min_bin = min(bin_means)
    max_bin = max(bin_means)
    for bin_idx in range(min_bin, max_bin + 1):
        xs.append(((bin_idx + 0.5) * bin_sec) / 60.0)
        ys.append(bin_means.get(bin_idx, 0.0))

    return xs, ys


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(11.8, 5.0),
        gridspec_kw={"width_ratios": [2.6, 1.3]},
    )

    base_label, base_dir, base_color = BASE_RUN
    hard_label, hard_dir, hard_color = HARD_RUN

    base_rows = load_checkpoint_events(base_dir)
    hard_rows = load_checkpoint_events(hard_dir)
    hard_info = load_hard_failure_info(hard_dir)
    if hard_info is None:
        raise RuntimeError("Could not load hard-failure metadata from log.txt")

    resume_step = hard_info["resume_step"]
    recovery = estimate_recovery_overhead(hard_rows, resume_step)
    if recovery is None:
        raise RuntimeError("Could not infer recovery window from hard-failure run.")

    base_xs, base_ys, _base_t0 = build_progress_series(base_rows)
    hard_xs, hard_ys, hard_t0 = build_progress_series(hard_rows)
    hard_xs, hard_ys, pause = build_hard_series_with_pause(
        hard_rows,
        resume_step,
        recovery["recovery_overhead_sec"],
    )
    hard_util_xs, hard_util_ys = load_compute_utilization_series(hard_dir, hard_t0, bin_sec=15.0)

    base_total_min = get_total_minutes(base_rows)
    hard_total_min = get_total_minutes(hard_rows)
    target_gpu = hard_info["target_gpu"] if hard_info["target_gpu"] is not None else "?"
    fail_after = hard_info["fail_after_batches"] if hard_info["fail_after_batches"] is not None else "?"
    resume_text = resume_step if resume_step is not None else "?"

    ax_left.plot(
        base_xs,
        base_ys,
        color=base_color,
        linewidth=2.6,
        linestyle="--",
        marker="o",
        markersize=4,
        label=base_label,
    )
    ax_left.plot(
        hard_xs,
        hard_ys,
        color=hard_color,
        linewidth=2.8,
        marker="o",
        markersize=4,
        label=hard_label,
    )

    if pause is not None:
        ax_left.axvspan(
            pause["failure_x"],
            pause["pause_end_x"],
            color="#fed7d7",
            alpha=0.45,
            label="Recovery pause",
        )
        ax_left.axvline(
            pause["failure_x"],
            color=hard_color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )

        gpu_text = "GPU ?"
        if hard_info["target_gpu"] is not None:
            gpu_text = f"GPU {hard_info['target_gpu']}"
        step_text = f"step {resume_step}" if resume_step is not None else "unknown step"

        ax_left.text(
            pause["failure_x"] + 0.12,
            pause["failure_y"] + 9,
            f"{gpu_text} out @ {step_text}",
            color=hard_color,
            fontsize=10,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": hard_color, "alpha": 0.92},
        )
        ax_left.text(
            pause["pause_end_x"] + 0.18,
            pause["failure_y"] + 22,
            "Recovered on 3 GPUs\n(from 4 GPUs)",
            color="#7b341e",
            fontsize=10,
            ha="left",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#cbd5e0", "alpha": 0.92},
        )

    ax_left.scatter([base_xs[-1]], [base_ys[-1]], color=base_color, s=50, zorder=5)
    ax_left.scatter([hard_xs[-1]], [hard_ys[-1]], color=hard_color, s=50, zorder=5)
    ax_left.text(
        base_xs[-1] - 0.08,
        base_ys[-1] + 2.5,
        f"Step 200\n{base_total_min:.2f} min",
        color=base_color,
        fontsize=10,
        weight="bold",
        ha="right",
        va="bottom",
    )
    ax_left.text(
        hard_xs[-1] - 0.10,
        hard_ys[-1] + 2.5,
        f"Step 200\n{hard_total_min:.2f} min",
        color=hard_color,
        fontsize=10,
        weight="bold",
        ha="right",
        va="bottom",
    )

    ax_left.text(
        0.03,
        0.97,
        "Experiment setting:\n1 epoch, max 200 steps",
        transform=ax_left.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e0", "alpha": 0.92},
    )

    ax_left.set_title("(a) Training Progress", fontsize=13, pad=10)
    ax_left.set_xlabel("Wall-clock Time (min)", fontsize=12)
    ax_left.set_ylabel("Training Progress (Global Step)", fontsize=12)
    ax_left.set_xlim(left=0)
    ax_left.set_ylim(0, 212)
    ax_left.legend(frameon=False, fontsize=10, loc="lower right")
    style_axes(ax_left)

    if pause is not None:
        rel_util_xs = [x - pause["failure_x"] for x in hard_util_xs]
        rel_util_ys = smooth_series(hard_util_ys, window=3)
        rel_pause_end_x = pause["pause_end_x"] - pause["failure_x"]
        rel_util_ys = zero_fill_pause_bins(rel_util_xs, rel_util_ys, 0.0, rel_pause_end_x)
        zoom_lo = -1.5
        zoom_hi = max(3.5, rel_pause_end_x + 1.5)

        plot_xs = []
        plot_ys = []
        for x, y in zip(rel_util_xs, rel_util_ys):
            if zoom_lo <= x <= zoom_hi:
                plot_xs.append(x)
                plot_ys.append(y)

        ax_right.plot(
            plot_xs,
            plot_ys,
            color=hard_color,
            linewidth=2.8,
        )
        ax_right.fill_between(
            plot_xs,
            plot_ys,
            [0] * len(plot_xs),
            color="#feb2b2",
            alpha=0.20,
        )
        ax_right.axvspan(
            0,
            rel_pause_end_x,
            color="#fed7d7",
            alpha=0.50,
        )
        ax_right.axvline(
            0,
            color=hard_color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )

        ax_right.text(
            rel_pause_end_x / 2.0,
            18,
            "Recovery\npause",
            color="#7b341e",
            fontsize=10,
            ha="center",
            va="center",
            weight="bold",
        )
        ax_right.text(
            min(zoom_hi - 1.2, rel_pause_end_x + 0.55),
            70,
            "Running on\n3 GPUs",
            color="#7b341e",
            fontsize=10,
            ha="left",
            va="center",
        )

        ax_right.text(
            0.03,
            0.97,
            (
                f"GPU {target_gpu} failed at step {fail_after}\n"
                "compute_* tasks only\n"
                "4 GPUs -> 3 GPUs after restart\n"
                f"Resume from step {resume_text}"
            ),
            transform=ax_right.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e0", "alpha": 0.92},
        )

        ax_right.set_xlim(zoom_lo, zoom_hi)

    ax_right.set_title("(b) GPU Utilization (compute_* only)", fontsize=13, pad=10)
    ax_right.set_xlabel("Minutes Relative to Hard Failure", fontsize=12)
    ax_right.set_ylabel("Mean GPU Utilization (%)", fontsize=12)
    ax_right.set_ylim(0, 100)
    style_axes(ax_right)

    fig.suptitle(
        "Hard GPU Failure Recovery",
        fontsize=15,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=240, bbox_inches="tight")
    fig.savefig(LEGACY_OUTPUT_PATH, dpi=240, bbox_inches="tight")
    print(f"saved: {OUTPUT_PATH}")
    print(f"saved: {LEGACY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
