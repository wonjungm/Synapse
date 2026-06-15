import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BASE_RUN = ("Failover Armed, No Failure", Path("results/paper_hard_base"), "#2b6cb0")
HARD_RUN = ("Hard Failure + Recovery", Path("results/paper_hard_fig_gpu3"), "#c53030")
OUTPUT_PATH = Path("analysis/hard_failure_recovery_figure.png")
LEGACY_OUTPUT_PATH = Path("analysis/hard_failure_progress_vs_time.png")


def load_checkpoint_events(run_dir: Path):
    event_path = run_dir / "exp0_checkpoint_save_events.jsonl"
    if not event_path.exists():
        raise FileNotFoundError(f"Missing checkpoint events: {event_path}")

    rows = []
    with event_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append(
                {
                    "timestamp_sec": float(rec["timestamp_sec"]),
                    "batch_count": int(rec["batch_count"]),
                    "pre_step_time_ms": float(rec.get("pre_step_time_ms", 0.0)),
                }
            )

    rows.sort(key=lambda row: (row["timestamp_sec"], row["batch_count"]))
    if not rows:
        raise ValueError(f"No checkpoint events found in {event_path}")
    return rows


def estimate_start_timestamp(rows):
    first = rows[0]
    pre_step_sec = max(first["pre_step_time_ms"], 1.0) / 1000.0
    estimated_start = first["timestamp_sec"] - (first["batch_count"] * pre_step_sec)
    return min(estimated_start, first["timestamp_sec"])


def build_progress_series(rows):
    t0 = estimate_start_timestamp(rows)
    xs = [0.0]
    ys = [0]

    for row in rows:
        xs.append((row["timestamp_sec"] - t0) / 60.0)
        ys.append(row["batch_count"])

    return xs, ys, t0


def get_intervals(rows):
    intervals = []
    for prev_row, next_row in zip(rows, rows[1:]):
        intervals.append(
            {
                "start_step": prev_row["batch_count"],
                "end_step": next_row["batch_count"],
                "delta_step": next_row["batch_count"] - prev_row["batch_count"],
                "delta_time_sec": next_row["timestamp_sec"] - prev_row["timestamp_sec"],
            }
        )
    return intervals


def get_row_by_step(rows, step):
    for row in rows:
        if row["batch_count"] == step:
            return row
    return None


def load_hard_failure_info(run_dir: Path):
    log_path = run_dir / "log.txt"
    if not log_path.exists():
        return None

    target_gpu = None
    fail_after = None
    resume_step = None

    target_pat = re.compile(r"--target-fail-gpu=(\-?\d+)")
    fail_pat = re.compile(r"--fail-after-batches=(\d+)")
    resume_pat = re.compile(r"Resuming training from step \[(\d+)\]")

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if target_gpu is None:
                m = target_pat.search(line)
                if m:
                    target_gpu = int(m.group(1))
            if fail_after is None:
                m = fail_pat.search(line)
                if m:
                    fail_after = int(m.group(1))
            if resume_step is None:
                m = resume_pat.search(line)
                if m:
                    resume_step = int(m.group(1))

    return {
        "target_gpu": target_gpu,
        "fail_after_batches": fail_after,
        "resume_step": resume_step,
    }


def estimate_recovery_overhead(rows, resume_step):
    if resume_step is None:
        return None

    intervals = get_intervals(rows)
    recovery_interval = None
    post_recovery_times = []

    for interval in intervals:
        if interval["start_step"] == resume_step:
            recovery_interval = interval
            continue
        if interval["start_step"] >= resume_step + 20 and interval["delta_step"] == 20:
            post_recovery_times.append(interval["delta_time_sec"])

    if recovery_interval is None:
        return None

    if post_recovery_times:
        nominal_20step_sec = statistics.median(post_recovery_times)
    else:
        comparable = [i["delta_time_sec"] for i in intervals if i["delta_step"] == recovery_interval["delta_step"]]
        nominal_20step_sec = statistics.median(comparable)

    recovery_overhead_sec = max(recovery_interval["delta_time_sec"] - nominal_20step_sec, 0.0)
    return {
        "nominal_20step_sec": nominal_20step_sec,
        "actual_20step_sec": recovery_interval["delta_time_sec"],
        "recovery_overhead_sec": recovery_overhead_sec,
        "window_start_step": recovery_interval["start_step"],
        "window_end_step": recovery_interval["end_step"],
    }


def build_hard_series_with_pause(rows, resume_step, recovery_overhead_sec):
    t0 = estimate_start_timestamp(rows)
    xs = [0.0]
    ys = [0]
    pause = None

    for idx, row in enumerate(rows):
        x = (row["timestamp_sec"] - t0) / 60.0
        y = row["batch_count"]
        xs.append(x)
        ys.append(y)

        if row["batch_count"] != resume_step:
            continue

        if idx + 1 >= len(rows):
            continue

        pause_end_x = x + (recovery_overhead_sec / 60.0)
        xs.append(pause_end_x)
        ys.append(y)
        pause = {
            "failure_x": x,
            "failure_y": y,
            "pause_end_x": pause_end_x,
        }

    return xs, ys, pause


def get_total_minutes(rows):
    t0 = estimate_start_timestamp(rows)
    return (rows[-1]["timestamp_sec"] - t0) / 60.0


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22, linewidth=0.8)


def load_utilization_series(run_dir: Path, t0: float, bin_sec: float = 10.0):
    profiling_dir = run_dir / "profiling_logs"
    if not profiling_dir.exists():
        return [], []

    # Aggregate per-partition medians inside each time bin, then average across partitions.
    bin_partition_values = defaultdict(list)

    for path in sorted(profiling_dir.glob("gpu_task_summary_partition*.jsonl")):
        partition_match = re.search(r"partition(\d+)", path.name)
        partition_id = int(partition_match.group(1)) if partition_match else -1

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
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


def smooth_series(values, window: int = 3):
    if window <= 1 or len(values) < 3:
        return values[:]

    smoothed = []
    half = window // 2
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        smoothed.append(statistics.mean(values[lo:hi]))
    return smoothed


def zero_fill_pause_bins(xs, ys, pause_start_x, pause_end_x):
    adjusted = []
    for x, y in zip(xs, ys):
        if pause_start_x <= x <= pause_end_x:
            adjusted.append(0.0)
        else:
            adjusted.append(y)
    return adjusted


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

    base_xs, base_ys, base_t0 = build_progress_series(base_rows)
    hard_xs, hard_ys, hard_t0 = build_progress_series(hard_rows)
    hard_xs, hard_ys, pause = build_hard_series_with_pause(
        hard_rows,
        resume_step,
        recovery["recovery_overhead_sec"],
    )
    hard_util_xs, hard_util_ys = load_utilization_series(hard_dir, hard_t0, bin_sec=15.0)

    base_total_min = get_total_minutes(base_rows)
    hard_total_min = get_total_minutes(hard_rows)
    nominal_sec = recovery["nominal_20step_sec"]
    overhead_sec = recovery["recovery_overhead_sec"]
    actual_sec = recovery["actual_20step_sec"]
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
            f"GPU {target_gpu} failed at step {fail_after}\n4 GPUs -> 3 GPUs after restart\nResume from step {resume_text}",
            transform=ax_right.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e0", "alpha": 0.92},
        )

        ax_right.set_xlim(zoom_lo, zoom_hi)

    ax_right.set_title("(b) GPU Utilization", fontsize=13, pad=10)
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
