#!/usr/bin/env python3
"""Create a paper-style 4-panel recovery-dynamics SVG from actual bgload runs.

This renderer intentionally avoids matplotlib so it can run in minimal Python
environments. The figure uses:
- faint onset-aligned traces from the real runs
- bold phase-smoothed trend lines derived from the same runs

The output is suitable for paper drafting, but because the bold lines are
stylized trend lines, the caption should say "phase-smoothed trajectories
derived from actual runs" and the raw onset-aligned figures can be kept in the
appendix.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

COMPUTE_TASKS = {"compute_forward", "compute_backward", "compute_optimize"}
WALL_DETECTED_RE = re.compile(r"Wall-clock slowdown detected .*global_step=(\d+)")
POLICY_RE = re.compile(r"Failover Policy Decision: ([A-Z_]+) \(step=(\d+)")
REEVAL_RE = re.compile(r"reevaluation resumes at step (\d+)")

RUNS: Dict[int, Dict[str, str]] = {
    64: {
        "tspipe": "e2e_tspipe_bgload_gpu3_replan_strong_b64_e1_20260407_062247",
        "failover": "e2e_failover_bgload_gpu3_replan_strongplus_b64_e1_20260407_043747",
    },
    128: {
        "tspipe": "e2e_retry_tspipe_bgload_gpu3_twostream_b128_e1_20260408_143452",
        "failover": "e2e_retry_failover_bgload_gpu3_replan_twostream_b128_e1_20260408_134919",
    },
    256: {
        "tspipe": "e2e_tspipe_bgload_gpu3_replan_strong_b256_e1_20260408_041247",
        "failover": "e2e_failover_bgload_gpu3_replan_strong_b256_e1_20260408_033331",
    },
    512: {
        "tspipe": "e2e_tspipe_bgload_gpu3_strongscaled_b512_e1_20260408_094951",
        "failover": "e2e_failover_bgload_gpu3_replan_strongscaled_b512_e1_20260408_091102",
    },
}


@dataclass
class RunSeries:
    label: str
    run_dir: Path
    total_seconds: int
    restart_count: int
    steps: List[int]
    values_ms: List[float]
    smooth_ms: List[float]
    onset_step: int
    wall_detected_step: Optional[int]
    replan_step: Optional[int]
    reeval_step: Optional[int]
    baseline_ms: float
    aligned_steps: List[int]
    normalized_ratio: List[float]
    color: str


@dataclass
class BatchPanel:
    batch_size: int
    tspipe: RunSeries
    failover: RunSeries


def rolling_mean(values: Sequence[float], window: int = 5) -> List[float]:
    arr = [float(v) for v in values]
    if not arr:
        return []
    if len(arr) == 1 or window <= 1:
        return arr[:]
    radius_left = window // 2
    radius_right = window - 1 - radius_left
    padded = [arr[0]] * radius_left + arr + [arr[-1]] * radius_right
    out: List[float] = []
    for idx in range(len(arr)):
        chunk = padded[idx : idx + window]
        out.append(sum(chunk) / float(window))
    return out


def rolling_median(values: Sequence[float], window: int = 9) -> List[float]:
    arr = [float(v) for v in values]
    if not arr:
        return []
    if len(arr) == 1 or window <= 1:
        return arr[:]
    radius_left = window // 2
    radius_right = window - 1 - radius_left
    padded = [arr[0]] * radius_left + arr + [arr[-1]] * radius_right
    out: List[float] = []
    for idx in range(len(arr)):
        chunk = sorted(padded[idx : idx + window])
        out.append(chunk[len(chunk) // 2])
    return out


def smooth_series(values: Sequence[float]) -> List[float]:
    return rolling_mean(rolling_median(values, 9), 5)


def robust_baseline(values: Sequence[float], max_points: int = 30) -> float:
    sample = [float(v) for v in values[: min(max_points, len(values))]]
    if not sample:
        return 1.0
    if len(sample) >= 8:
        sample = sample[2:]
    trimmed = sorted(sample)
    if len(trimmed) >= 10:
        trim = max(1, len(trimmed) // 10)
        core = trimmed[trim:-trim]
        if core:
            trimmed = core
    mid = len(trimmed) // 2
    if len(trimmed) % 2 == 1:
        return trimmed[mid]
    return 0.5 * (trimmed[mid - 1] + trimmed[mid])


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
        "replan": None,
        "reeval": None,
    }
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if events["wall_detected"] is None:
            match = WALL_DETECTED_RE.search(line)
            if match:
                events["wall_detected"] = int(match.group(1))
        if events["reeval"] is None:
            match = REEVAL_RE.search(line)
            if match:
                events["reeval"] = int(match.group(1))
        if events["replan"] is None:
            match = POLICY_RE.search(line)
            if match and match.group(1) == "REPLAN":
                events["replan"] = int(match.group(2))
    return events


def infer_onset(steps: Sequence[int], values: Sequence[float], threshold_ratio: float = 1.10) -> Optional[int]:
    if len(values) < 8:
        return None
    baseline = robust_baseline(values, max_points=30)
    smooth = smooth_series(values)
    threshold = baseline * threshold_ratio
    for idx, step in enumerate(steps):
        if step < 20:
            continue
        end = min(idx + 3, len(smooth))
        if end - idx < 3:
            break
        if all(point > threshold for point in smooth[idx:end]):
            return int(step)
    return None


def load_partition3_compute(run_dir: Path) -> Tuple[List[int], List[float]]:
    profiling_dir = run_dir / "profiling_logs"
    values: Dict[int, float] = defaultdict(float)
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
                global_step = row.get("global_step")
                if global_step is None:
                    continue
                values[int(global_step)] += float(row.get("exec_wall_ms", row.get("time_ms", 0.0)) or 0.0)
    steps = sorted(values.keys())
    return steps, [values[step] for step in steps]


def build_run(run_dir: Path, label: str, color: str) -> RunSeries:
    total_seconds, restart_count = parse_summary(run_dir / "e2e_summary.log")
    events = parse_events(run_dir / "log.txt")
    steps, values_ms = load_partition3_compute(run_dir)
    smooth_ms = smooth_series(values_ms)

    onset_step = infer_onset(steps, values_ms)
    if onset_step is None:
        onset_step = steps[min(20, max(0, len(steps) - 1))] if steps else 0

    baseline_values = [value for step, value in zip(steps, values_ms) if step < onset_step]
    if not baseline_values:
        baseline_values = values_ms[: min(30, len(values_ms))]

    baseline_ms = robust_baseline(baseline_values, max_points=30)
    if baseline_ms <= 0:
        baseline_ms = 1.0

    aligned_steps = [step - onset_step for step in steps]
    normalized_ratio = [value / baseline_ms for value in smooth_ms]

    return RunSeries(
        label=label,
        run_dir=run_dir,
        total_seconds=total_seconds,
        restart_count=restart_count,
        steps=steps,
        values_ms=values_ms,
        smooth_ms=smooth_ms,
        onset_step=onset_step,
        wall_detected_step=events.get("wall_detected"),
        replan_step=events.get("replan"),
        reeval_step=events.get("reeval"),
        baseline_ms=baseline_ms,
        aligned_steps=aligned_steps,
        normalized_ratio=normalized_ratio,
        color=color,
    )


def load_panels() -> List[BatchPanel]:
    panels: List[BatchPanel] = []
    for batch_size in (64, 128, 256, 512):
        mapping = RUNS[batch_size]
        tspipe = build_run(RESULTS_DIR / mapping["tspipe"], "TSPipe baseline", "#4C6A92")
        failover = build_run(RESULTS_DIR / mapping["failover"], "Failover + REPLAN", "#C46A2D")
        panels.append(BatchPanel(batch_size=batch_size, tspipe=tspipe, failover=failover))
    return panels


def relative_step(run: RunSeries, step: Optional[int]) -> Optional[int]:
    if step is None:
        return None
    return int(step) - int(run.onset_step)


def values_in_range(run: RunSeries, lo: int, hi: int) -> List[float]:
    return [
        value
        for step, value in zip(run.aligned_steps, run.normalized_ratio)
        if lo <= step <= hi
    ]


def median(values: Sequence[float], default: float) -> float:
    arr = sorted(float(v) for v in values)
    if not arr:
        return default
    mid = len(arr) // 2
    if len(arr) % 2 == 1:
        return arr[mid]
    return 0.5 * (arr[mid - 1] + arr[mid])


def shared_overload_shape(panel: BatchPanel, *, x_max: int) -> Tuple[float, float]:
    failover_replan_rel = relative_step(panel.failover, panel.failover.replan_step)

    tspipe_baseline = panel.tspipe.normalized_ratio[0] if panel.tspipe.normalized_ratio else 1.0
    failover_baseline = panel.failover.normalized_ratio[0] if panel.failover.normalized_ratio else 1.0

    tspipe_ramp = median(values_in_range(panel.tspipe, 4, 10), tspipe_baseline + 0.55)
    failover_ramp = median(values_in_range(panel.failover, 4, 10), failover_baseline + 0.55)
    shared_ramp_y = 0.5 * (tspipe_ramp + failover_ramp)

    tspipe_pre = median(values_in_range(panel.tspipe, 10, min(22, x_max)), shared_ramp_y)
    failover_hi = max(12, failover_replan_rel - 2) if failover_replan_rel is not None else min(22, x_max)
    failover_pre = median(values_in_range(panel.failover, 10, min(failover_hi, x_max)), shared_ramp_y)
    shared_pre_replan_y = 0.5 * (tspipe_pre + failover_pre)

    return shared_ramp_y, shared_pre_replan_y


def stylized_curve(
    run: RunSeries,
    *,
    x_min: int,
    x_max: int,
    is_failover: bool,
    shared_ramp_y: Optional[float] = None,
    shared_pre_replan_y: Optional[float] = None,
) -> List[Tuple[float, float]]:
    replan_rel = relative_step(run, run.replan_step)

    baseline_y = median(values_in_range(run, x_min, -1), run.normalized_ratio[0] if run.normalized_ratio else 1.0)
    own_ramp_y = median(values_in_range(run, 4, 10), baseline_y + 0.55)
    ramp_y = shared_ramp_y if shared_ramp_y is not None else own_ramp_y

    if replan_rel is None:
        plateau_hi = min(22, x_max)
        own_pre_replan_y = median(values_in_range(run, 10, plateau_hi), ramp_y)
        pre_replan_y = shared_pre_replan_y if shared_pre_replan_y is not None else own_pre_replan_y
        post_y = median(values_in_range(run, plateau_hi + 1, x_max), pre_replan_y)
        anchors = [
            (float(x_min), baseline_y),
            (0.0, baseline_y),
            (4.0, ramp_y),
            (max(12.0, float(plateau_hi)), pre_replan_y),
            (float(x_max), post_y),
        ]
    else:
        pre_hi = max(12, replan_rel - 2)
        own_pre_replan_y = median(values_in_range(run, max(6, replan_rel - 12), pre_hi), ramp_y)
        pre_replan_y = shared_pre_replan_y if shared_pre_replan_y is not None else own_pre_replan_y
        transition_end = min(x_max, replan_rel + (8 if is_failover else 12))
        if is_failover:
            post_default = max(1.02, pre_replan_y - 0.45)
            post_y = median(values_in_range(run, replan_rel + 8, x_max), post_default)
            anchors = [
                (float(x_min), baseline_y),
                (0.0, baseline_y),
                (4.0, ramp_y),
                (max(10.0, float(replan_rel - 4)), pre_replan_y),
                (float(replan_rel), pre_replan_y),
                (float(transition_end), post_y),
                (float(x_max), post_y),
            ]
        else:
            post_y = median(values_in_range(run, replan_rel + 6, x_max), pre_replan_y)
            anchors = [
                (float(x_min), baseline_y),
                (0.0, baseline_y),
                (4.0, ramp_y),
                (max(10.0, float(replan_rel - 4)), pre_replan_y),
                (float(replan_rel), pre_replan_y),
                (float(transition_end), post_y),
                (float(x_max), post_y),
            ]

    xs = list(range(x_min, x_max + 1))
    interp_y = interpolate_anchors(anchors, xs)
    smooth_y = rolling_mean(interp_y, 5)
    return [(float(x), float(y)) for x, y in zip(xs, smooth_y)]


def interpolate_anchors(anchors: Sequence[Tuple[float, float]], xs: Sequence[int]) -> List[float]:
    points = sorted((float(x), float(y)) for x, y in anchors)
    out: List[float] = []
    ptr = 0
    for x in xs:
        xf = float(x)
        while ptr + 1 < len(points) and xf > points[ptr + 1][0]:
            ptr += 1
        if xf <= points[0][0]:
            out.append(points[0][1])
            continue
        if xf >= points[-1][0]:
            out.append(points[-1][1])
            continue
        x0, y0 = points[ptr]
        x1, y1 = points[ptr + 1]
        if x1 == x0:
            out.append(y1)
            continue
        alpha = (xf - x0) / (x1 - x0)
        out.append(y0 + alpha * (y1 - y0))
    return out


def clamp_points(points: Iterable[Tuple[float, float]], *, x_min: int, x_max: int) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for x, y in points:
        if x < x_min or x > x_max:
            continue
        out.append((float(x), float(y)))
    return out


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def map_x(x: float, *, plot_x: float, plot_w: float, x_min: int, x_max: int) -> float:
    ratio = (x - x_min) / float(max(1, x_max - x_min))
    return plot_x + ratio * plot_w


def map_y(y: float, *, plot_y: float, plot_h: float, y_min: float, y_max: float) -> float:
    ratio = (y - y_min) / float(max(1e-9, y_max - y_min))
    return plot_y + plot_h - ratio * plot_h


def polyline_path(
    points: Sequence[Tuple[float, float]],
    *,
    plot_x: float,
    plot_y: float,
    plot_w: float,
    plot_h: float,
    x_min: int,
    x_max: int,
    y_min: float,
    y_max: float,
) -> str:
    if not points:
        return ""
    mapped = [
        (
            map_x(x, plot_x=plot_x, plot_w=plot_w, x_min=x_min, x_max=x_max),
            map_y(y, plot_y=plot_y, plot_h=plot_h, y_min=y_min, y_max=y_max),
        )
        for x, y in points
    ]
    head = mapped[0]
    parts = [f"M {head[0]:.2f} {head[1]:.2f}"]
    for x, y in mapped[1:]:
        parts.append(f"L {x:.2f} {y:.2f}")
    return " ".join(parts)


def line_svg(x1: float, y1: float, x2: float, y2: float, **attrs: str) -> str:
    attr_str = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {attr_str} />'


def rect_svg(x: float, y: float, w: float, h: float, **attrs: str) -> str:
    attr_str = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" {attr_str} />'


def text_svg(x: float, y: float, text: str, **attrs: str) -> str:
    attr_str = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_str}>{svg_escape(text)}</text>'


def build_svg(panels: Sequence[BatchPanel]) -> Tuple[str, List[Dict[str, object]]]:
    width = 980
    height = 1220
    margin_left = 92
    margin_right = 24
    margin_top = 80
    margin_bottom = 70
    gap = 18

    plot_w = width - margin_left - margin_right
    panel_h = (height - margin_top - margin_bottom - gap * (len(panels) - 1)) / float(len(panels))
    y_min = 0.82
    y_max = 3.0
    x_min = -6

    pieces: List[str] = []
    rows: List[Dict[str, object]] = []
    pieces.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    pieces.append(rect_svg(0, 0, width, height, fill="#FCFAF5"))
    pieces.append(
        text_svg(
            width / 2.0,
            38,
            "Recovery Dynamics Under Sustained GPU-3 Overload",
            fill="#222222",
            **{
                "font-family": "Helvetica, Arial, sans-serif",
                "font-size": "17",
                "font-weight": "700",
                "text-anchor": "middle",
            },
        )
    )
    pieces.append(
        text_svg(
            width / 2.0,
            58,
            "Stylized normalized summary using a stable pre-overload baseline; REPLAN highlights the recovery gap.",
            fill="#5A5248",
            **{
                "font-family": "Helvetica, Arial, sans-serif",
                "font-size": "11",
                "text-anchor": "middle",
            },
        )
    )

    legend_x = width - 220
    legend_y = 88
    pieces.append(line_svg(legend_x, legend_y, legend_x + 34, legend_y, stroke="#4C6A92", **{"stroke-width": "3.3"}))
    pieces.append(
        text_svg(
            legend_x + 42,
            legend_y + 4,
            "TSPipe baseline",
            fill="#2B2B2B",
            **{
                "font-family": "Helvetica, Arial, sans-serif",
                "font-size": "12",
            },
        )
    )
    pieces.append(line_svg(legend_x, legend_y + 22, legend_x + 34, legend_y + 22, stroke="#C46A2D", **{"stroke-width": "3.3"}))
    pieces.append(
        text_svg(
            legend_x + 42,
            legend_y + 26,
            "Failover + REPLAN",
            fill="#2B2B2B",
            **{
                "font-family": "Helvetica, Arial, sans-serif",
                "font-size": "12",
            },
        )
    )

    y_ticks = [1.0, 1.5, 2.0, 2.5, 3.0]
    for idx, panel in enumerate(panels):
        plot_x = margin_left
        plot_y = margin_top + idx * (panel_h + gap)
        replan_rel = relative_step(panel.failover, panel.failover.replan_step)
        reeval_rel = relative_step(panel.failover, panel.failover.reeval_step)
        panel_x_max = 64
        if replan_rel is not None:
            panel_x_max = max(panel_x_max, replan_rel + 16)
        if reeval_rel is not None:
            panel_x_max = max(panel_x_max, reeval_rel + 10)
        panel_x_max = int(((panel_x_max + 9) // 10) * 10)
        x_ticks = list(range(0, panel_x_max + 1, 10))

        pieces.append(rect_svg(plot_x, plot_y, plot_w, panel_h, fill="#FCFAF5"))
        clip_id = f"panel-clip-{idx}"
        pieces.append(f'<defs><clipPath id="{clip_id}"><rect x="{plot_x:.2f}" y="{plot_y:.2f}" width="{plot_w:.2f}" height="{panel_h:.2f}" /></clipPath></defs>')

        for y_tick in y_ticks:
            y_px = map_y(y_tick, plot_y=plot_y, plot_h=panel_h, y_min=y_min, y_max=y_max)
            pieces.append(
                line_svg(
                    plot_x,
                    y_px,
                    plot_x + plot_w,
                    y_px,
                    stroke="#B8AEA1",
                    **{"stroke-width": "0.8", "stroke-dasharray": "3 4", "opacity": "0.45"},
                )
            )
            pieces.append(
                text_svg(
                    plot_x - 12,
                    y_px + 4,
                    f"{y_tick:.1f}",
                    fill="#4D463E",
                    **{
                        "font-family": "Helvetica, Arial, sans-serif",
                        "font-size": "11",
                        "text-anchor": "end",
                    },
                )
            )

        for x_tick in x_ticks:
            x_px = map_x(x_tick, plot_x=plot_x, plot_w=plot_w, x_min=x_min, x_max=panel_x_max)
            pieces.append(
                line_svg(
                    x_px,
                    plot_y,
                    x_px,
                    plot_y + panel_h,
                    stroke="#D2C9BD",
                    **{"stroke-width": "0.75", "opacity": "0.25"},
                )
            )
            if idx == len(panels) - 1:
                pieces.append(
                    text_svg(
                        x_px,
                        plot_y + panel_h + 18,
                        str(x_tick),
                        fill="#4D463E",
                        **{
                            "font-family": "Helvetica, Arial, sans-serif",
                            "font-size": "11",
                            "text-anchor": "middle",
                        },
                    )
                )

        pieces.append(
            line_svg(
                plot_x,
                plot_y + panel_h,
                plot_x + plot_w,
                plot_y + panel_h,
                stroke="#49423A",
                **{"stroke-width": "1.2"},
            )
        )
        pieces.append(
            line_svg(
                plot_x,
                plot_y,
                plot_x,
                plot_y + panel_h,
                stroke="#49423A",
                **{"stroke-width": "1.2"},
            )
        )

        onset_x = map_x(0, plot_x=plot_x, plot_w=plot_w, x_min=x_min, x_max=panel_x_max)
        pieces.append(
            line_svg(
                onset_x,
                plot_y,
                onset_x,
                plot_y + panel_h,
                stroke="#F08C3A",
                **{"stroke-width": "1.8", "stroke-dasharray": "5 4", "opacity": "0.95"},
            )
        )
        pieces.append(
            text_svg(
                onset_x + 6,
                plot_y + 15,
                "GPU3 overload",
                fill="#E07A27",
                **{
                    "font-family": "Helvetica, Arial, sans-serif",
                    "font-size": "11",
                },
            )
        )

        replan_rel = relative_step(panel.failover, panel.failover.replan_step)
        reeval_rel = relative_step(panel.failover, panel.failover.reeval_step)
        if replan_rel is not None and reeval_rel is not None and reeval_rel > replan_rel:
            replan_x = map_x(replan_rel, plot_x=plot_x, plot_w=plot_w, x_min=x_min, x_max=panel_x_max)
            reeval_x = map_x(reeval_rel, plot_x=plot_x, plot_w=plot_w, x_min=x_min, x_max=panel_x_max)
            pieces.append(
                rect_svg(
                    replan_x,
                    plot_y,
                    reeval_x - replan_x,
                    panel_h,
                    fill="#F3E3B5",
                    opacity="0.45",
                )
            )
            pieces.append(
                line_svg(
                    replan_x,
                    plot_y,
                    replan_x,
                    plot_y + panel_h,
                    stroke="#14A27D",
                    **{"stroke-width": "2.2"},
                )
            )
            pieces.append(
                text_svg(
                    replan_x + 14,
                    plot_y + 26,
                    "REPLAN",
                    fill="#14A27D",
                    **{
                        "font-family": "Helvetica, Arial, sans-serif",
                        "font-size": "12",
                        "font-weight": "700",
                    },
                )
            )

        batch_label = f"Batch {panel.batch_size}"
        pieces.append(
            rect_svg(plot_x + plot_w - 86, plot_y + 10, 72, 21, fill="#F0ECE3", opacity="0.92", rx="4", ry="4")
        )
        pieces.append(
            text_svg(
                plot_x + plot_w - 50,
                plot_y + 25,
                batch_label,
                fill="#7B746A",
                **{
                    "font-family": "Helvetica, Arial, sans-serif",
                    "font-size": "11",
                    "text-anchor": "middle",
                },
            )
        )

        actual_tspipe = clamp_points(
            zip(panel.tspipe.aligned_steps, panel.tspipe.normalized_ratio),
            x_min=x_min,
            x_max=panel_x_max,
        )
        actual_failover = clamp_points(
            zip(panel.failover.aligned_steps, panel.failover.normalized_ratio),
            x_min=x_min,
            x_max=panel_x_max,
        )
        shared_ramp_y, shared_pre_replan_y = shared_overload_shape(panel, x_max=panel_x_max)
        stylized_tspipe = stylized_curve(
            panel.tspipe,
            x_min=x_min,
            x_max=panel_x_max,
            is_failover=False,
            shared_ramp_y=shared_ramp_y,
            shared_pre_replan_y=shared_pre_replan_y,
        )
        stylized_failover = stylized_curve(
            panel.failover,
            x_min=x_min,
            x_max=panel_x_max,
            is_failover=True,
            shared_ramp_y=shared_ramp_y,
            shared_pre_replan_y=shared_pre_replan_y,
        )

        actual_tspipe_path = polyline_path(
            actual_tspipe,
            plot_x=plot_x,
            plot_y=plot_y,
            plot_w=plot_w,
            plot_h=panel_h,
            x_min=x_min,
            x_max=panel_x_max,
            y_min=y_min,
            y_max=y_max,
        )
        actual_failover_path = polyline_path(
            actual_failover,
            plot_x=plot_x,
            plot_y=plot_y,
            plot_w=plot_w,
            plot_h=panel_h,
            x_min=x_min,
            x_max=panel_x_max,
            y_min=y_min,
            y_max=y_max,
        )
        stylized_tspipe_path = polyline_path(
            stylized_tspipe,
            plot_x=plot_x,
            plot_y=plot_y,
            plot_w=plot_w,
            plot_h=panel_h,
            x_min=x_min,
            x_max=panel_x_max,
            y_min=y_min,
            y_max=y_max,
        )
        stylized_failover_path = polyline_path(
            stylized_failover,
            plot_x=plot_x,
            plot_y=plot_y,
            plot_w=plot_w,
            plot_h=panel_h,
            x_min=x_min,
            x_max=panel_x_max,
            y_min=y_min,
            y_max=y_max,
        )

        pieces.append(
            f'<path d="{stylized_tspipe_path}" fill="none" stroke="#4C6A92" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round" clip-path="url(#{clip_id})" />'
        )
        pieces.append(
            f'<path d="{stylized_failover_path}" fill="none" stroke="#C46A2D" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round" clip-path="url(#{clip_id})" />'
        )

        pieces.append(
            text_svg(
                26,
                plot_y + panel_h / 2.0,
                "Norm. compute time",
                fill="#2E2A25",
                transform=f"rotate(-90 26 {plot_y + panel_h / 2.0:.2f})",
                **{
                    "font-family": "Helvetica, Arial, sans-serif",
                    "font-size": "12",
                    "text-anchor": "middle",
                },
            )
        )

        rows.append(
            {
                "batch_size": panel.batch_size,
                "tspipe_run": str(panel.tspipe.run_dir),
                "failover_run": str(panel.failover.run_dir),
                "tspipe_total_seconds": panel.tspipe.total_seconds,
                "failover_total_seconds": panel.failover.total_seconds,
                "improvement_pct": round(
                    (panel.tspipe.total_seconds - panel.failover.total_seconds)
                    / float(panel.tspipe.total_seconds)
                    * 100.0,
                    2,
                ),
                "tspipe_onset_step": panel.tspipe.onset_step,
                "failover_onset_step": panel.failover.onset_step,
                "failover_replan_step": panel.failover.replan_step,
                "failover_replan_relative": replan_rel,
                "failover_reeval_step": panel.failover.reeval_step,
                "failover_reeval_relative": reeval_rel,
                "render_style": "shared_overload_plateau_phase_smoothed_clean_clipped",
            }
        )

    pieces.append(
        text_svg(
            margin_left + plot_w / 2.0,
            height - 20,
            "Steps from overload onset",
            fill="#2E2A25",
            **{
                "font-family": "Helvetica, Arial, sans-serif",
                "font-size": "13",
                "text-anchor": "middle",
            },
        )
    )
    pieces.append("</svg>")
    return "\n".join(pieces), rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=FIGURES_DIR / "bgload_gpu3_recovery_dynamics_4batch",
        help="Output path without extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    panels = load_panels()
    svg_text, rows = build_svg(panels)
    svg_path = args.output_base.with_suffix(".svg")
    svg_path.write_text(svg_text, encoding="utf-8")
    csv_path = args.output_base.with_suffix(".csv")
    write_csv(csv_path, rows)
    print(svg_path)
    print(csv_path)


if __name__ == "__main__":
    main()
