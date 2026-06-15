#!/usr/bin/env python3
"""Plot per-GPU step-time timelines with slowdown/policy overlays.

This script reads:
  - `<experiment_dir>/profiling_logs/gpu_task_summary_partition*.jsonl`
  - `<experiment_dir>/log.txt`

and creates a line chart where:
  - X axis: step or elapsed time
  - Y axis: per-GPU step wall-clock span (ms)
  - markers: slowdown injection / end / policy decision points

Example:
  python scripts/plot_gpu_step_timeline.py results/e2e_failover_0329 results/e2e_failover_0330
  python scripts/plot_gpu_step_timeline.py --search-root results --x-axis time --gpu-ids 0 1 2 3
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


POLICY_COLORS = {
    "NORMAL": "#ffffff",
    "KEEP": "#f3d46b",
    "REPLAN": "#7bd389",
    "DEGRADE": "#8fb8ff",
}

POLICY_MARKERS = {
    "KEEP": "o",
    "REPLAN": "s",
    "DEGRADE": "D",
}

GPU_COLORS = [
    "#2b6cb0",
    "#d97706",
    "#2f855a",
    "#c53030",
    "#6b46c1",
    "#0f766e",
    "#9f1239",
    "#4a5568",
]

GPU_LINESTYLES = {
    0: "-",
    1: "--",
    2: "-.",
    3: ":",
}

SLOWDOWN_STYLES = {
    "start": {
        "color": "#b91c1c",
        "linestyle": "--",
        "linewidth": 2.0,
        "marker": "^",
    },
    "end": {
        "color": "#0f766e",
        "linestyle": "-.",
        "linewidth": 2.2,
        "marker": "v",
    },
}

FIRST_LINE_PATTERNS = {
    "batch_size": re.compile(r"\bbatch_size=(\d+)"),
    "note": re.compile(r"\bnote='([^']*)'"),
    "inject_slowdown_gpu": re.compile(r"\binject_slowdown_gpu=([^,\s)]+)"),
    "slowdown_factor": re.compile(r"\bslowdown_factor=([^,\s)]+)"),
    "slowdown_start": re.compile(r"\bslowdown_start=([^,\s)]+)"),
    "slowdown_end": re.compile(r"\bslowdown_end=([^,\s)]+)"),
    "failover_inject_scenario": re.compile(r"\bfailover_inject_scenario='([^']*)'"),
}

SYNTHETIC_INJECTION_RE = re.compile(
    r"Injected synthetic slowdown scenario\s+(?P<scenario>\S+)\s+"
    r"\(phase=(?P<phase>[^)]+)\)\s+at step\s+(?P<step>\d+):\s+"
    r"gpu=(?P<gpu>\d+),\s+slowdown=(?P<slowdown>[0-9.]+)x"
)

REAL_INJECTION_RE = re.compile(
    r"REAL slowdown injected\s+\(scenario=(?P<scenario>[^,]*),.*?"
    r"gpu=(?P<gpu>\d+),.*?target≈(?P<slowdown>[0-9.]+)x,\s+"
    r"step=(?P<step>\d+),"
)

POLICY_DECISION_RE = re.compile(
    r"Failover Policy Decision:\s+(?P<policy>[A-Z_]+)\s+\((?P<details>.+)\)"
)

STEP_IN_DETAILS_RE = re.compile(r"\bstep=(\d+)")
GPU_IN_DETAILS_RE = re.compile(r"\bgpu=(\d+)")
SLOWDOWN_IN_DETAILS_RE = re.compile(r"\bslowdown=([0-9.]+)x")
WALL_RATIO_IN_DETAILS_RE = re.compile(r"\bwall_ratio=([0-9.]+)x")
LOCAL_RATIO_IN_DETAILS_RE = re.compile(r"\blocalized_ratio=([0-9.]+)x")
RESUME_STEP_RE = re.compile(r"Resuming training from step \[(\d+)\]")


@dataclass
class StepPoint:
    run_index: int
    raw_step: int
    occurrence_index: int
    plot_step: int
    step_time_ms: float
    sum_time_ms: float
    start_time_sec: Optional[float]


@dataclass
class InjectionEvent:
    run_index: int
    step: int
    gpu_id: Optional[int]
    label: str
    event_type: str = "start"
    slowdown_ratio: Optional[float] = None


@dataclass
class PolicyEvent:
    run_index: int
    step: int
    policy: str
    gpu_id: Optional[int]
    slowdown_ratio: Optional[float] = None


@dataclass
class ExperimentData:
    name: str
    path: Path
    batch_size: Optional[int]
    scenario: str
    step_points_by_gpu: Dict[int, List[StepPoint]]
    run_step_time_sec: Dict[Tuple[int, int], float]
    run_step_plot_step: Dict[Tuple[int, int], int]
    run_resume_steps: Dict[int, int]
    policies: List[PolicyEvent]
    injections: List[InjectionEvent]


def parse_optional_int(raw: Optional[str]) -> Optional[int]:
    if raw is None or raw in {"None", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def parse_optional_float(raw: Optional[str]) -> Optional[float]:
    if raw is None or raw in {"None", ""}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    q = min(max(q, 0.0), 1.0)
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * q)
    return float(sorted_values[index])


def parse_experiment_meta_line(first_line: str) -> Dict[str, Optional[object]]:
    meta: Dict[str, Optional[object]] = {}
    for key, pattern in FIRST_LINE_PATTERNS.items():
        match = pattern.search(first_line)
        meta[key] = match.group(1) if match else None

    batch_size = parse_optional_int(meta["batch_size"])
    if batch_size is None:
        raise ValueError(f"Could not parse batch_size from log line: {first_line[:120]}")

    return {
        "batch_size": batch_size,
        "note": meta["note"],
        "inject_slowdown_gpu": parse_optional_int(meta["inject_slowdown_gpu"]),
        "slowdown_factor": parse_optional_float(meta["slowdown_factor"]),
        "slowdown_start": parse_optional_int(meta["slowdown_start"]),
        "slowdown_end": parse_optional_int(meta["slowdown_end"]),
        "scenario": meta["failover_inject_scenario"] or "",
    }


def aggregate_step_points(
    profiling_dir: Path,
) -> Tuple[Dict[int, List[StepPoint]], Dict[Tuple[int, int], float], Dict[Tuple[int, int], int]]:
    files = sorted(profiling_dir.glob("gpu_task_summary_partition*.jsonl"))
    if not files:
        raise ValueError(f"No profiling traces found under {profiling_dir}")

    earliest_start: Optional[float] = None
    occurrence_gap_sec = 30.0

    records_by_gpu_batch: Dict[Tuple[int, int], List[Dict[str, Optional[float]]]] = defaultdict(list)
    for trace_path in files:
        with trace_path.open("r", encoding="utf-8") as handle:
            line_no = 0
            for line in handle:
                line_no += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                gpu_id = record.get("device")
                batch_id = record.get("batch_id")
                if gpu_id is None or batch_id is None:
                    continue

                try:
                    gpu_id = int(gpu_id)
                    batch_id = int(batch_id)
                except (TypeError, ValueError):
                    continue

                duration_ms = record.get("exec_wall_ms", record.get("time_ms", 0.0))
                try:
                    duration_ms = float(duration_ms or 0.0)
                except (TypeError, ValueError):
                    continue
                if duration_ms <= 0:
                    continue

                start_time = record.get("start_time")
                try:
                    start_time = float(start_time) if start_time is not None else None
                except (TypeError, ValueError):
                    start_time = None

                if start_time is not None:
                    end_time = start_time + (duration_ms / 1000.0)
                    if earliest_start is None or start_time < earliest_start:
                        earliest_start = start_time
                else:
                    end_time = None

                records_by_gpu_batch[(gpu_id, batch_id)].append(
                    {
                        "line_no": float(line_no),
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_ms": duration_ms,
                    }
                )

    temp_points_by_gpu: Dict[int, List[Dict[str, Optional[float]]]] = defaultdict(list)
    occurrence_anchor_time_sec: Dict[Tuple[int, int], float] = {}
    occurrence_index_by_gpu_batch: Dict[Tuple[int, int], int] = defaultdict(int)

    for (gpu_id, batch_id), records in records_by_gpu_batch.items():
        records.sort(
            key=lambda item: (
                float(item["start_time"]) if item["start_time"] is not None else float("inf"),
                float(item["line_no"] or 0.0),
            )
        )

        current_min_start: Optional[float] = None
        current_max_end: Optional[float] = None
        current_sum_ms = 0.0
        last_end_time: Optional[float] = None

        def flush_current():
            nonlocal current_min_start, current_max_end, current_sum_ms, last_end_time
            if current_sum_ms <= 0:
                current_min_start = None
                current_max_end = None
                current_sum_ms = 0.0
                last_end_time = None
                return

            rel_start = None
            if current_min_start is not None and earliest_start is not None:
                rel_start = current_min_start - earliest_start

            occurrence_index = occurrence_index_by_gpu_batch[(gpu_id, batch_id)]
            occurrence_index_by_gpu_batch[(gpu_id, batch_id)] += 1

            if rel_start is not None:
                anchor_key = (batch_id, occurrence_index)
                anchor_existing = occurrence_anchor_time_sec.get(anchor_key)
                if anchor_existing is None or rel_start < anchor_existing:
                    occurrence_anchor_time_sec[anchor_key] = rel_start

            temp_points_by_gpu[gpu_id].append(
                {
                    "raw_step": float(batch_id),
                    "occurrence_index": float(occurrence_index),
                    # Step time should represent wall-clock latency for the step on
                    # this GPU, not the sum of every task duration. Summing task
                    # durations overcounts overlapped work and inflates spikes.
                    "step_time_ms": (
                        max((current_max_end - current_min_start) * 1000.0, 0.0)
                        if current_min_start is not None and current_max_end is not None
                        else current_sum_ms
                    ),
                    "sum_time_ms": current_sum_ms,
                    "start_time_sec": rel_start,
                }
            )

            current_min_start = None
            current_max_end = None
            current_sum_ms = 0.0
            last_end_time = None

        for item in records:
            start_time = item["start_time"]
            end_time = item["end_time"]
            duration_ms = float(item["duration_ms"] or 0.0)

            if (
                current_sum_ms > 0
                and start_time is not None
                and last_end_time is not None
                and (start_time - last_end_time) > occurrence_gap_sec
            ):
                flush_current()

            current_sum_ms += duration_ms

            if start_time is not None:
                current_min_start = start_time if current_min_start is None else min(current_min_start, start_time)
            if end_time is not None:
                current_max_end = end_time if current_max_end is None else max(current_max_end, end_time)
                last_end_time = end_time

        flush_current()

    if not temp_points_by_gpu:
        raise ValueError(f"No valid step points found under {profiling_dir}")

    occurrence_plot_step: Dict[Tuple[int, int], int] = {}
    occurrence_run_index: Dict[Tuple[int, int], int] = {}

    sorted_occurrences = sorted(
        occurrence_anchor_time_sec.items(),
        key=lambda item: (item[1], item[0][0], item[0][1]),
    )

    current_run_index = 0
    prev_raw_step: Optional[int] = None
    for plot_step, (occurrence_key, anchor_time) in enumerate(sorted_occurrences, start=1):
        raw_step, _ = occurrence_key
        if prev_raw_step is not None and raw_step + 20 < prev_raw_step:
            current_run_index += 1
        occurrence_plot_step[occurrence_key] = plot_step
        occurrence_run_index[occurrence_key] = current_run_index
        prev_raw_step = raw_step

    step_points_by_gpu: Dict[int, List[StepPoint]] = defaultdict(list)
    run_step_time_sec: Dict[Tuple[int, int], float] = {}
    run_step_plot_step: Dict[Tuple[int, int], int] = {}

    for gpu_id, raw_points in temp_points_by_gpu.items():
        for raw_point in raw_points:
            raw_step = int(raw_point["raw_step"] or 0.0)
            occurrence_index = int(raw_point["occurrence_index"] or 0.0)
            occurrence_key = (raw_step, occurrence_index)
            plot_step = occurrence_plot_step.get(occurrence_key)
            if plot_step is None:
                plot_step = raw_step
            run_index = occurrence_run_index.get(occurrence_key, 0)

            run_key = (run_index, raw_step)
            run_step_plot_step.setdefault(run_key, plot_step)
            if raw_point["start_time_sec"] is not None:
                run_step_time_sec.setdefault(run_key, float(raw_point["start_time_sec"]))

            step_points_by_gpu[gpu_id].append(
                StepPoint(
                    run_index=run_index,
                    raw_step=raw_step,
                    occurrence_index=occurrence_index,
                    plot_step=plot_step,
                    step_time_ms=float(raw_point["step_time_ms"] or 0.0),
                    sum_time_ms=float(raw_point["sum_time_ms"] or 0.0),
                    start_time_sec=(
                        float(raw_point["start_time_sec"])
                        if raw_point["start_time_sec"] is not None
                        else None
                    ),
                )
            )

    for gpu_id in list(step_points_by_gpu.keys()):
        step_points_by_gpu[gpu_id].sort(
            key=lambda point: (
                float(point.start_time_sec) if point.start_time_sec is not None else float("inf"),
                point.run_index,
                point.plot_step,
                point.raw_step,
            )
        )

    return dict(step_points_by_gpu), run_step_time_sec, run_step_plot_step


def parse_policy_events(lines: Iterable[str], run_index: int) -> List[PolicyEvent]:
    events: List[PolicyEvent] = []
    for line in lines:
        match = POLICY_DECISION_RE.search(line)
        if not match:
            continue

        details = match.group("details")
        step_match = STEP_IN_DETAILS_RE.search(details)
        if not step_match:
            continue

        gpu_match = GPU_IN_DETAILS_RE.search(details)
        slowdown_match = SLOWDOWN_IN_DETAILS_RE.search(details)
        if slowdown_match is None:
            slowdown_match = WALL_RATIO_IN_DETAILS_RE.search(details)
        if slowdown_match is None:
            slowdown_match = LOCAL_RATIO_IN_DETAILS_RE.search(details)

        events.append(
            PolicyEvent(
                run_index=run_index,
                step=int(step_match.group(1)),
                policy=match.group("policy"),
                gpu_id=int(gpu_match.group(1)) if gpu_match else None,
                slowdown_ratio=float(slowdown_match.group(1)) if slowdown_match else None,
            )
        )

    events.sort(key=lambda event: event.step)
    return events


def parse_resume_step(lines: Iterable[str]) -> int:
    for line in lines:
        match = RESUME_STEP_RE.search(line)
        if match:
            return int(match.group(1))
    return 0


def parse_injection_events(
    lines: Iterable[str],
    meta: Dict[str, Optional[object]],
    run_index: int,
) -> List[InjectionEvent]:
    raw_events: List[InjectionEvent] = []

    for line in lines:
        synthetic = SYNTHETIC_INJECTION_RE.search(line)
        if synthetic:
            phase = synthetic.group("phase").strip()
            raw_events.append(
                InjectionEvent(
                    run_index=run_index,
                    step=int(synthetic.group("step")),
                    gpu_id=int(synthetic.group("gpu")),
                    label=f"slowdown_injected:{phase}",
                    event_type="start",
                    slowdown_ratio=float(synthetic.group("slowdown")),
                )
            )
            continue

        real = REAL_INJECTION_RE.search(line)
        if real:
            raw_events.append(
                InjectionEvent(
                    run_index=run_index,
                    step=int(real.group("step")),
                    gpu_id=int(real.group("gpu")),
                    label="slowdown_injected:REAL",
                    event_type="start",
                    slowdown_ratio=float(real.group("slowdown")),
                )
            )

    slowdown_start = meta.get("slowdown_start")
    slowdown_end = meta.get("slowdown_end")
    slowdown_gpu = meta.get("inject_slowdown_gpu")
    slowdown_factor = meta.get("slowdown_factor")

    if isinstance(slowdown_start, int):
        raw_events.append(
            InjectionEvent(
                run_index=run_index,
                step=slowdown_start,
                gpu_id=slowdown_gpu if isinstance(slowdown_gpu, int) else None,
                label="slowdown_config_start",
                event_type="start",
                slowdown_ratio=slowdown_factor if isinstance(slowdown_factor, float) else None,
            )
        )

    if isinstance(slowdown_end, int) and isinstance(slowdown_start, int) and slowdown_end > slowdown_start:
        raw_events.append(
            InjectionEvent(
                run_index=run_index,
                step=slowdown_end,
                gpu_id=slowdown_gpu if isinstance(slowdown_gpu, int) else None,
                label="slowdown_config_end",
                event_type="end",
                slowdown_ratio=None,
            )
        )

    raw_events.sort(key=lambda event: (event.run_index, event.step, event.event_type, event.label))

    collapsed: List[InjectionEvent] = []
    prev_by_key: Dict[Tuple[int, Optional[int], str, str], int] = {}
    for event in raw_events:
        key = (event.run_index, event.gpu_id, event.event_type, event.label)
        prev_step = prev_by_key.get(key)
        if prev_step is not None and event.step <= prev_step + 1:
            prev_by_key[key] = event.step
            continue
        collapsed.append(event)
        prev_by_key[key] = event.step

    collapsed.sort(key=lambda event: event.step)
    return collapsed


def split_log_sections(log_path: Path) -> List[List[str]]:
    with log_path.open("r", encoding="utf-8") as handle:
        lines = list(handle)

    sections: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if line.startswith("args = Namespace("):
            if current:
                sections.append(current)
            current = [line]
        elif current:
            current.append(line)

    if current:
        sections.append(current)

    if not sections:
        sections.append(lines)
    return sections


def parse_log_file(
    log_path: Path,
) -> Tuple[Dict[str, Optional[object]], Dict[int, int], List[PolicyEvent], List[InjectionEvent]]:
    sections = split_log_sections(log_path)
    first_section = sections[0]
    first_args_line = next((line.strip() for line in first_section if line.strip()), "")
    meta = parse_experiment_meta_line(first_args_line)
    if meta.get("note") is None:
        meta["note"] = log_path.parent.name

    run_resume_steps: Dict[int, int] = {}
    policies: List[PolicyEvent] = []
    injections: List[InjectionEvent] = []
    for run_index, section in enumerate(sections):
        args_line = next((line.strip() for line in section if line.strip()), "")
        try:
            section_meta = parse_experiment_meta_line(args_line)
        except ValueError:
            section_meta = dict(meta)
        if section_meta.get("note") is None:
            section_meta["note"] = meta.get("note")

        run_resume_steps[run_index] = parse_resume_step(section)
        policies.extend(parse_policy_events(section, run_index))
        injections.extend(parse_injection_events(section, section_meta, run_index))

    return meta, run_resume_steps, policies, injections


def load_experiment(exp_dir: Path) -> ExperimentData:
    log_path = exp_dir / "log.txt"
    profiling_dir = exp_dir / "profiling_logs"

    if not profiling_dir.exists():
        raise ValueError(f"Missing profiling_logs: {exp_dir}")

    if log_path.exists():
        meta, run_resume_steps, policies, injections = parse_log_file(log_path)
        batch_size = int(meta["batch_size"])
        note = str(meta["note"] or exp_dir.name)
        scenario = str(meta["scenario"] or "")
    else:
        batch_size = None
        note = exp_dir.name
        scenario = ""
        run_resume_steps = {}
        policies = []
        injections = []

    step_points_by_gpu, run_step_time_sec, run_step_plot_step = aggregate_step_points(profiling_dir)

    return ExperimentData(
        name=note,
        path=exp_dir,
        batch_size=batch_size,
        scenario=scenario,
        step_points_by_gpu=step_points_by_gpu,
        run_step_time_sec=run_step_time_sec,
        run_step_plot_step=run_step_plot_step,
        run_resume_steps=run_resume_steps,
        policies=policies,
        injections=injections,
    )


def discover_experiment_dirs(search_root: Path) -> List[Path]:
    candidates: List[Path] = []
    if not search_root.exists():
        return candidates
    for child in sorted(search_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "log.txt").exists() and (child / "profiling_logs").exists():
            candidates.append(child)
    return candidates


def choose_gpu_ids(experiment: ExperimentData, requested_gpu_ids: Optional[Sequence[int]]) -> List[int]:
    seen = sorted(experiment.step_points_by_gpu.keys())
    if requested_gpu_ids:
        return [gpu_id for gpu_id in requested_gpu_ids if gpu_id in experiment.step_points_by_gpu]
    return seen[:4]


def build_point_lookup(points: Sequence[StepPoint]) -> Dict[Tuple[int, int], List[StepPoint]]:
    grouped: Dict[Tuple[int, int], List[StepPoint]] = defaultdict(list)
    for point in points:
        grouped[(point.run_index, point.raw_step)].append(point)
    return dict(grouped)


def resolve_run_step(experiment: ExperimentData, run_index: int, step: int) -> Optional[int]:
    run_key = (run_index, step)
    if run_key in experiment.run_step_time_sec or run_key in experiment.run_step_plot_step:
        return step

    resume_step = experiment.run_resume_steps.get(run_index, 0)
    adjusted_step = step - resume_step
    adjusted_key = (run_index, adjusted_step)
    if adjusted_step >= 0 and (
        adjusted_key in experiment.run_step_time_sec or adjusted_key in experiment.run_step_plot_step
    ):
        return adjusted_step

    return None


def step_to_axis_value(experiment: ExperimentData, run_index: int, step: int, x_axis: str) -> Optional[float]:
    resolved_step = resolve_run_step(experiment, run_index, step)
    if resolved_step is None:
        return None
    run_key = (run_index, resolved_step)
    if x_axis == "step":
        plot_step = experiment.run_step_plot_step.get(run_key)
        return float(plot_step) if plot_step is not None else None
    return experiment.run_step_time_sec.get(run_key)


def point_axis_value(point: StepPoint, experiment: ExperimentData, x_axis: str) -> Optional[float]:
    if x_axis == "step":
        return float(point.plot_step)
    if point.start_time_sec is not None:
        return point.start_time_sec
    return experiment.run_step_time_sec.get((point.run_index, point.raw_step))


def select_marker_point(
    candidates: Sequence[StepPoint],
    experiment: ExperimentData,
    x_axis: str,
    event_x: float,
) -> Optional[StepPoint]:
    best_point: Optional[StepPoint] = None
    best_distance: Optional[float] = None
    for candidate in candidates:
        candidate_x = point_axis_value(candidate, experiment, x_axis)
        if candidate_x is None:
            continue
        distance = abs(candidate_x - event_x)
        if best_distance is None or distance < best_distance:
            best_point = candidate
            best_distance = distance
    return best_point


def build_policy_segments(experiment: ExperimentData, x_axis: str, x_min: float, x_max: float) -> List[Tuple[str, float, float]]:
    segments: List[Tuple[str, float, float]] = []
    run_ranges: Dict[int, Tuple[float, float]] = {}
    for points in experiment.step_points_by_gpu.values():
        for point in points:
            point_x = point_axis_value(point, experiment, x_axis)
            if point_x is None:
                continue
            existing = run_ranges.get(point.run_index)
            if existing is None:
                run_ranges[point.run_index] = (point_x, point_x)
            else:
                run_ranges[point.run_index] = (min(existing[0], point_x), max(existing[1], point_x))

    carried_policy: Optional[str] = None

    for run_index in sorted(run_ranges.keys()):
        run_min, run_max = run_ranges[run_index]
        current_policy = carried_policy or "NORMAL"
        current_start = run_min

        run_events = [
            event for event in experiment.policies
            if event.run_index == run_index
        ]
        run_events.sort(
            key=lambda item: (
                step_to_axis_value(experiment, item.run_index, item.step, x_axis)
                if step_to_axis_value(experiment, item.run_index, item.step, x_axis) is not None
                else float("inf")
            )
        )

        for event in run_events:
            event_x = step_to_axis_value(experiment, event.run_index, event.step, x_axis)
            if event_x is None:
                continue
            event_x = max(min(event_x, run_max), run_min)

            if event.policy == current_policy:
                continue

            if event_x > current_start:
                segments.append((current_policy, current_start, event_x))

            current_policy = event.policy
            current_start = event_x

        if run_max > current_start:
            segments.append((current_policy, current_start, run_max))

        # REPLAN/DEGRADE are reconfiguration actions whose effect persists into
        # the next restarted run, so carry them forward until another policy changes it.
        if current_policy in {"REPLAN", "DEGRADE"}:
            carried_policy = current_policy
        else:
            carried_policy = None

    return segments


def format_injection_annotation(event: InjectionEvent) -> str:
    if event.event_type == "end":
        base = "Slowdown End"
    else:
        base = "Slowdown Start"
    suffix = event.label.split(":", 1)[-1]
    if suffix not in {"REAL", "CONFIG", "slowdown_config_start", "slowdown_config_end"} and "config" not in suffix:
        base = f"{base} ({suffix})"
    if event.slowdown_ratio is not None:
        base = f"{base} {event.slowdown_ratio:.2f}x"
    if event.gpu_id is not None:
        base = f"{base} / GPU {event.gpu_id}"
    return base


def format_policy_annotation(event: PolicyEvent) -> str:
    text = event.policy
    if event.gpu_id is not None:
        text = f"{text} / GPU {event.gpu_id}"
    if event.slowdown_ratio is not None:
        text = f"{text} / {event.slowdown_ratio:.2f}x"
    return text


def plot_experiment(
    experiment: ExperimentData,
    output_dir: Path,
    x_axis: str,
    gpu_ids: Optional[Sequence[int]],
) -> Optional[Path]:
    selected_gpus = choose_gpu_ids(experiment, gpu_ids)
    if not selected_gpus:
        return None

    plotted_points: Dict[int, List[StepPoint]] = {
        gpu_id: experiment.step_points_by_gpu[gpu_id]
        for gpu_id in selected_gpus
    }

    all_x: List[float] = []
    all_y: List[float] = []
    point_lookup_by_gpu: Dict[int, Dict[Tuple[int, int], List[StepPoint]]] = {}
    run_x_ranges: Dict[int, Tuple[float, float]] = {}

    for gpu_id, points in plotted_points.items():
        point_lookup_by_gpu[gpu_id] = build_point_lookup(points)
        for point in points:
            x_value = point_axis_value(point, experiment, x_axis)
            if x_value is None:
                continue
            all_x.append(x_value)
            all_y.append(point.step_time_ms)
            existing_range = run_x_ranges.get(point.run_index)
            if existing_range is None:
                run_x_ranges[point.run_index] = (x_value, x_value)
            else:
                run_x_ranges[point.run_index] = (
                    min(existing_range[0], x_value),
                    max(existing_range[1], x_value),
                )

    if not all_x or not all_y:
        return None

    x_min = min(all_x)
    x_max = max(all_x)
    y_max = max(all_y)
    y_padding = max(y_max * 0.08, 10.0)
    full_y_top = y_max + y_padding
    figure, axis = plt.subplots(figsize=(15, 6.8))

    def plot_gpu_lines(target_axis: plt.Axes) -> None:
        draw_order = sorted(selected_gpus, key=lambda gpu_id: (gpu_id != 3, gpu_id))
        color_index_by_gpu = {gpu_id: idx for idx, gpu_id in enumerate(selected_gpus)}

        for gpu_id in draw_order:
            color_idx = color_index_by_gpu[gpu_id]
            run_groups: Dict[int, List[StepPoint]] = defaultdict(list)
            for point in plotted_points[gpu_id]:
                run_groups[point.run_index].append(point)

            first_segment = True
            for run_index in sorted(run_groups.keys()):
                xs: List[float] = []
                ys: List[float] = []
                for point in run_groups[run_index]:
                    x_value = point_axis_value(point, experiment, x_axis)
                    if x_value is None:
                        continue
                    xs.append(x_value)
                    ys.append(point.step_time_ms)

                if not xs:
                    continue

                target_axis.plot(
                    xs,
                    ys,
                    label=f"GPU {gpu_id}" if first_segment else None,
                    color=GPU_COLORS[color_idx % len(GPU_COLORS)],
                    linestyle=GPU_LINESTYLES.get(gpu_id, "-"),
                    linewidth=1.4 if gpu_id == 3 else 1.9,
                    alpha=0.68 if gpu_id == 3 else 0.95,
                    zorder=1 if gpu_id == 3 else 2,
                )
                first_segment = False

    def draw_events(target_axis: plt.Axes, axis_top: float) -> None:
        for event in experiment.injections:
            event_x = step_to_axis_value(experiment, event.run_index, event.step, x_axis)
            if event_x is None and event.event_type == "end":
                # In these failover probe runs, slowdown_end is often configured far
                # beyond the run's actual last step (for example 10000), so the
                # process restarts before the explicit end step is ever reached.
                # Treat the run boundary as the effective slowdown end to avoid
                # graphs that show repeated starts with no corresponding end.
                run_range = run_x_ranges.get(event.run_index)
                if run_range is not None:
                    event_x = run_range[1]
            if event_x is None:
                continue

            style = SLOWDOWN_STYLES.get(event.event_type, SLOWDOWN_STYLES["start"])
            target_axis.axvline(
                event_x,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=0.92,
                zorder=3,
                label=None,
            )
            target_axis.scatter(
                [event_x],
                [axis_top * 0.965],
                marker=style["marker"],
                s=58,
                color=style["color"],
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
                clip_on=False,
            )

        for event in experiment.policies:
            event_x = step_to_axis_value(experiment, event.run_index, event.step, x_axis)
            if event_x is None:
                continue

            marker = POLICY_MARKERS.get(event.policy, "X")
            color = POLICY_COLORS.get(event.policy, "#666666")
            y_value = axis_top * 0.90
            if event.gpu_id is not None and event.gpu_id in point_lookup_by_gpu:
                resolved_step = resolve_run_step(experiment, event.run_index, event.step)
                matched = select_marker_point(
                    point_lookup_by_gpu[event.gpu_id].get((event.run_index, resolved_step or event.step), []),
                    experiment,
                    x_axis,
                    event_x,
                )
                if matched is not None:
                    y_value = min(matched.step_time_ms, axis_top * 0.94)

            target_axis.scatter(
                [event_x],
                [y_value],
                marker=marker,
                s=90,
                color=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
                label=None,
            )

    plot_gpu_lines(axis)
    draw_events(axis, full_y_top)

    axis.set_xlim(x_min, x_max)
    axis.set_ylim(bottom=0.0, top=full_y_top)
    axis.grid(True, axis="y", alpha=0.25, linestyle="--")
    axis.set_ylabel("Step Time (ms)")
    axis.set_xlabel("Step" if x_axis == "step" else "Elapsed Time (s)")

    batch_suffix = (
        f"batch_size={experiment.batch_size}"
        if experiment.batch_size is not None
        else "batch_size=unknown"
    )
    scenario_suffix = f" | scenario={experiment.scenario}" if experiment.scenario else ""
    axis.set_title(
        f"{experiment.name} | {batch_suffix}{scenario_suffix}\n"
        "Per-GPU Step Time with Event Markers"
    )

    overlay_handles = [
        Line2D(
            [0],
            [0],
            color=SLOWDOWN_STYLES["start"]["color"],
            linestyle=SLOWDOWN_STYLES["start"]["linestyle"],
            linewidth=SLOWDOWN_STYLES["start"]["linewidth"],
            marker=SLOWDOWN_STYLES["start"]["marker"],
            markersize=7,
            label="Slowdown Start",
        ),
        Line2D(
            [0],
            [0],
            color=SLOWDOWN_STYLES["end"]["color"],
            linestyle=SLOWDOWN_STYLES["end"]["linestyle"],
            linewidth=SLOWDOWN_STYLES["end"]["linewidth"],
            marker=SLOWDOWN_STYLES["end"]["marker"],
            markersize=7,
            label="Slowdown End",
        ),
        Line2D(
            [0],
            [0],
            marker=POLICY_MARKERS["KEEP"],
            color="none",
            markerfacecolor=POLICY_COLORS["KEEP"],
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=8,
            label="Policy Decision: KEEP",
        ),
        Line2D(
            [0],
            [0],
            marker=POLICY_MARKERS["REPLAN"],
            color="none",
            markerfacecolor=POLICY_COLORS["REPLAN"],
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=8,
            label="Policy Decision: REPLAN",
        ),
        Line2D(
            [0],
            [0],
            marker=POLICY_MARKERS["DEGRADE"],
            color="none",
            markerfacecolor=POLICY_COLORS["DEGRADE"],
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=8,
            label="Policy Decision: DEGRADE",
        ),
    ]

    line_handles, line_labels = axis.get_legend_handles_labels()
    merged_handles = line_handles + overlay_handles
    merged_labels = line_labels + [handle.get_label() for handle in overlay_handles]

    dedup_handles: List[object] = []
    dedup_labels: List[str] = []
    seen_labels = set()
    for handle, label in zip(merged_handles, merged_labels):
        if not label or label in seen_labels:
            continue
        dedup_handles.append(handle)
        dedup_labels.append(label)
        seen_labels.add(label)

    axis.legend(
        dedup_handles,
        dedup_labels,
        loc="upper right",
        framealpha=0.92,
        fontsize=9,
        ncol=2,
    )

    figure.tight_layout()

    batch_dir_name = f"batch_{experiment.batch_size}" if experiment.batch_size is not None else "batch_unknown"
    batch_dir = output_dir / batch_dir_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    output_path = batch_dir / f"{experiment.path.name}_gpu_step_timeline_{x_axis}.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dirs",
        nargs="*",
        help=(
            "Experiment directories like results/e2e_failover_0330 or "
            "profiling_logs directories like results/e2e_failover_0330/profiling_logs. "
            "If omitted, directories are auto-discovered."
        ),
    )
    parser.add_argument(
        "--search-root",
        default="results",
        help="Root directory used for auto-discovery when experiment_dirs is empty.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/gpu_step_timeline_plots",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("step", "time"),
        default="step",
        help="X axis type.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3],
        help="GPU ids to plot. Missing GPUs are skipped automatically.",
    )
    return parser.parse_args()


def resolve_experiment_dirs(args: argparse.Namespace) -> List[Path]:
    if args.experiment_dirs:
        resolved_paths = []
        for raw_path in args.experiment_dirs:
            path = Path(raw_path).resolve()
            if path.is_dir() and path.name == "profiling_logs":
                resolved_paths.append(path.parent)
            else:
                resolved_paths.append(path)
        return resolved_paths
    return [path.resolve() for path in discover_experiment_dirs(Path(args.search_root).resolve())]


def main() -> int:
    args = parse_args()
    exp_dirs = resolve_experiment_dirs(args)
    if not exp_dirs:
        raise SystemExit("No experiment directories found.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    successes = []
    failures = []

    for exp_dir in exp_dirs:
        try:
            experiment = load_experiment(exp_dir)
            output_path = plot_experiment(
                experiment=experiment,
                output_dir=output_dir,
                x_axis=args.x_axis,
                gpu_ids=args.gpu_ids,
            )
            if output_path is None:
                failures.append((exp_dir, "No plottable points"))
                continue
            successes.append((exp_dir, output_path, experiment.batch_size))
        except Exception as exc:  # pragma: no cover - CLI diagnostics
            failures.append((exp_dir, str(exc)))

    for _, output_path, batch_size in successes:
        batch_label = batch_size if batch_size is not None else "unknown"
        print(f"[OK] batch_size={batch_label} -> {output_path}")

    for exp_dir, reason in failures:
        print(f"[SKIP] {exp_dir}: {reason}")

    if not successes:
        raise SystemExit("No plots were generated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
