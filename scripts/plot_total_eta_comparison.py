#!/usr/bin/env python3
"""Plot grouped bars comparing total completion ETA by batch size.

This script is designed for slowdown/failover experiments where each batch size
has two comparison targets:
  - TSPipe static partitioning (typically KEEP-only)
  - Failover-aware dynamic partitioning (AUTO policy selection)

It reads a JSON config file. Empty paths are allowed so unfinished baselines can
stay blank until the corresponding logs are available.

Example:
  python scripts/plot_total_eta_comparison.py \
    --config analysis/failover_total_eta_config.json \
    --output analysis/failover_total_eta_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]

BATCH_SIZE_RE = re.compile(r"\bbatch_size=(\d+)")
REASONING_ETA_RE = re.compile(
    r"Reasoning:\s+ETA:\s+KEEP=(?P<keep>[0-9.]+)s,\s+"
    r"REPLAN=(?P<replan>[0-9.]+)s,\s+DEGRADE=(?P<degrade>[0-9.]+)s",
    re.IGNORECASE,
)
OPTIMAL_RE = re.compile(r"ETA Analysis:\s+K_rem=\d+,\s+Optimal=(?P<optimal>[a-z_]+)", re.IGNORECASE)
POLICY_DECISION_RE = re.compile(r"Failover Policy Decision:\s+(?P<policy>[A-Z_]+)\s+\(")
STEP_RE = re.compile(r"\bstep=(\d+)")
TOTAL_WALL_CLOCK_RE = re.compile(r"Total wall-clock time:\s+(?P<seconds>\d+)s")

POLICY_TO_KEY = {
    "KEEP": "keep_sec",
    "REPLAN": "replan_sec",
    "DEGRADE": "degrade_sec",
}

SERIES_STYLES = {
    "tspipe_static": {
        "label": "TSPipe (Static KEEP)",
        "color": "#d97706",
    },
    "failover_aware": {
        "label": "Failover-aware (Dynamic)",
        "color": "#2f855a",
    },
}


@dataclass
class EtaEvent:
    line_no: int
    step: Optional[int]
    policy: str
    optimal: Optional[str]
    keep_sec: float
    replan_sec: float
    degrade_sec: float

    def seconds_for_policy(self, policy: str) -> float:
        return float(getattr(self, POLICY_TO_KEY[policy]))


@dataclass
class Measurement:
    series_key: str
    raw_path: str
    batch_size: Optional[int]
    total_sec: Optional[float]
    selected_policy: Optional[str]
    source: str
    line_no: Optional[int]
    step: Optional[int]
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot grouped bars comparing total ETA by batch size.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "analysis" / "failover_total_eta_config.json",
        help="JSON config describing batch sizes and experiment paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "analysis" / "failover_total_eta_comparison.png",
        help="PNG path for the grouped bar chart.",
    )
    parser.add_argument(
        "--unit",
        choices=("auto", "sec", "min"),
        default="auto",
        help="Display unit for the Y axis.",
    )
    parser.add_argument(
        "--title",
        default="Batch Size vs Total ETA under Slowdown",
        help="Plot title.",
    )
    return parser.parse_args()


def resolve_path(raw_path: str) -> Optional[Path]:
    if not raw_path or not raw_path.strip():
        return None
    candidate = Path(raw_path.strip())
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def read_first_batch_size(log_path: Path) -> Optional[int]:
    if not log_path.exists():
        return None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = BATCH_SIZE_RE.search(line)
            if match:
                return int(match.group(1))
    return None


def parse_eta_events(log_path: Path) -> List[EtaEvent]:
    events: List[EtaEvent] = []
    pending_eta: Optional[Dict[str, object]] = None
    pending_signature: Optional[str] = None
    current_optimal: Optional[str] = None

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            optimal_match = OPTIMAL_RE.search(line)
            if optimal_match:
                current_optimal = optimal_match.group("optimal").upper()

            reasoning_match = REASONING_ETA_RE.search(line)
            if reasoning_match:
                signature = reasoning_match.group(0)
                if signature != pending_signature:
                    pending_eta = {
                        "line_no": line_no,
                        "optimal": current_optimal,
                        "keep_sec": float(reasoning_match.group("keep")),
                        "replan_sec": float(reasoning_match.group("replan")),
                        "degrade_sec": float(reasoning_match.group("degrade")),
                    }
                    pending_signature = signature
                continue

            decision_match = POLICY_DECISION_RE.search(line)
            if not decision_match or pending_eta is None:
                continue

            if line_no - int(pending_eta["line_no"]) > 8:
                continue

            step_match = STEP_RE.search(line)
            events.append(
                EtaEvent(
                    line_no=line_no,
                    step=int(step_match.group(1)) if step_match else None,
                    policy=decision_match.group("policy").upper(),
                    optimal=str(pending_eta["optimal"]).upper()
                    if pending_eta["optimal"] is not None
                    else None,
                    keep_sec=float(pending_eta["keep_sec"]),
                    replan_sec=float(pending_eta["replan_sec"]),
                    degrade_sec=float(pending_eta["degrade_sec"]),
                )
            )
            pending_eta = None
            pending_signature = None

    return events


def parse_total_wall_clock(summary_path: Path) -> Optional[float]:
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = TOTAL_WALL_CLOCK_RE.search(line)
            if match:
                return float(match.group("seconds"))
    return None


def choose_event(events: Sequence[EtaEvent], policy_hint: str, event_index: int) -> Optional[EtaEvent]:
    if not events:
        return None

    filtered: List[EtaEvent]
    policy_hint = policy_hint.upper()
    if policy_hint in POLICY_TO_KEY:
        filtered = [event for event in events if event.policy == policy_hint]
    else:
        filtered = [event for event in events if event.policy in {"REPLAN", "DEGRADE"}]
        if not filtered:
            filtered = list(events)

    if not filtered:
        return None
    if event_index < 0:
        event_index = 0
    if event_index >= len(filtered):
        event_index = len(filtered) - 1
    return filtered[event_index]


def infer_log_and_summary(target_path: Path) -> Dict[str, Optional[Path]]:
    if target_path.is_dir():
        log_path = target_path / "log.txt"
        summary_path = target_path / "e2e_summary.log"
        return {
            "log": log_path if log_path.exists() else None,
            "summary": summary_path if summary_path.exists() else None,
        }

    if target_path.name == "log.txt":
        summary_path = target_path.parent / "e2e_summary.log"
        return {
            "log": target_path,
            "summary": summary_path if summary_path.exists() else None,
        }

    if target_path.name == "e2e_summary.log":
        log_path = target_path.parent / "log.txt"
        return {
            "log": log_path if log_path.exists() else None,
            "summary": target_path,
        }

    return {"log": None, "summary": None}


def measure_series(
    series_key: str,
    series_config: Dict[str, object],
    batch_size_hint: Optional[int],
) -> Measurement:
    raw_path = str(series_config.get("path", "") or "")
    target_path = resolve_path(raw_path)
    if target_path is None:
        return Measurement(
            series_key=series_key,
            raw_path=raw_path,
            batch_size=batch_size_hint,
            total_sec=None,
            selected_policy=None,
            source="missing_path",
            line_no=None,
            step=None,
            error="Path is empty",
        )

    if not target_path.exists():
        return Measurement(
            series_key=series_key,
            raw_path=raw_path,
            batch_size=batch_size_hint,
            total_sec=None,
            selected_policy=None,
            source="missing_path",
            line_no=None,
            step=None,
            error=f"Path not found: {target_path}",
        )

    mode = str(series_config.get("mode", "eta")).lower()
    policy_hint = str(series_config.get("policy", "AUTO")).upper()
    event_index = int(series_config.get("event_index", 0) or 0)
    files = infer_log_and_summary(target_path)
    log_path = files["log"]
    summary_path = files["summary"]

    batch_size = batch_size_hint
    if batch_size is None and log_path is not None:
        batch_size = read_first_batch_size(log_path)

    if mode in {"eta", "auto"} and log_path is not None:
        events = parse_eta_events(log_path)
        event = choose_event(events, policy_hint=policy_hint, event_index=event_index)
        if event is not None:
            selected_policy = policy_hint if policy_hint in POLICY_TO_KEY else event.policy
            return Measurement(
                series_key=series_key,
                raw_path=raw_path,
                batch_size=batch_size,
                total_sec=event.seconds_for_policy(selected_policy),
                selected_policy=selected_policy,
                source=f"log_eta:{log_path.relative_to(REPO_ROOT)}",
                line_no=event.line_no,
                step=event.step,
            )
        if mode == "eta":
            return Measurement(
                series_key=series_key,
                raw_path=raw_path,
                batch_size=batch_size,
                total_sec=None,
                selected_policy=None,
                source=f"log_eta:{log_path.relative_to(REPO_ROOT)}",
                line_no=None,
                step=None,
                error="No ETA/policy decision pair found",
            )

    if mode in {"wall_clock", "auto"} and summary_path is not None:
        total_sec = parse_total_wall_clock(summary_path)
        if total_sec is not None:
            return Measurement(
                series_key=series_key,
                raw_path=raw_path,
                batch_size=batch_size,
                total_sec=total_sec,
                selected_policy=policy_hint if policy_hint in POLICY_TO_KEY else None,
                source=f"e2e_summary:{summary_path.relative_to(REPO_ROOT)}",
                line_no=None,
                step=None,
            )

    return Measurement(
        series_key=series_key,
        raw_path=raw_path,
        batch_size=batch_size,
        total_sec=None,
        selected_policy=None,
        source="unresolved",
        line_no=None,
        step=None,
        error="Could not extract a total-time value from the configured path",
    )


def choose_unit(unit_arg: str, values_sec: Sequence[float]) -> str:
    if unit_arg != "auto":
        return unit_arg
    if not values_sec:
        return "sec"
    return "min" if max(values_sec) >= 600.0 else "sec"


def convert_seconds(seconds: float, unit: str) -> float:
    return seconds if unit == "sec" else seconds / 60.0


def format_value(seconds: float, unit: str) -> str:
    converted = convert_seconds(seconds, unit)
    suffix = "s" if unit == "sec" else "m"
    precision = 0 if unit == "sec" else 1
    return f"{converted:,.{precision}f}{suffix}"


def write_csv(output_path: Path, rows: List[Dict[str, object]]) -> Path:
    csv_path = output_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "series",
        "label",
        "path",
        "total_sec",
        "selected_policy",
        "source",
        "line_no",
        "step",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_grouped_bars(
    output_path: Path,
    title: str,
    unit: str,
    entries: List[Dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_sizes = [int(entry["batch_size"]) for entry in entries]
    available_seconds = [
        float(measurement.total_sec)
        for entry in entries
        for measurement in (entry["tspipe_static"], entry["failover_aware"])
        if measurement.total_sec is not None
    ]
    if not available_seconds:
        raise SystemExit("No plottable total-time values were extracted.")

    x_positions = list(range(len(entries)))
    width = 0.34
    converted_max = max(convert_seconds(value, unit) for value in available_seconds)
    figure_width = max(9.0, 2.0 * len(entries) + 3.0)
    figure, axis = plt.subplots(figsize=(figure_width, 6.4))

    for series_key, x_offset in (("tspipe_static", -width / 2), ("failover_aware", width / 2)):
        style = SERIES_STYLES[series_key]
        xs: List[float] = []
        heights: List[float] = []
        for group_index, entry in enumerate(entries):
            measurement: Measurement = entry[series_key]
            if measurement.total_sec is None:
                continue
            xs.append(x_positions[group_index] + x_offset)
            heights.append(convert_seconds(measurement.total_sec, unit))

        if xs:
            axis.bar(
                xs,
                heights,
                width=width,
                color=style["color"],
                alpha=0.92,
                label=style["label"],
            )

    for group_index, entry in enumerate(entries):
        tspipe: Measurement = entry["tspipe_static"]
        failover: Measurement = entry["failover_aware"]
        base_x = x_positions[group_index]

        if tspipe.total_sec is None:
            axis.text(
                base_x - width / 2,
                converted_max * 0.035,
                "TBD",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#6b7280",
                rotation=90,
            )
        else:
            axis.text(
                base_x - width / 2,
                convert_seconds(tspipe.total_sec, unit) + converted_max * 0.02,
                f"{tspipe.selected_policy or 'KEEP'}\n{format_value(tspipe.total_sec, unit)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#7c2d12",
            )

        if failover.total_sec is not None:
            axis.text(
                base_x + width / 2,
                convert_seconds(failover.total_sec, unit) + converted_max * 0.02,
                f"{failover.selected_policy or 'AUTO'}\n{format_value(failover.total_sec, unit)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#14532d",
            )

        top_y_candidates = []
        if tspipe.total_sec is not None:
            top_y_candidates.append(convert_seconds(tspipe.total_sec, unit))
        if failover.total_sec is not None:
            top_y_candidates.append(convert_seconds(failover.total_sec, unit))
        if not top_y_candidates:
            continue

        label_y = max(top_y_candidates) + converted_max * 0.11
        if tspipe.total_sec is not None and failover.total_sec is not None and tspipe.total_sec > 0:
            reduction_pct = ((tspipe.total_sec - failover.total_sec) / tspipe.total_sec) * 100.0
            if reduction_pct >= 0:
                speedup_label = f"{reduction_pct:.1f}% shorter"
            else:
                speedup_label = f"{abs(reduction_pct):.1f}% slower"
            axis.text(
                base_x,
                label_y,
                speedup_label,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#111827",
            )
        else:
            axis.text(
                base_x,
                label_y,
                "Speedup TBD",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#6b7280",
            )

    axis.set_xticks(x_positions)
    axis.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    axis.set_xlabel("Batch Size")
    axis.set_ylabel(f"Total Time ({'sec' if unit == 'sec' else 'min'})")
    axis.set_title(title)
    axis.set_ylim(0, converted_max * 1.28)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=SERIES_STYLES["tspipe_static"]["color"], label=SERIES_STYLES["tspipe_static"]["label"]),
        Patch(facecolor=SERIES_STYLES["failover_aware"]["color"], label=SERIES_STYLES["failover_aware"]["label"]),
        Patch(facecolor="#f3f4f6", edgecolor="#9ca3af", label="TBD baseline path"),
    ]
    axis.legend(handles=legend_handles, loc="upper right")

    figure.text(
        0.5,
        0.01,
        "Percentage labels are shown automatically once both bars are available for a batch size.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )

    figure.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    raw_entries = config.get("entries", [])
    if not raw_entries:
        raise SystemExit(f"No config entries found in {config_path}")

    processed_entries: List[Dict[str, object]] = []
    csv_rows: List[Dict[str, object]] = []
    for raw_entry in raw_entries:
        batch_size_hint = raw_entry.get("batch_size")
        tspipe_measurement = measure_series(
            "tspipe_static",
            raw_entry.get("tspipe_static", {}),
            batch_size_hint=batch_size_hint,
        )
        failover_measurement = measure_series(
            "failover_aware",
            raw_entry.get("failover_aware", {}),
            batch_size_hint=batch_size_hint,
        )

        resolved_batch_size = (
            batch_size_hint
            or tspipe_measurement.batch_size
            or failover_measurement.batch_size
        )
        if resolved_batch_size is None:
            raise SystemExit(f"Could not determine batch size for config entry: {raw_entry}")

        processed_entries.append(
            {
                "batch_size": int(resolved_batch_size),
                "tspipe_static": tspipe_measurement,
                "failover_aware": failover_measurement,
            }
        )

        for series_key, measurement in (
            ("tspipe_static", tspipe_measurement),
            ("failover_aware", failover_measurement),
        ):
            csv_rows.append(
                {
                    "batch_size": int(resolved_batch_size),
                    "series": series_key,
                    "label": SERIES_STYLES[series_key]["label"],
                    "path": measurement.raw_path,
                    "total_sec": measurement.total_sec,
                    "selected_policy": measurement.selected_policy,
                    "source": measurement.source,
                    "line_no": measurement.line_no,
                    "step": measurement.step,
                    "error": measurement.error,
                }
            )

    processed_entries.sort(key=lambda entry: int(entry["batch_size"]))
    plottable_values = [
        float(measurement.total_sec)
        for entry in processed_entries
        for measurement in (entry["tspipe_static"], entry["failover_aware"])
        if measurement.total_sec is not None and not math.isnan(measurement.total_sec)
    ]
    unit = choose_unit(args.unit, plottable_values)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    plot_grouped_bars(
        output_path=output_path,
        title=config.get("title", args.title),
        unit=unit,
        entries=processed_entries,
    )
    csv_path = write_csv(output_path, csv_rows)

    print(f"Wrote plot PNG: {output_path}")
    print(f"Wrote summary CSV: {csv_path}")
    for entry in processed_entries:
        batch_size = int(entry["batch_size"])
        tspipe: Measurement = entry["tspipe_static"]
        failover: Measurement = entry["failover_aware"]
        print(
            "Batch size {} -> TSPipe: {} | Failover-aware: {}".format(
                batch_size,
                format_value(tspipe.total_sec, unit) if tspipe.total_sec is not None else "TBD",
                format_value(failover.total_sec, unit) if failover.total_sec is not None else "TBD",
            )
        )


if __name__ == "__main__":
    main()
