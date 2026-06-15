#!/usr/bin/env python3
"""Plot a 4-batch bar chart comparing TSPipe and Failover runs."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DEFAULT_RUNS: Dict[int, Dict[str, str]] = {
    64: {
        "tspipe": "e2e_tspipe_bgload_gpu3_replan_strong_b64_e1_20260407_062247",
        "failover": "e2e_failover_bgload_gpu3_replan_strongplus_b64_e1_20260407_043747",
    },
    128: {
        "tspipe": "e2e_tspipe_bgload_gpu3_replan_strong_b128_e1_20260406_152710",
        "failover": "e2e_failover_bgload_gpu3_replan_strong_b128_e1_20260406_230448",
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

FIXED_RUNS: Dict[int, Dict[str, str]] = {
    128: {
        "tspipe": "e2e_retry_tspipe_bgload_gpu3_twostream_b128_e1_20260408_143452",
        "failover": "e2e_retry_failover_bgload_gpu3_replan_twostream_b128_e1_20260408_134919",
    }
}

BATCH_SIZES = (64, 128, 256, 512)


def _extract_run_timestamp(run_name: str) -> Optional[datetime]:
    match = re.search(r"(\d{8}_\d{6})", run_name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _auto_pick_run(batch_size: int, role: str) -> Optional[str]:
    role = role.lower().strip()
    candidates = []
    for path in RESULTS_DIR.iterdir():
        if not path.is_dir():
            continue
        if not (path / "e2e_summary.log").exists():
            continue

        name = path.name.lower()
        if f"b{batch_size}" not in name:
            continue
        if "bgload_gpu3" not in name:
            continue

        if role == "tspipe":
            if "tspipe" not in name or "failover" in name:
                continue
        elif role == "failover":
            if "failover" not in name:
                continue
        else:
            continue

        ts = _extract_run_timestamp(path.name)
        ts_value = ts if ts is not None else datetime.min
        candidates.append((ts_value, path.stat().st_mtime, path.name))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][2]


def _resolve_runs(run_overrides: Optional[Dict[int, Dict[str, str]]] = None) -> Dict[int, Dict[str, str]]:
    resolved: Dict[int, Dict[str, str]] = {
        batch: {
            "tspipe": pair["tspipe"],
            "failover": pair["failover"],
        }
        for batch, pair in DEFAULT_RUNS.items()
    }

    for batch, pair in FIXED_RUNS.items():
        resolved[batch] = {
            "tspipe": pair["tspipe"],
            "failover": pair["failover"],
        }

    for batch_size in BATCH_SIZES:
        if batch_size in FIXED_RUNS:
            continue
        resolved.setdefault(batch_size, {})
        for role in ("tspipe", "failover"):
            picked = _auto_pick_run(batch_size, role)
            if picked is not None:
                resolved[batch_size][role] = picked

    if run_overrides:
        for batch_size, pair in run_overrides.items():
            resolved.setdefault(batch_size, {})
            for role, run_name in pair.items():
                resolved[batch_size][role] = run_name

    return resolved


@dataclass
class RunSummary:
    run_dir: Path
    status: str
    total_seconds: int
    total_minutes: float
    restart_count: int


def parse_summary(path: Path) -> RunSummary:
    status = "UNKNOWN"
    total_seconds = None
    restart_count = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        if "Status:" in line:
            status = line.split("Status:", 1)[1].strip()
        elif "Total wall-clock time:" in line:
            match = re.search(r"Total wall-clock time:\s+(\d+)s", line)
            if match:
                total_seconds = int(match.group(1))
        elif "Restart count:" in line:
            match = re.search(r"Restart count:\s+(\d+)", line)
            if match:
                restart_count = int(match.group(1))

    if total_seconds is None:
        raise ValueError(f"Could not parse total wall-clock time from {path}")

    return RunSummary(
        run_dir=path.parent,
        status=status,
        total_seconds=total_seconds,
        total_minutes=total_seconds / 60.0,
        restart_count=restart_count,
    )


def load_runs(runs: Dict[int, Dict[str, str]]) -> List[dict]:
    rows: List[dict] = []
    for batch_size in BATCH_SIZES:
        pair = runs[batch_size]
        if "tspipe" not in pair or "failover" not in pair:
            raise ValueError(
                f"Missing run mapping for batch={batch_size}: {pair}. "
                "Check results directory names or update DEFAULT_RUNS/FIXED_RUNS."
            )
        tspipe = parse_summary(RESULTS_DIR / pair["tspipe"] / "e2e_summary.log")
        failover = parse_summary(RESULTS_DIR / pair["failover"] / "e2e_summary.log")
        improvement_pct = (tspipe.total_seconds - failover.total_seconds) / tspipe.total_seconds * 100.0
        rows.append(
            {
                "batch_size": batch_size,
                "tspipe": tspipe,
                "failover": failover,
                "improvement_pct": improvement_pct,
            }
        )
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_size",
                "tspipe_run_dir",
                "tspipe_status",
                "tspipe_total_seconds",
                "failover_run_dir",
                "failover_status",
                "failover_total_seconds",
                "failover_restart_count",
                "improvement_pct",
            ],
        )
        writer.writeheader()
        for row in rows:
            tspipe = row["tspipe"]
            failover = row["failover"]
            writer.writerow(
                {
                    "batch_size": row["batch_size"],
                    "tspipe_run_dir": str(tspipe.run_dir),
                    "tspipe_status": tspipe.status,
                    "tspipe_total_seconds": tspipe.total_seconds,
                    "failover_run_dir": str(failover.run_dir),
                    "failover_status": failover.status,
                    "failover_total_seconds": failover.total_seconds,
                    "failover_restart_count": failover.restart_count,
                    "improvement_pct": f"{row['improvement_pct']:.1f}",
                }
            )


def add_bar_labels(ax, bars, values: List[float], color: str) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.35,
            f"{value:.1f}m",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=color,
        )


def add_delta_labels(ax, failover_bars, rows: List[dict]) -> None:
    for bar, row in zip(failover_bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 2.45,
            f"-{row['improvement_pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#1B7F5A",
        )


def make_plot(rows: List[dict], output_base: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 180,
        }
    )

    batches = [row["batch_size"] for row in rows]
    tspipe_minutes = [row["tspipe"].total_minutes for row in rows]
    failover_minutes = [row["failover"].total_minutes for row in rows]

    x = np.arange(len(batches), dtype=float)
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 6.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    tspipe_bars = ax.bar(
        x - width / 2.0,
        tspipe_minutes,
        width=width,
        color="#4C78A8",
        edgecolor="#365B7D",
        linewidth=1.0,
        label="TSPipe baseline",
    )
    failover_bars = ax.bar(
        x + width / 2.0,
        failover_minutes,
        width=width,
        color="#D27A2C",
        edgecolor="#9E5B20",
        linewidth=1.0,
        label="Failover + REPLAN",
    )

    add_bar_labels(ax, tspipe_bars, tspipe_minutes, "#365B7D")
    add_bar_labels(ax, failover_bars, failover_minutes, "#9E5B20")
    add_delta_labels(ax, failover_bars, rows)

    ax.set_title("End-to-End Completion Time Under GPU-3 Background Load")
    ax.set_xlabel("Batch size", labelpad=10)
    ax.set_ylabel("Completion time (minutes)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(batch) for batch in batches])
    ax.set_ylim(0.0, max(tspipe_minutes + failover_minutes) + 6.8)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_run_override(value: str) -> Tuple[int, str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Run override must use BATCH:ROLE:RUN_NAME format."
        )

    batch_text, role, run_name = parts
    try:
        batch_size = int(batch_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid batch size in override: {batch_text!r}"
        ) from exc

    if batch_size not in BATCH_SIZES:
        raise argparse.ArgumentTypeError(
            f"Batch size must be one of {BATCH_SIZES}, got {batch_size}."
        )

    role = role.lower().strip()
    if role not in {"tspipe", "failover"}:
        raise argparse.ArgumentTypeError(
            f"Role must be 'tspipe' or 'failover', got {role!r}."
        )

    run_name = run_name.strip()
    if not run_name:
        raise argparse.ArgumentTypeError("Run name cannot be empty.")

    return batch_size, role, run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=FIGURES_DIR / "bgload_batch_bar_compare",
        help="Output path without extension. Defaults to results/figures/bgload_batch_bar_compare.",
    )
    parser.add_argument(
        "--run-override",
        action="append",
        default=[],
        metavar="BATCH:ROLE:RUN_NAME",
        help="Override a resolved run for one bar, e.g. 128:tspipe:my_run_dir.",
    )
    parser.add_argument(
        "--print-runs",
        action="store_true",
        help="Print the resolved run mapping before plotting.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Write the summary CSV only and skip the matplotlib PNG/PDF render.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_overrides: Dict[int, Dict[str, str]] = {}
    for override in args.run_override:
        batch_size, role, run_name = parse_run_override(override)
        run_overrides.setdefault(batch_size, {})[role] = run_name

    runs = _resolve_runs(run_overrides)
    if args.print_runs:
        for batch_size in BATCH_SIZES:
            pair = runs[batch_size]
            print(
                f"batch={batch_size} tspipe={pair.get('tspipe', '-')}"
                f" failover={pair.get('failover', '-')}"
            )

    output_base = args.output_base
    output_base.parent.mkdir(parents=True, exist_ok=True)
    rows = load_runs(runs)
    write_csv(output_base.with_suffix(".csv"), rows)
    print(f"Saved summary to {output_base.with_suffix('.csv')}")

    if args.csv_only:
        return

    make_plot(rows, output_base)
    print(f"Saved figure to {output_base.with_suffix('.png')}")
    print(f"Saved figure to {output_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
