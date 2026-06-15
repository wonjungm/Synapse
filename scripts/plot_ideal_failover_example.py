#!/usr/bin/env python3
"""Create a minimal illustrative paper-style failover figure.

This plot is intentionally synthetic. It focuses on the two figures that are
most important for a base-vs-dynamic failover comparison:
1. Base-normalized end-to-end completion time
2. First ETA-minimizing policy map among KEEP / REPLAN / DEGRADE
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PNG = REPO_ROOT / "analysis" / "ideal_failover_example_figure.png"
OUTPUT_PDF = REPO_ROOT / "analysis" / "ideal_failover_example_figure.pdf"

COLORS = {
    "static": "#b45309",
    "dynamic": "#166534",
    "keep": "#6d28d9",
    "replan": "#2563eb",
    "degrade": "#dc2626",
    "slowdown": "#f59e0b",
    "restart": "#93c5fd",
}


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.18, linewidth=0.8)


def plot_normalized_completion(axis):
    batch_sizes = [32, 64, 128]
    static = [1.0, 1.0, 1.0]
    # Illustrative but intentionally conservative:
    # dynamic benefit grows as the remaining runtime after slowdown grows.
    dynamic = [0.96, 0.86, 0.74]
    x = list(range(len(batch_sizes)))
    width = 0.34

    axis.bar(
        [v - width / 2 for v in x],
        static,
        width=width,
        color=COLORS["static"],
        label="Static TSPipe",
        alpha=0.9,
    )
    axis.bar(
        [v + width / 2 for v in x],
        dynamic,
        width=width,
        color=COLORS["dynamic"],
        label="ETA-aware Dynamic",
        alpha=0.92,
    )

    for idx, value in enumerate(dynamic):
        reduction = (1.0 - value) * 100.0
        axis.text(
            idx + width / 2,
            value + 0.025,
            f"-{reduction:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLORS["dynamic"],
        )

    axis.set_xticks(x)
    axis.set_xticklabels([str(v) for v in batch_sizes])
    axis.set_ylim(0.0, 1.16)
    axis.set_xlabel("Batch Size")
    axis.set_ylabel("Normalized End-to-End Time\n(static base = 1.0)")
    axis.set_title("A. Must-Have: Base-Normalized Completion Time")
    axis.legend(frameon=False, loc="upper right")
    style_axis(axis)


POLICY_TO_ID = {
    "KEEP": 0,
    "REPLAN": 1,
    "DEGRADE": 2,
}

ID_TO_POLICY = {value: key for key, value in POLICY_TO_ID.items()}


def eta_keep(slowdown: float, k_rem: int) -> float:
    return k_rem * (0.70 * slowdown)


def eta_replan(slowdown: float, k_rem: int) -> float:
    # REPLAN keeps the slow GPU but reduces the bottleneck via repartition.
    # Under severe slowdown it still degrades somewhat, so DEGRADE can win.
    stage_time = 0.74 + 0.18 * (slowdown - 1.0)
    restart_cost = 190.0
    return restart_cost + k_rem * stage_time


def eta_degrade(slowdown: float, k_rem: int) -> float:
    # DEGRADE pays a larger one-time reconfiguration cost but removes the slow GPU.
    stage_time = 0.98
    restart_cost = 240.0
    return restart_cost + k_rem * stage_time


def choose_policy(slowdown: float, k_rem: int) -> str:
    eta_values = {
        "KEEP": eta_keep(slowdown, k_rem),
        "REPLAN": eta_replan(slowdown, k_rem),
        "DEGRADE": eta_degrade(slowdown, k_rem),
    }
    return min(eta_values, key=eta_values.get)


def plot_policy_map(axis):
    slowdown_factors = [1.1, 1.3, 1.6, 2.0, 2.8, 3.5]
    scenarios = [
        ("Early stage\n(large K_rem)", 2200),
        ("Mid stage\n(medium K_rem)", 900),
        ("Late stage\n(small K_rem)", 220),
    ]

    matrix = []
    for _label, k_rem in scenarios:
        row = []
        for slowdown in slowdown_factors:
            policy = choose_policy(slowdown, k_rem)
            row.append(POLICY_TO_ID[policy])
        matrix.append(row)

    cmap = ListedColormap(
        [
            "#ede9fe",
            "#dbeafe",
            "#fee2e2",
        ]
    )
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    for row_idx, row in enumerate(matrix):
        for col_idx, policy_id in enumerate(row):
            policy = ID_TO_POLICY[policy_id]
            text_color = {
                "KEEP": COLORS["keep"],
                "REPLAN": COLORS["replan"],
                "DEGRADE": COLORS["degrade"],
            }[policy]
            axis.text(
                col_idx,
                row_idx,
                policy,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=text_color,
            )

    axis.set_xticks(range(len(slowdown_factors)))
    axis.set_xticklabels([f"{value:.1f}x" for value in slowdown_factors])
    axis.set_yticks(range(len(scenarios)))
    axis.set_yticklabels([label for label, _k_rem in scenarios])
    axis.set_xlabel("Effective Slowdown Factor")
    axis.set_ylabel("Scenario")
    axis.set_title("B. Must-Have: First ETA-Minimizing Policy by Scenario")

    axis.set_xticks([v - 0.5 for v in range(1, len(slowdown_factors))], minor=True)
    axis.set_yticks([v - 0.5 for v in range(1, len(scenarios))], minor=True)
    axis.grid(which="minor", color="white", linewidth=2.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor="#ede9fe", edgecolor="none", label="KEEP"),
        Patch(facecolor="#dbeafe", edgecolor="none", label="REPLAN"),
        Patch(facecolor="#fee2e2", edgecolor="none", label="DEGRADE"),
    ]
    axis.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    axis.text(
        0.5,
        -0.28,
        "One cell = first policy chosen at the first slowdown trigger in that scenario.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#4b5563",
    )


def main():
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    figure.patch.set_facecolor("white")

    plot_normalized_completion(axes[0])
    plot_policy_map(axes[1])

    figure.suptitle(
        "Illustrative Must-Have Figures for Base vs Dynamic Failover",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.015,
        "Synthetic guide figure for intuition. Panel B should show ETA-based policy boundaries, not raw trigger counts.",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )

    plt.tight_layout(rect=[0.02, 0.04, 1.0, 0.95])
    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
