#!/usr/bin/env python3
"""Generate a combined CR/no-CR statistical figure across modes and datasets."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASELINES = [
    "Early Fusion",
    "Late Fusion",
    "Shared Latent",
    "Cross Attention",
    "FOCAL",
    "TACA",
    "DeepAV",
]

PANELS = [
    ("sync", "WearGait", "WearGait Sync"),
    ("sync", "FOG", "FOG Sync"),
    ("async", "WearGait", "WearGait Async"),
    ("async", "FOG", "FOG Async"),
]

CSV_SOURCES = {
    ("sync", "WearGait", "w/ CR"): "results/ablation/figures/table4_sync_stats/sync_weargait_cr_stat_tests.csv",
    ("sync", "WearGait", "w/o CR"): "results/ablation/figures/table4_sync_stats/sync_weargait_no_cr_stat_tests.csv",
    ("sync", "FOG", "w/ CR"): "results/ablation/figures/table4_sync_stats/sync_fog_cr_stat_tests.csv",
    ("sync", "FOG", "w/o CR"): "results/ablation/figures/table4_sync_stats/sync_fog_no_cr_stat_tests.csv",
    ("async", "WearGait", "w/ CR"): "results/ablation/figures/async_weargait_stats/async_weargait_cr_stat_tests.csv",
    ("async", "WearGait", "w/o CR"): "results/ablation/figures/async_weargait_stats/async_weargait_no_cr_stat_tests.csv",
    ("async", "FOG", "w/ CR"): "results/ablation/figures/async_fog_stats/async_fog_cr_stat_tests.csv",
    ("async", "FOG", "w/o CR"): "results/ablation/figures/async_fog_stats/async_fog_no_cr_stat_tests.csv",
}

COLORS = {
    "w/ CR": "#2F7BD7",
    "w/o CR": "#D9822B",
}

DISPLAY_CR = {
    "w/ CR": "w/ Class Rebalancing",
    "w/o CR": "w/o Class Rebalancing",
}


def read_rows(root: Path) -> dict[tuple[str, str, str, str], dict[str, float]]:
    data: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for (mode, dataset, cr_block), rel_path in CSV_SOURCES.items():
        path = root / rel_path
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                key = (mode, dataset, cr_block, row["baseline"])
                data[key] = {
                    "trip_mean": float(row["trip_mean"]),
                    "delta": float(row["paired_delta"]),
                    "lo": float(row["delta_ci_low"]),
                    "hi": float(row["delta_ci_high"]),
                    "p": float(row["exact_signflip_p"]),
                }
    return data


def write_combined_csv(data: dict[tuple[str, str, str, str], dict[str, float]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "dataset", "cr_block", "baseline", "trip_mean", "paired_delta", "delta_ci_low", "delta_ci_high", "exact_signflip_p"])
        for mode, dataset, _title in PANELS:
            for baseline in BASELINES:
                for cr_block in ("w/ CR", "w/o CR"):
                    stat = data.get((mode, dataset, cr_block, baseline))
                    if stat is None:
                        continue
                    writer.writerow([mode, dataset, cr_block, baseline, stat["trip_mean"], stat["delta"], stat["lo"], stat["hi"], stat["p"]])


def plot(data: dict[tuple[str, str, str, str], dict[str, float]], out_base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(22.0, 12.5), dpi=180, sharex=False)
    axes = axes.ravel()
    bar_h = 0.34
    offsets = {"w/ CR": -bar_h / 2, "w/o CR": bar_h / 2}

    for ax, (mode, dataset, title) in zip(axes, PANELS):
        y = np.arange(len(BASELINES))
        panel_max = 0.0
        trip_means = []
        for cr_block in ("w/ CR", "w/o CR"):
            deltas = []
            xerr_low = []
            xerr_high = []
            y_positions = []
            for i, baseline in enumerate(BASELINES):
                stat = data.get((mode, dataset, cr_block, baseline))
                if stat is None:
                    continue
                trip_means.append(stat["trip_mean"])
                deltas.append(stat["delta"])
                xerr_low.append(stat["delta"] - stat["lo"])
                xerr_high.append(stat["hi"] - stat["delta"])
                y_positions.append(i + offsets[cr_block])
                panel_max = max(panel_max, stat["hi"])

            ax.barh(
                y_positions,
                deltas,
                height=bar_h * 0.88,
                color=COLORS[cr_block],
                edgecolor="#2F3A45",
                linewidth=0.8,
                label=DISPLAY_CR[cr_block],
                zorder=2,
            )
            ax.errorbar(
                deltas,
                y_positions,
                xerr=np.vstack([xerr_low, xerr_high]),
                fmt="none",
                ecolor="#26323D",
                elinewidth=1.3,
                capsize=3.5,
                capthick=1.0,
                zorder=3,
            )

        ax.axvline(0, color="#5A5A5A", linewidth=1.0)
        trip_mean = np.mean(trip_means) if trip_means else float("nan")
        ax.set_title(f"{title}  (TRIP acc={trip_mean:.2f}%)", fontsize=16, pad=12)
        ax.set_yticks(y)
        ax.set_yticklabels(BASELINES, fontsize=11)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#E3E3E3", linewidth=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(-0.5, panel_max + 9.0)
        ax.tick_params(axis="x", labelsize=10)

        for cr_block in ("w/ CR", "w/o CR"):
            for i, baseline in enumerate(BASELINES):
                stat = data.get((mode, dataset, cr_block, baseline))
                if stat is None:
                    continue
                marker = "*" if stat["p"] < 0.05 else "n.s."
                ax.text(
                    stat["hi"] + 0.35,
                    i + offsets[cr_block],
                    f"Δ={stat['delta']:.1f} [{stat['lo']:.1f}, {stat['hi']:.1f}], p={stat['p']:.3f} {marker}",
                    va="center",
                    ha="left",
                    fontsize=7.4,
                    color="#202020",
                )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.985))
    fig.supxlabel("Paired improvement over baseline (TRIP - baseline, percentage points)", fontsize=14, y=0.035)
    fig.suptitle("Paired Statistical Testing Across Datasets and Evaluation Modes", fontsize=22, y=1.02)
    fig.text(
        0.012,
        0.012,
        "Bars show mean paired improvement across six seeds; whiskers show 95% CI. * exact two-sided sign-flip p<0.05; n.s. not significant.",
        fontsize=10.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.055, 1.0, 0.95))

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(".")
    data = read_rows(root)
    out_base = root / "results/ablation/figures/combined_stats/combined_sync_async_cr_no_cr"
    write_combined_csv(data, out_base.with_suffix(".csv"))
    plot(data, out_base)


if __name__ == "__main__":
    main()
