#!/usr/bin/env python3
"""Generate rebuttal statistical-test figures from ablation logs/results."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


SEEDS = (2, 3, 4, 42, 43, 44)
BASELINE_LABELS = {
    "early_fusion": "Early Fusion",
    "late_fusion": "Late Fusion",
    "shared_latent": "Shared Latent",
    "cheap_xattn": "Cross Attention",
    "focal": "FOCAL",
    "taca": "TACA",
    "deepav_lite": "DeepAV",
}

AVG_PATTERNS = (
    re.compile(r"mean skel=.*?avg=([0-9.]+)%", re.IGNORECASE),
    re.compile(r"Mean Eval Acc:\s*([0-9.]+)%", re.IGNORECASE),
    re.compile(r"Macro acc mean\s*.*?:\s*([0-9.]+)%", re.IGNORECASE),
)

WEARGAIT_ASYNC_OURS = {
    2: 69.312,
    3: 70.768,
    4: 69.715,
    42: 71.529,
    43: 71.447,
    44: 71.817,
}


def parse_seed_score(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    matches: list[float] = []
    for pattern in AVG_PATTERNS:
        matches.extend(float(x) for x in pattern.findall(text))
    return matches[-1] if matches else None


def load_method_scores(folder: Path) -> dict[int, float]:
    scores = {}
    for seed in SEEDS:
        score = parse_seed_score(folder / f"seed{seed}.out")
        if score is not None:
            scores[seed] = score
    return scores


def paired_stats(ours: dict[int, float], base: dict[int, float]) -> dict[str, float]:
    seeds = [seed for seed in SEEDS if seed in ours and seed in base]
    if len(seeds) != len(SEEDS):
        missing = sorted(set(SEEDS) - set(seeds))
        raise ValueError(f"missing paired seeds: {missing}")
    ours_arr = np.array([ours[s] for s in seeds], dtype=float)
    base_arr = np.array([base[s] for s in seeds], dtype=float)
    diff = ours_arr - base_arr
    delta = float(np.mean(diff))
    sem = float(stats.sem(diff))
    margin = float(stats.t.ppf(0.975, len(diff) - 1) * sem) if sem > 0 else 0.0

    perm_means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(diff)):
        perm_means.append(abs(float(np.mean(diff * np.array(signs)))))
    exact_p = sum(v >= abs(delta) - 1e-12 for v in perm_means) / len(perm_means)

    try:
        wilcoxon_p = float(stats.wilcoxon(diff, alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon_p = math.nan

    return {
        "trip_mean": float(np.mean(ours_arr)),
        "baseline_mean": float(np.mean(base_arr)),
        "paired_delta": delta,
        "delta_ci_low": delta - margin,
        "delta_ci_high": delta + margin,
        "exact_signflip_p": exact_p,
        "wilcoxon_p": wilcoxon_p,
    }


def rows_from_async_logs(
    log_root: Path,
    cr_block: str,
    dataset: str,
    *,
    ours_scores: dict[int, float] | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    ours = ours_scores if ours_scores is not None else load_method_scores(log_root / "cr" / "ours")
    if len(ours) != len(SEEDS):
        fallback = load_method_scores(log_root / "ours")
        if len(fallback) == len(SEEDS):
            ours = fallback
    baseline_root = log_root / cr_block
    rows = []
    skipped = []
    for method, label in BASELINE_LABELS.items():
        folder = baseline_root / method
        if not folder.exists():
            skipped.append(f"{label}: folder missing")
            continue
        base = load_method_scores(folder)
        if len(base) != len(SEEDS):
            skipped.append(f"{label}: {len(base)}/6 complete seeds")
            continue
        stat = paired_stats(ours, base)
        rows.append(
            {
                "dataset": dataset,
                "cr_block": "w/ CR" if cr_block == "cr" else "w/o CR",
                "baseline": label,
                **{k: f"{v:.5f}" for k, v in stat.items()},
            }
        )
    return rows, skipped


def rows_from_sync_csv(csv_path: Path, dataset: str, cr_block: str) -> list[dict[str, str]]:
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset and row["cr_block"] == cr_block:
                rows.append(row)
    return rows


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda r: float(r["paired_delta"]), reverse=True)


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fields = [
        "dataset",
        "cr_block",
        "baseline",
        "trip_mean",
        "baseline_mean",
        "paired_delta",
        "delta_ci_low",
        "delta_ci_high",
        "exact_signflip_p",
        "wilcoxon_p",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, str]], path: Path, title: str, skipped: list[str] | None = None) -> None:
    lines = [
        f"# {title}: Paired TRIP Improvement vs Baselines",
        "",
        "| Baseline | TRIP mean | Baseline mean | Paired delta | 95% CI | Exact p | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {baseline} | {trip:.2f} | {base:.2f} | {delta:+.2f} | [{lo:.2f}, {hi:.2f}] | {exact:.5f} | {wilc:.5f} |".format(
                baseline=row["baseline"],
                trip=float(row["trip_mean"]),
                base=float(row["baseline_mean"]),
                delta=float(row["paired_delta"]),
                lo=float(row["delta_ci_low"]),
                hi=float(row["delta_ci_high"]),
                exact=float(row["exact_signflip_p"]),
                wilc=float(row["wilcoxon_p"]),
            )
        )
    if skipped:
        lines.extend(["", "Skipped incomplete baselines:"])
        lines.extend(f"- {item}" for item in skipped)
    path.write_text("\n".join(lines) + "\n")


def plot_rows(rows: list[dict[str, str]], path_base: Path, title: str) -> None:
    if not rows:
        raise ValueError(f"no rows to plot for {title}")

    labels = [r["baseline"] for r in rows]
    deltas = np.array([float(r["paired_delta"]) for r in rows])
    lows = np.array([float(r["delta_ci_low"]) for r in rows])
    highs = np.array([float(r["delta_ci_high"]) for r in rows])
    ps = np.array([float(r["exact_signflip_p"]) for r in rows])
    sig = ps < 0.05

    y = np.arange(len(rows))
    fig_h = max(6.4, 1.0 + 0.84 * len(rows))
    fig, ax = plt.subplots(figsize=(11.8, fig_h), dpi=180)

    colors = np.where(sig, "#3F7FD5", "#8EABC8")
    edge = "#2F3A45"
    ax.barh(y, deltas, height=0.46, color=colors, edgecolor=edge, linewidth=1.1, zorder=2)
    ax.errorbar(
        deltas,
        y,
        xerr=np.vstack([deltas - lows, highs - deltas]),
        fmt="none",
        ecolor="#26323D",
        elinewidth=1.8,
        capsize=5,
        capthick=1.5,
        zorder=3,
    )

    ax.axvline(0, color="#575757", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=18, pad=16)
    ax.set_xlabel("Paired improvement over baseline (percentage points)", fontsize=13)
    ax.grid(axis="x", color="#E0E0E0", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    x_min = min(-0.5, float(np.min(lows)) - 1.0)
    x_max = float(np.max(highs)) + 8.4
    ax.set_xlim(x_min, x_max)

    for i, row in enumerate(rows):
        hi = float(row["delta_ci_high"])
        delta = float(row["paired_delta"])
        lo = float(row["delta_ci_low"])
        p = float(row["exact_signflip_p"])
        suffix = "*" if p < 0.05 else "n.s."
        ax.text(
            hi + 0.45,
            i,
            f"Δ={delta:+.2f}  CI [{lo:.2f}, {hi:.2f}]  p={p:.5f} {suffix}",
            va="center",
            ha="left",
            fontsize=11.3,
            color="#202020",
        )

    trip_mean = float(rows[0]["trip_mean"])
    fig.text(
        0.012,
        0.018,
        f"TRIP mean={trip_mean:.2f}% over six seeds; exact p is two-sided sign-flip permutation on paired seed scores. * p<0.05.",
        fontsize=10.6,
        color="#444444",
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))

    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def render(rows: list[dict[str, str]], out_base: Path, title: str, skipped: list[str] | None = None) -> None:
    rows = sort_rows(rows)
    write_csv(rows, out_base.with_name(out_base.name + "_stat_tests.csv"))
    write_md(rows, out_base.with_suffix(".md"), title, skipped)
    if not rows:
        return
    plot_rows(rows, out_base, f"{title}: TRIP vs Baselines")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    args = p.parse_args()
    root = args.root

    async_root = root / "results/ablation/logs/async/fog"
    async_out = root / "results/ablation/figures/async_fog_stats"
    for block, name in (("cr", "w/ Class Rebalancing"), ("no_cr", "w/o Class Rebalancing")):
        rows, skipped = rows_from_async_logs(async_root, block, "FOG")
        render(rows, async_out / f"async_fog_{block}", f"FOG Async, {name}", skipped)

    weargait_async_root = root / "results/ablation/logs/async/weargait"
    weargait_async_out = root / "results/ablation/figures/async_weargait_stats"
    for block, name in (("cr", "w/ Class Rebalancing"), ("no_cr", "w/o Class Rebalancing")):
        rows, skipped = rows_from_async_logs(
            weargait_async_root,
            block,
            "WearGait",
            ours_scores=WEARGAIT_ASYNC_OURS,
        )
        render(rows, weargait_async_out / f"async_weargait_{block}", f"WearGait Async, {name}", skipped)

    sync_csv = root / "results/ablation/table4_sync_stat_tests.csv"
    sync_out = root / "results/ablation/figures/table4_sync_stats"
    for dataset, dataset_name in (("FOG", "FOG"), ("WearGait", "WearGait")):
        for block, filename_block, title_block in (
            ("w/ CR", "cr", "w/ Class Rebalancing"),
            ("w/o CR", "no_cr", "w/o Class Rebalancing"),
        ):
            rows = rows_from_sync_csv(sync_csv, dataset, block)
            render(rows, sync_out / f"sync_{dataset_name.lower()}_{filename_block}", f"{dataset_name} Sync, {title_block}")


if __name__ == "__main__":
    main()
