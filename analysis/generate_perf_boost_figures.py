#!/usr/bin/env python3
"""
Generate CUDA_DISABLE_PERF_BOOST figures for the paper.

Data files:
  - H100 SXM: data/raw/perf_boost_h100_sxm.jsonl
  - A100 SXM4: data/raw/perf_boost_a100_sxm4.jsonl

Produces:
  1. perf_boost_power_comparison.png — grouped bar chart of idle power
  2. perf_boost_latency.png — latency distribution baseline vs flag_on
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style: match generate_paper_figures.py exactly ──────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

FIGURES_DIR = Path("figures")
PAPER_DIR = Path("paper")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Canonical colors from generate_paper_figures.py
GPU_COLORS = {
    "H100 SXM": "#3b82f6",
    "A100 SXM4": "#10b981",
}

DATA_FILES = {
    "H100 SXM": "data/raw/perf_boost_h100_sxm.jsonl",
    "A100 SXM4": "data/raw/perf_boost_a100_sxm4.jsonl",
}

# Condition display order and labels
POWER_CONDITIONS = ["bare", "baseline", "flag_on", "vllm_baseline", "vllm_flag_on"]
CONDITION_LABELS = {
    "bare": "Bare idle",
    "baseline": "CUDA ctx\n(default)",
    "flag_on": "CUDA ctx\n(flag on)",
    "vllm_baseline": "vLLM idle\n(default)",
    "vllm_flag_on": "vLLM idle\n(flag on)",
}
CONDITION_HATCHES = {
    "bare": "",
    "baseline": "",
    "flag_on": "//",
    "vllm_baseline": "",
    "vllm_flag_on": "//",
}


def load_data(filepath):
    """Load JSONL, separate power rows from latency rows."""
    power_rows = []
    latency_rows = []
    with open(filepath) as f:
        for line in f:
            row = json.loads(line.strip())
            if "power_w" in row:
                power_rows.append(row)
            elif "latency" in row:
                latency_rows.append(row)
    return power_rows, latency_rows


def compute_power_stats(power_rows):
    """Group by condition, compute mean/std."""
    by_cond = defaultdict(list)
    for r in power_rows:
        by_cond[r["condition"]].append(r["power_w"])
    stats = {}
    for cond, powers in by_cond.items():
        stats[cond] = {
            "mean": np.mean(powers),
            "std": np.std(powers, ddof=1) if len(powers) > 1 else 0,
            "n": len(powers),
            "values": powers,
        }
    return stats


def perf_boost_power_comparison():
    """Grouped bar chart: idle power across all conditions for both GPUs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, (gpu_label, filepath) in zip(axes, DATA_FILES.items()):
        power_rows, _ = load_data(filepath)
        stats = compute_power_stats(power_rows)
        color = GPU_COLORS[gpu_label]

        # Filter to conditions that exist
        conds = [c for c in POWER_CONDITIONS if c in stats]
        x = np.arange(len(conds))
        width = 0.6

        means = [stats[c]["mean"] for c in conds]
        stds = [stats[c]["std"] for c in conds]
        labels = [CONDITION_LABELS[c] for c in conds]

        # Color: bare=gray, baseline/vllm_baseline=gpu color, flag variants=lighter
        bar_colors = []
        for c in conds:
            if c == "bare":
                bar_colors.append("#94a3b8")
            elif "flag_on" in c:
                # Lighter version of GPU color
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                bar_colors.append(f"#{min(r+80,255):02x}{min(g+80,255):02x}{min(b+80,255):02x}")
            else:
                bar_colors.append(color)

        bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                      color=bar_colors, edgecolor="black", linewidth=0.5,
                      alpha=0.85, zorder=5)

        # Add hatching for flag_on conditions
        for i, c in enumerate(conds):
            if CONDITION_HATCHES[c]:
                bars[i].set_hatch(CONDITION_HATCHES[c])

        # Annotate power values
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 1.5, f"{m:.1f}W", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

        # Annotate savings arrows
        if "baseline" in stats and "flag_on" in stats:
            baseline_idx = conds.index("baseline")
            flag_idx = conds.index("flag_on")
            diff = stats["baseline"]["mean"] - stats["flag_on"]["mean"]
            mid_y = (stats["baseline"]["mean"] + stats["flag_on"]["mean"]) / 2
            ax.annotate("", xy=(flag_idx, stats["flag_on"]["mean"] + stds[flag_idx] + 3),
                       xytext=(baseline_idx, stats["baseline"]["mean"] - 2),
                       arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
            ax.text((baseline_idx + flag_idx) / 2, mid_y,
                    f"$-${diff:.0f}W", ha="center", va="center", fontsize=9,
                    fontweight="bold", color="red",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                             edgecolor="red", alpha=0.9))

        if "vllm_baseline" in stats and "vllm_flag_on" in stats:
            vb_idx = conds.index("vllm_baseline")
            vf_idx = conds.index("vllm_flag_on")
            diff = stats["vllm_baseline"]["mean"] - stats["vllm_flag_on"]["mean"]
            mid_y = (stats["vllm_baseline"]["mean"] + stats["vllm_flag_on"]["mean"]) / 2
            ax.annotate("", xy=(vf_idx, stats["vllm_flag_on"]["mean"] + stds[vf_idx] + 3),
                       xytext=(vb_idx, stats["vllm_baseline"]["mean"] - 2),
                       arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
            ax.text((vb_idx + vf_idx) / 2, mid_y,
                    f"$-${diff:.0f}W", ha="center", va="center", fontsize=9,
                    fontweight="bold", color="red",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                             edgecolor="red", alpha=0.9))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Idle Power (W)")
        ax.set_title(gpu_label, fontweight="bold")

        # y-axis from 0 to reasonable max
        ax.set_ylim(0, max(means) + 20)

    fig.suptitle("CUDA_DISABLE_PERF_BOOST: Idle Power Comparison",
                fontweight="bold", fontsize=12, y=1.03)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "perf_boost_power_comparison.png")
    fig.savefig(PAPER_DIR / "perf_boost_power_comparison.png")
    plt.close(fig)
    print("  Saved perf_boost_power_comparison")


LATENCY_RETEST_FILES = {
    "H100 SXM": "data/raw/latency_retest_h100_sxm.json",
    "A100 SXM4": "data/raw/latency_retest_a100_sxm4.json",
}


def perf_boost_latency():
    """Latency comparison using retest data: cold burst + warm steady-state."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (gpu_label, filepath) in zip(axes, LATENCY_RETEST_FILES.items()):
        with open(filepath) as f:
            data = json.load(f)
        color = GPU_COLORS[gpu_label]

        # Extract warm latencies (ms) for both conditions
        baseline_result = [r for r in data["results"] if r["condition"] == "baseline"][0]
        flag_result = [r for r in data["results"] if r["condition"] == "flag_on"][0]

        baseline_warm = np.array([l for l in baseline_result["warm"]["all"] if l is not None]) * 1000
        flag_warm = np.array([l for l in flag_result["warm"]["all"] if l is not None]) * 1000

        # Cold first-request latencies
        baseline_cold1 = baseline_result["cold"]["all"][0] * 1000
        flag_cold1 = flag_result["cold"]["all"][0] * 1000

        # Box plot for warm (steady-state) latencies
        bp = ax.boxplot(
            [baseline_warm, flag_warm],
            positions=[0, 1],
            widths=0.4,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=5, markerfacecolor="red",
                           markeredgecolor="red", alpha=0.7),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )

        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(color)
        bp["boxes"][1].set_alpha(0.4)
        bp["boxes"][1].set_hatch("//")

        # Overlay individual points (jittered)
        for i, (lats, pos) in enumerate([(baseline_warm, 0), (flag_warm, 1)]):
            jitter = np.random.default_rng(42).normal(0, 0.04, len(lats))
            ax.scatter(np.full_like(lats, pos) + jitter, lats,
                      c=color, alpha=0.5, s=15, zorder=4,
                      edgecolors="black", linewidths=0.3)

        # Mark cold first-request as a distinct marker
        ax.scatter([0], [baseline_cold1], marker="v", s=80, c="gray",
                  edgecolors="black", linewidths=1, zorder=10, label=f"Cold req 1: {baseline_cold1:.0f}ms")
        ax.scatter([1], [flag_cold1], marker="v", s=80, c="red",
                  edgecolors="black", linewidths=1, zorder=10, label=f"Cold req 1: {flag_cold1:.0f}ms")

        # Stats annotation
        b_mean = np.mean(baseline_warm)
        f_mean = np.mean(flag_warm)
        diff_mean = f_mean - b_mean
        ramp = flag_cold1 - f_mean

        stats_text = (
            f"Warm mean: {f_mean:.1f} vs {b_mean:.1f}ms\n"
            f"$\\Delta$mean: {diff_mean:+.1f}ms ({diff_mean/b_mean*100:+.1f}%)\n"
            f"Cold ramp: +{ramp:.0f}ms (req 1 only)"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                         edgecolor=color, alpha=0.9))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline\n(default)", "Flag on\n(PERF_BOOST=1)"],
                           fontsize=9)
        ax.set_ylabel("Request Latency (ms)")
        ax.set_title(gpu_label, fontweight="bold")
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(0.0, 0.55))

    fig.suptitle("Inference Latency: Baseline vs CUDA_DISABLE_PERF_BOOST",
                fontweight="bold", fontsize=12, y=1.03)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "perf_boost_latency.png")
    fig.savefig(PAPER_DIR / "perf_boost_latency.png")
    plt.close(fig)
    print("  Saved perf_boost_latency")


def main():
    print("Generating CUDA_DISABLE_PERF_BOOST figures...")
    for label, path in DATA_FILES.items():
        print(f"  {label}: {path}")
    print()

    perf_boost_power_comparison()
    perf_boost_latency()

    print(f"\nAll figures saved to {FIGURES_DIR}/ and {PAPER_DIR}/")


if __name__ == "__main__":
    main()
