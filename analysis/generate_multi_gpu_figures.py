#!/usr/bin/env python3
"""Generate multi-GPU figures for parking_tax.tex v2."""

import json
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch, FancyArrowPatch
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "multi_gpu.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent / "paper"

BARE_IDLE_W = 71.7  # mean across 4 GPUs

CONDITION_SHORT = {
    "tp2_baseline":  "TP=2",  "tp2_flag_on":  "TP=2",
    "tp4_baseline":  "TP=4",  "tp4_flag_on":  "TP=4",
    "dp2_baseline":  "DP=2",  "dp2_flag_on":  "DP=2",
    "tpdp_baseline": "TP2×DP2", "tpdp_flag_on": "TP2×DP2",
}

# Colors — match existing paper style (from analysis/generate_perf_boost_figures.py)
C_BASELINE = "#3b82f6"   # H100 blue from v1
C_FLAG_ON  = "#10b981"   # A100 green from v1
C_BARE     = "#ef4444"   # red for reference lines
C_CTX      = "#4a7cc9"   # CUDA context blue from v1
C_BASE_IDLE = "#94a3b8"  # gray from v1

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


def load_records():
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Figure 1: DVFS Decay Curve — the killer figure
# ---------------------------------------------------------------------------
def plot_decay_curve(records):
    """Power & clock vs time: baseline 1980 MHz plateau vs flag-on 2-second cliff.

    Uses TP=2 GPU 0 (clean data, no transient spikes).
    Two rows: SM Clock (top) and Power (bottom), overlaid baseline vs flag-on.
    """
    fig, (ax_clock, ax_power) = plt.subplots(
        2, 1, figsize=(7, 4.5), sharex=True,
        gridspec_kw={"hspace": 0.1, "height_ratios": [1, 1]})

    for cond, color, label in [
        ("tp2_baseline", C_BASELINE, "Baseline (flag off)"),
        ("tp2_flag_on",  C_FLAG_ON,  "Flag on (PERF_BOOST disabled)"),
    ]:
        samples = [r for r in records
                   if r["phase"] == "decay_curve"
                   and r["condition"] == cond
                   and r.get("gpu_id") == 0]
        if not samples:
            continue

        samples.sort(key=lambda r: r["timestamp"])
        t0 = datetime.fromisoformat(samples[0]["timestamp"])
        times = [(datetime.fromisoformat(s["timestamp"]) - t0).total_seconds()
                 for s in samples]
        powers = [s["power_w"] for s in samples]
        clocks = [s["sm_clock_mhz"] for s in samples]

        ax_clock.plot(times, clocks, color=color, lw=1.5, label=label, alpha=0.9)
        ax_power.plot(times, powers, color=color, lw=1.5, label=label, alpha=0.9)

    # Reference lines
    ax_clock.axhline(345, color=C_BARE, ls="--", lw=1, alpha=0.5)
    ax_power.axhline(BARE_IDLE_W, color=C_BARE, ls="--", lw=1, alpha=0.5)

    # Annotations on clock panel
    ax_clock.annotate(
        "DVFS governor never downclocks\n— locked at 1980 MHz\nfor 5 full minutes",
        xy=(200, 1980), xytext=(150, 1250),
        fontsize=8, color=C_BASELINE, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_BASELINE, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_BASELINE, alpha=0.85))

    ax_clock.annotate(
        "With flag: clocks settle\nto 345 MHz in ~1s",
        xy=(4, 400), xytext=(90, 700),
        fontsize=8, color=C_FLAG_ON, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_FLAG_ON, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_FLAG_ON, alpha=0.85))

    # Annotations on power panel
    ax_power.annotate(
        f"~125W parking tax\n(+53W over bare idle)",
        xy=(200, 128), xytext=(200, 260),
        fontsize=8, color=C_BASELINE, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_BASELINE, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_BASELINE, alpha=0.85))

    ax_power.annotate(
        f"~72W (= bare idle)",
        xy=(100, 72), xytext=(180, 160),
        fontsize=8, color=C_FLAG_ON, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_FLAG_ON, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_FLAG_ON, alpha=0.85))

    ax_clock.set_ylabel("SM Clock (MHz)")
    ax_clock.set_ylim(0, 2200)
    ax_clock.legend(loc="center right", framealpha=0.9, fontsize=8)

    ax_power.set_ylabel("GPU Power (W)")
    ax_power.set_xlabel("Time after last inference request (s)")
    ax_power.set_ylim(40, 450)
    ax_power.set_xlim(-5, 305)

    fig.suptitle("DVFS Decay After Inference: The Parking Tax in Real Time (TP=2, GPU 0)",
                 fontsize=11, fontweight="bold", y=1.01)

    out = OUT_DIR / "multi_gpu_decay_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Multi-GPU Parking Tax Bar Chart
# ---------------------------------------------------------------------------
def plot_parking_tax_bars(records):
    """Per-GPU mean idle power: grouped bars for each parallelism strategy."""
    pairs = [
        ("tp2_baseline", "tp2_flag_on"),
        ("tp4_baseline", "tp4_flag_on"),
        ("dp2_baseline", "dp2_flag_on"),
        ("tpdp_baseline", "tpdp_flag_on"),
    ]

    # Compute stats
    idle_data = {}
    for cond in [c for pair in pairs for c in pair]:
        samples = [r for r in records
                   if r["phase"] == "idle_power" and r["condition"] == cond]
        if not samples:
            continue
        gpu_ids = sorted(set(s["gpu_id"] for s in samples))
        per_gpu_w = []
        for gid in gpu_ids:
            pw = [s["power_w"] for s in samples if s["gpu_id"] == gid]
            per_gpu_w.append(np.mean(pw))
        idle_data[cond] = {
            "mean": np.mean(per_gpu_w),
            "std": np.std(per_gpu_w) if len(per_gpu_w) > 1 else 0,
            "n_gpus": len(gpu_ids),
        }

    fig, ax = plt.subplots(figsize=(8, 4))

    n_groups = len(pairs)
    bar_width = 0.35
    group_width = 2 * bar_width + 0.15
    group_centers = np.arange(n_groups) * (group_width + 0.4)

    for i, (base_cond, flag_cond) in enumerate(pairs):
        base = idle_data.get(base_cond, {"mean": 0, "std": 0})
        flag = idle_data.get(flag_cond, {"mean": 0, "std": 0})
        x_base = group_centers[i] - bar_width / 2 - 0.05
        x_flag = group_centers[i] + bar_width / 2 + 0.05

        # Baseline bar
        b1 = ax.bar(x_base, base["mean"], bar_width, yerr=base["std"],
                     capsize=3, color=C_BASELINE, edgecolor="black", linewidth=0.5,
                     alpha=0.85)
        # Flag bar
        b2 = ax.bar(x_flag, flag["mean"], bar_width, yerr=flag["std"],
                     capsize=3, color=C_FLAG_ON, edgecolor="black", linewidth=0.5,
                     hatch="//", alpha=0.85)

        # Value labels
        ax.text(x_base, base["mean"] + base["std"] + 1.5, f'{base["mean"]:.1f}W',
                ha="center", fontsize=7.5, fontweight="bold")
        ax.text(x_flag, flag["mean"] + flag["std"] + 1.5, f'{flag["mean"]:.1f}W',
                ha="center", fontsize=7.5, fontweight="bold")

        # Tax annotation with arrow between bars
        tax = base["mean"] - BARE_IDLE_W
        mid_x = group_centers[i]
        ax.annotate(f"−{tax:.0f}W", xy=(mid_x, base["mean"] - 5),
                    fontsize=8, color=C_BARE, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=C_BARE,
                              alpha=0.8))

    # Bare idle reference
    ax.axhline(BARE_IDLE_W, color=C_BARE, ls="--", lw=1.2, alpha=0.6)
    ax.text(group_centers[-1] + 0.7, BARE_IDLE_W + 1.5,
            f"Bare idle ({BARE_IDLE_W}W)", fontsize=8, color=C_BARE, va="bottom")

    # Group labels
    group_labels = [CONDITION_SHORT[p[0]] for p in pairs]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=10, fontweight="bold")

    # Legend
    legend_elements = [
        Patch(facecolor=C_BASELINE, edgecolor="black", label="Baseline (flag off)"),
        Patch(facecolor=C_FLAG_ON, edgecolor="black", hatch="//", label="Flag on (PERF_BOOST disabled)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    ax.set_ylabel("Per-GPU Idle Power (W)")
    ax.set_ylim(0, 150)
    ax.set_title("Multi-GPU Parking Tax Across Parallelism Strategies (4×H100 SXM)",
                 fontsize=11, fontweight="bold", pad=10)

    out = OUT_DIR / "multi_gpu_parking_tax.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Cold-Start Penalty — no compounding with N
# ---------------------------------------------------------------------------
def plot_cold_start_penalty(records):
    """Bar chart of first-request penalty across multi-GPU configs."""
    cold_records = [r for r in records if r["phase"] == "cold_start"]

    penalties = {}
    for r in cold_records:
        cond = r["condition"]
        cs = r.get("cold_start", {})
        penalty = cs.get("cold_penalty_ms")
        if penalty is not None:
            penalties.setdefault(cond, []).append(penalty)

    SINGLE_GPU_PENALTY = 150  # ms

    fig, ax = plt.subplots(figsize=(7, 3.8))

    flag_conds = ["tp2_flag_on", "tp4_flag_on", "dp2_flag_on", "tpdp_flag_on"]
    labels = ["TP=2\n(2 GPUs)", "TP=4\n(4 GPUs)", "DP=2\n(2 GPUs)", "TP2×DP2\n(4 GPUs)"]
    x = np.arange(len(flag_conds))

    penalty_means = []
    for cond in flag_conds:
        vals = penalties.get(cond, [])
        penalty_means.append(np.mean(vals) if vals else 0)

    bars = ax.bar(x, penalty_means, color=C_FLAG_ON, edgecolor="black",
                  linewidth=0.5, width=0.55, hatch="//", alpha=0.85)

    # Single-GPU reference
    ax.axhline(SINGLE_GPU_PENALTY, color=C_BARE, ls="--", lw=1.5, alpha=0.6)
    ax.annotate(
        f"Single-GPU reference\n(+{SINGLE_GPU_PENALTY}ms)",
        xy=(len(flag_conds) - 0.7, SINGLE_GPU_PENALTY),
        xytext=(len(flag_conds) - 0.7, 50),
        fontsize=8, color=C_BARE, ha="center", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_BARE, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_BARE, alpha=0.85))

    for i, (xp, m) in enumerate(zip(x, penalty_means)):
        ax.text(xp, m + 3, f"+{m:.0f}ms", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("First-Request Penalty (ms)")
    ax.set_ylim(0, 220)
    ax.set_title("Cold-Start Penalty Does Not Compound with GPU Count",
                 fontsize=11, fontweight="bold", pad=10)

    out = OUT_DIR / "multi_gpu_cold_start_penalty.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    records = load_records()
    print(f"  {len(records)} records loaded")

    print("\nFigure 1: DVFS Decay Curve...")
    plot_decay_curve(records)

    print("\nFigure 2: Multi-GPU Parking Tax Bars...")
    plot_parking_tax_bars(records)

    print("\nFigure 3: Cold-Start Penalty...")
    plot_cold_start_penalty(records)

    print("\nDone!")


if __name__ == "__main__":
    main()
