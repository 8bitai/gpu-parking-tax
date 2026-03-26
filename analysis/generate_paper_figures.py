#!/usr/bin/env python3
"""
Generate paper figures using experiment data from all three GPU architectures.

Data files (n=40 per phase, 20-min recording):
  - H100: h100_dose_response.jsonl
  - A100: a100_dose_response.jsonl
  - L40S: l40s_dose_response.jsonl
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Canonical experiment files (n=40 per phase, 20-min recording)
DATA_FILES = {
    "H100 (HBM3)": "data/experiments/h100_dose_response.jsonl",
    "A100 (HBM2e)": "data/experiments/a100_dose_response.jsonl",
    "L40S (GDDR6)": "data/experiments/l40s_dose_response.jsonl",
}

GPU_COLORS = {
    "H100 (HBM3)": "#3b82f6",
    "A100 (HBM2e)": "#10b981",
    "L40S (GDDR6)": "#f59e0b",
}

GPU_MARKERS = {
    "H100 (HBM3)": "o",
    "A100 (HBM2e)": "s",
    "L40S (GDDR6)": "D",
}


def load_experiment(filepath):
    """Load JSONL experiment file and compute phase stats."""
    rows = []
    with open(filepath) as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    df = pd.DataFrame(rows)

    phase_stats = []
    for phase, g in df.groupby("phase"):
        if "_pre" in str(phase) or "_post" in str(phase):
            continue
        phase_stats.append({
            "phase": phase,
            "target_vram_gb": g["target_vram_gb"].iloc[0],
            "power_mean": g["power_w"].mean(),
            "power_std": g["power_w"].std(),
            "power_min": g["power_w"].min(),
            "power_max": g["power_w"].max(),
            "temp_mean": g["gpu_temp_c"].mean(),
            "mem_temp_mean": g["mem_temp_c"].mean() if "mem_temp_c" in g and g["mem_temp_c"].notna().any() else np.nan,
            "sm_clock_mean": g["sm_clock_mhz"].mean(),
            "n": len(g),
        })

    sdf = pd.DataFrame(phase_stats).sort_values("target_vram_gb")
    bare = sdf[sdf["phase"] == "bare_idle"]
    cuda = sdf[sdf["phase"] != "bare_idle"].sort_values("target_vram_gb")

    # Regression
    if len(cuda) >= 3:
        slope, intercept, r_val, p_val, se = stats.linregress(
            cuda["target_vram_gb"], cuda["power_mean"]
        )
    else:
        slope = intercept = r_val = p_val = se = np.nan

    return {
        "all": sdf,
        "bare": bare,
        "cuda": cuda,
        "regression": {
            "slope": slope, "intercept": intercept,
            "r_squared": r_val**2, "p_value": p_val, "se": se,
        },
    }


def cross_architecture_dose_response():
    """Three-panel dose-response: one panel per GPU, absolute power scale."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (gpu_label, filepath) in zip(axes, DATA_FILES.items()):
        data = load_experiment(filepath)
        bare = data["bare"]
        cuda = data["cuda"]
        reg = data["regression"]
        color = GPU_COLORS[gpu_label]
        marker = GPU_MARKERS[gpu_label]

        # Bare idle point (far left, separate)
        if len(bare) > 0:
            bp = bare.iloc[0]
            ax.scatter([-3], [bp["power_mean"]], marker=marker, s=120,
                      c="white", edgecolors=color, linewidths=2, zorder=10,
                      label=f"Bare idle ({bp['power_mean']:.1f}W)")

        # CUDA phases
        ax.errorbar(cuda["target_vram_gb"], cuda["power_mean"],
                   yerr=cuda["power_std"], fmt=f"{marker}-", color=color,
                   markersize=8, capsize=4, linewidth=2,
                   markeredgecolor="black", markeredgewidth=0.5, zorder=5,
                   label=f"CUDA active")

        # Regression line
        if not np.isnan(reg["slope"]):
            x_fit = np.linspace(-1, cuda["target_vram_gb"].max() + 5, 100)
            y_fit = reg["intercept"] + reg["slope"] * x_fit
            ax.plot(x_fit, y_fit, "--", color="gray", alpha=0.5, linewidth=1,
                   label=f"β={reg['slope']:.3f} W/GB")

        # DVFS step annotation
        if len(bare) > 0 and len(cuda) > 0:
            bp_power = bare.iloc[0]["power_mean"]
            ctx_power = cuda.iloc[0]["power_mean"]
            overhead = ctx_power - bp_power
            mid_y = (bp_power + ctx_power) / 2
            ax.annotate(f"+{overhead:.1f}W\nDVFS step",
                       xy=(2, mid_y), fontsize=8, fontweight="bold",
                       ha="left", va="center", color=color,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                edgecolor=color, alpha=0.9))

        ax.set_xlabel("VRAM Allocation (GB)")
        ax.set_ylabel("Idle Power (W)")
        ax.set_title(gpu_label, fontweight="bold")
        ax.legend(fontsize=7, loc="center right")

        # Set y-axis to show both bare and CUDA with context
        if len(bare) > 0:
            y_min = bare.iloc[0]["power_mean"] - 10
            y_max = cuda["power_mean"].max() + 15
            ax.set_ylim(y_min, y_max)

    fig.suptitle("The Model Parking Tax: Cross-Architecture Dose-Response",
                fontweight="bold", fontsize=12, y=1.03)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"cross_architecture_dose_response.{ext}")
        fig.savefig(PAPER_DIR / f"cross_architecture_dose_response.{ext}")
    plt.close(fig)
    print("  Saved cross_architecture_dose_response")


def parking_tax_decomposition():
    """Stacked bar: base idle + CUDA context + VRAM component per GPU."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    gpu_labels = []
    base_vals = []
    context_vals = []
    vram_vals = []
    colors_ctx = []

    for gpu_label, filepath in DATA_FILES.items():
        data = load_experiment(filepath)
        bare = data["bare"]
        cuda = data["cuda"]
        reg = data["regression"]

        if len(bare) == 0 or len(cuda) == 0:
            continue

        bp = bare.iloc[0]["power_mean"]
        ctx = cuda.iloc[0]["power_mean"]
        max_vram = cuda["target_vram_gb"].max()

        # VRAM contribution = slope * max_vram (essentially 0)
        vram_contribution = abs(reg["slope"]) * max_vram if not np.isnan(reg["slope"]) else 0
        context_contribution = ctx - bp

        gpu_labels.append(gpu_label)
        base_vals.append(bp)
        context_vals.append(context_contribution)
        vram_vals.append(vram_contribution)
        colors_ctx.append(GPU_COLORS[gpu_label])

    x = np.arange(len(gpu_labels))
    width = 0.5

    # Use a single consistent color for CUDA context so legend is accurate
    ctx_color = "#4a7cc9"

    ax.bar(x, base_vals, width, label="Base idle power", color="#94a3b8",
           edgecolor="black", linewidth=0.5)
    ax.bar(x, context_vals, width, bottom=base_vals,
           label="CUDA context overhead (>99% of tax)", color=ctx_color,
           edgecolor="black", linewidth=0.5, alpha=0.85)

    # VRAM bars are <1W and invisible at this scale; skip legend entry,
    # just annotate in the figure caption instead.

    # Annotate totals and context overhead
    for i, (b, c, v, label) in enumerate(zip(base_vals, context_vals, vram_vals, gpu_labels)):
        total = b + c + v
        ax.text(i, total + 2, f"{total:.0f}W total", ha="center",
                fontsize=9, fontweight="bold")
        ax.text(i, b + c/2, f"+{c:.1f}W",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(gpu_labels, fontsize=9)
    ax.set_ylabel("Idle Power (W)")
    ax.set_title("Parking Tax Decomposition: CUDA Context Dominates on All Architectures",
                fontweight="bold")
    ax.legend(fontsize=8, loc="upper center")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"parking_tax_decomposition.{ext}")
        fig.savefig(PAPER_DIR / f"parking_tax_decomposition.{ext}")
    plt.close(fig)
    print("  Saved parking_tax_decomposition")


def vram_regression_detail():
    """Zoomed regression: CUDA-active phases only, ±2.5W scale."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (gpu_label, filepath) in zip(axes, DATA_FILES.items()):
        data = load_experiment(filepath)
        cuda = data["cuda"]
        reg = data["regression"]
        color = GPU_COLORS[gpu_label]
        marker = GPU_MARKERS[gpu_label]

        # Center around mean power for zoomed view
        mean_power = cuda["power_mean"].mean()

        ax.errorbar(cuda["target_vram_gb"], cuda["power_mean"],
                   yerr=cuda["power_std"], fmt=f"{marker}", color=color,
                   markersize=8, capsize=4, linewidth=0,
                   markeredgecolor="black", markeredgewidth=0.5, zorder=5)

        # Regression line
        if not np.isnan(reg["slope"]):
            x_fit = np.linspace(-2, cuda["target_vram_gb"].max() + 5, 100)
            y_fit = reg["intercept"] + reg["slope"] * x_fit
            ax.plot(x_fit, y_fit, "-", color=color, alpha=0.7, linewidth=2)

            # CI band
            ci_hi = (reg["intercept"] + 1.96 * reg["se"]) + (reg["slope"] + 1.96 * reg["se"]) * x_fit
            ci_lo = (reg["intercept"] - 1.96 * reg["se"]) + (reg["slope"] - 1.96 * reg["se"]) * x_fit
            ax.fill_between(x_fit, ci_lo, ci_hi, alpha=0.1, color=color)

        # Zero-slope reference
        ax.axhline(mean_power, color="gray", linestyle=":", alpha=0.5, linewidth=1)

        # Annotation
        p_str = f"p={reg['p_value']:.3f}" if reg['p_value'] > 0.001 else f"p={reg['p_value']:.1e}"
        ax.text(0.95, 0.95,
               f"β = {reg['slope']:.4f} W/GB\n"
               f"95% CI: [{reg['slope']-1.96*reg['se']:.4f}, {reg['slope']+1.96*reg['se']:.4f}]\n"
               f"R² = {reg['r_squared']:.4f}\n"
               f"{p_str}",
               transform=ax.transAxes, fontsize=7, va="top", ha="right",
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor=color, alpha=0.9))

        ax.set_xlabel("VRAM Allocation (GB)")
        ax.set_ylabel("Idle Power (W)")
        ax.set_title(gpu_label, fontweight="bold")
        ax.set_ylim(mean_power - 2.5, mean_power + 2.5)

    fig.suptitle("VRAM Dose-Response Detail (CUDA-Active Only, Zoomed ±2.5W)",
                fontweight="bold", fontsize=12, y=1.03)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"vram_regression_detail.{ext}")
        fig.savefig(PAPER_DIR / f"vram_regression_detail.{ext}")
    plt.close(fig)
    print("  Saved vram_regression_detail")


def main():
    print("Generating paper figures...")
    for label, path in DATA_FILES.items():
        print(f"  {label}: {path}")
    print()

    cross_architecture_dose_response()
    parking_tax_decomposition()
    vram_regression_detail()

    print(f"\nAll figures saved to {FIGURES_DIR}/ and {PAPER_DIR}/")


if __name__ == "__main__":
    main()
