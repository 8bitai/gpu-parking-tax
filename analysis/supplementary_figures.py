#!/usr/bin/env python3
"""
Publication-quality figures for "The Model Parking Tax" paper.
Generates all main and supplementary figures.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path

# ============================================================================
# STYLE CONFIG
# ============================================================================
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
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
C_BARE = "#3b82f6"      # blue
C_CUDA = "#ef4444"      # red
C_VRAM_LOW = "#22c55e"  # green
C_VRAM_MED = "#f59e0b"  # amber
C_VRAM_HIGH = "#8b5cf6" # purple
HOST_COLORS = {"node-2": "#f97316", "node-1": "#06b6d4"}


# ============================================================================
# LOAD DATA
# ============================================================================
def load_data():
    df = pq.read_table("data/processed/telemetry_wide.parquet").to_pandas()
    idle = df[df["DCGM_FI_DEV_GPU_UTIL"] == 0].copy()

    # Build GPU profiles
    profiles = []
    for uuid, g in idle.groupby("UUID"):
        profiles.append({
            "UUID": uuid,
            "gpu_slot": int(g["gpu"].mode().iloc[0]),
            "hostname": g["Hostname"].mode().iloc[0],
            "workload_type": g["workload_type"].mode().iloc[0],
            "vram_mb": g["DCGM_FI_DEV_FB_USED"].median(),
            "vram_gb": g["DCGM_FI_DEV_FB_USED"].median() / 1024,
            "sm_clock": g["DCGM_FI_DEV_SM_CLOCK"].median(),
            "power_mean": g["DCGM_FI_DEV_POWER_USAGE"].mean(),
            "power_std": g["DCGM_FI_DEV_POWER_USAGE"].std(),
            "power_median": g["DCGM_FI_DEV_POWER_USAGE"].median(),
            "temp_mean": g["DCGM_FI_DEV_GPU_TEMP"].mean(),
            "mem_temp_mean": g["DCGM_FI_DEV_MEMORY_TEMP"].mean(),
            "n": len(g),
        })
    prof_df = pd.DataFrame(profiles)
    prof_df["power_state"] = np.where(prof_df["sm_clock"] > 1000, "CUDA Active\n(1980 MHz)", "Bare Idle\n(345 MHz)")

    uuid_map = prof_df.set_index("UUID")
    idle["power_state"] = idle["UUID"].map(uuid_map["power_state"])
    idle["vram_gb"] = idle["UUID"].map(uuid_map["vram_gb"])
    idle["gpu_label"] = idle["UUID"].map(
        lambda u: f"GPU{int(uuid_map.loc[u, 'gpu_slot'])}\n({uuid_map.loc[u, 'hostname'][:10]})"
        if u in uuid_map.index else ""
    )

    return idle, prof_df


# ============================================================================
# FIGURE 1: Power State Comparison (Main Result)
# ============================================================================
def fig1_power_state_violin(idle, prof_df):
    """Violin plot: bare idle vs CUDA-active idle power distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    metrics = [
        ("DCGM_FI_DEV_POWER_USAGE", "Power Usage (W)", "W"),
        ("DCGM_FI_DEV_GPU_TEMP", "GPU Temperature (°C)", "°C"),
        ("DCGM_FI_DEV_MEMORY_TEMP", "HBM Temperature (°C)", "°C"),
    ]

    for ax, (col, title, unit) in zip(axes, metrics):
        bare = idle[idle.power_state == "Bare Idle\n(345 MHz)"][col].dropna()
        cuda = idle[idle.power_state == "CUDA Active\n(1980 MHz)"][col].dropna()

        parts = ax.violinplot(
            [bare, cuda], positions=[0, 1], showmedians=True,
            showextrema=False, widths=0.7
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor([C_BARE, C_CUDA][i])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        # Annotate means
        for i, (data, color) in enumerate([(bare, C_BARE), (cuda, C_CUDA)]):
            ax.scatter([i], [data.mean()], color=color, zorder=5, s=30,
                      edgecolors="black", linewidths=0.5, marker="D")
            ax.annotate(f"{data.mean():.1f}{unit}",
                       xy=(i, data.mean()), xytext=(0.15, 0),
                       textcoords="offset fontsize", fontsize=8, va="center")

        # Delta annotation
        diff = cuda.mean() - bare.mean()
        ax.annotate(f"Δ = +{diff:.1f}{unit}",
                   xy=(0.5, (bare.mean() + cuda.mean()) / 2),
                   fontsize=9, fontweight="bold", ha="center",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7",
                            edgecolor="#f59e0b", alpha=0.9))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Bare Idle\n(345 MHz)", "CUDA Active\n(1980 MHz)"])
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(unit)

    fig.suptitle("Figure 1: CUDA Context Overhead on Idle H100 GPUs",
                fontweight="bold", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_power_state_violin.pdf")
    fig.savefig(FIGURES_DIR / "fig1_power_state_violin.png")
    plt.close(fig)
    print("  Saved fig1_power_state_violin")


# ============================================================================
# FIGURE 2: Dose-Response Curve (VRAM vs Power)
# ============================================================================
def fig2_vram_dose_response(idle, prof_df):
    """Scatter + regression: VRAM allocation vs idle power per GPU."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left: All GPUs (both states) ---
    for _, row in prof_df.iterrows():
        color = HOST_COLORS.get(row.hostname, "gray")
        marker = "o" if "1980" in str(row.power_state) else "s"
        edge = "black" if "1980" in str(row.power_state) else "gray"
        ax1.scatter(row.vram_gb, row.power_mean, c=color, s=80,
                   marker=marker, edgecolors=edge, linewidths=1, zorder=5)
        ax1.annotate(f"GPU{row.gpu_slot}",
                    xy=(row.vram_gb, row.power_mean),
                    xytext=(5, 5), textcoords="offset points", fontsize=7)

    # Horizontal bands for power states
    bare = prof_df[prof_df.sm_clock < 1000]
    cuda = prof_df[prof_df.sm_clock > 1000]
    ax1.axhspan(bare.power_mean.min() - 5, bare.power_mean.max() + 5,
               alpha=0.1, color=C_BARE, label="Bare idle range")
    ax1.axhspan(cuda.power_mean.min() - 5, cuda.power_mean.max() + 5,
               alpha=0.1, color=C_CUDA, label="CUDA active range")

    # Legend for hosts
    for host, color in HOST_COLORS.items():
        ax1.scatter([], [], c=color, s=60, label=host[:15])
    ax1.scatter([], [], c="gray", marker="o", s=60, edgecolors="black", label="CUDA active")
    ax1.scatter([], [], c="gray", marker="s", s=60, edgecolors="gray", label="Bare idle")

    ax1.set_xlabel("VRAM Allocation (GB)")
    ax1.set_ylabel("Mean Idle Power (W)")
    ax1.set_title("(a) All GPUs by Power State", fontweight="bold")
    ax1.legend(fontsize=7, loc="center left")

    # --- Right: CUDA-active only, with regression ---
    cuda_df = prof_df[prof_df.sm_clock > 1000].copy()

    for _, row in cuda_df.iterrows():
        color = HOST_COLORS.get(row.hostname, "gray")
        ax2.errorbar(row.vram_gb, row.power_mean, yerr=row.power_std,
                    fmt="o", color=color, markersize=8, capsize=3,
                    elinewidth=1, markeredgecolor="black", markeredgewidth=0.5)
        ax2.annotate(f"GPU{row.gpu_slot}\n({row.workload_type[:8]})",
                    xy=(row.vram_gb, row.power_mean),
                    xytext=(8, 0), textcoords="offset points", fontsize=7,
                    va="center")

    # Regression line
    slope, intercept, r_val, p_val, se = stats.linregress(
        cuda_df.vram_gb, cuda_df.power_mean
    )
    x_range = np.linspace(0, 85, 100)
    ax2.plot(x_range, intercept + slope * x_range, "--", color="gray",
            alpha=0.7, linewidth=1.5,
            label=f"OLS: {slope:.3f} W/GB\nR²={r_val**2:.4f}, p={p_val:.3f}")

    # Per-host regression
    for host, color in HOST_COLORS.items():
        host_data = cuda_df[cuda_df.hostname == host]
        if len(host_data) >= 2:
            sl, ic, rv, pv, _ = stats.linregress(host_data.vram_gb, host_data.power_mean)
            x_h = np.linspace(host_data.vram_gb.min() - 5, host_data.vram_gb.max() + 5, 50)
            ax2.plot(x_h, ic + sl * x_h, "-", color=color, alpha=0.5, linewidth=1,
                    label=f"{host[:10]}: {sl:.3f} W/GB")

    ax2.set_xlabel("VRAM Allocation (GB)")
    ax2.set_ylabel("Mean Idle Power (W)")
    ax2.set_title("(b) CUDA-Active GPUs: VRAM vs Power", fontweight="bold")
    ax2.legend(fontsize=7)

    fig.suptitle("Figure 2: VRAM Allocation vs. Idle Power Draw",
                fontweight="bold", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_vram_dose_response.pdf")
    fig.savefig(FIGURES_DIR / "fig2_vram_dose_response.png")
    plt.close(fig)
    print("  Saved fig2_vram_dose_response")


# ============================================================================
# FIGURE 3: Per-GPU Power Distribution (Box + Strip)
# ============================================================================
def fig3_per_gpu_power_boxes(idle, prof_df):
    """Per-GPU box plots ordered by VRAM, colored by power state."""
    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Sort GPUs: bare idle first (by VRAM), then CUDA active (by VRAM)
    order = prof_df.sort_values(["sm_clock", "vram_gb"])
    positions = list(range(len(order)))

    box_data = []
    labels = []
    colors = []

    for i, (_, row) in enumerate(order.iterrows()):
        gpu_power = idle[idle.UUID == row.UUID]["DCGM_FI_DEV_POWER_USAGE"].dropna()
        box_data.append(gpu_power.values)
        labels.append(f"GPU{row.gpu_slot}\n{row.hostname[:10]}\n{row.vram_gb:.0f}GB")
        colors.append(C_BARE if row.sm_clock < 1000 else C_CUDA)

    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel("Idle Power (W)")
    ax.set_title("Figure 3: Per-GPU Idle Power Distribution (ordered by VRAM allocation)",
                fontweight="bold")

    # Separator line between power states
    n_bare = len(prof_df[prof_df.sm_clock < 1000])
    ax.axvline(n_bare - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(n_bare / 2 - 0.5, ax.get_ylim()[1], "Bare Idle (345 MHz)",
           ha="center", fontsize=9, color=C_BARE, fontweight="bold")
    ax.text(n_bare + (len(order) - n_bare) / 2 - 0.5, ax.get_ylim()[1],
           "CUDA Active (1980 MHz)",
           ha="center", fontsize=9, color=C_CUDA, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_per_gpu_power_boxes.pdf")
    fig.savefig(FIGURES_DIR / "fig3_per_gpu_power_boxes.png")
    plt.close(fig)
    print("  Saved fig3_per_gpu_power_boxes")


# ============================================================================
# FIGURE 4: Parking Tax Decomposition (Stacked Bar)
# ============================================================================
def fig4_parking_tax_decomposition(prof_df):
    """Stacked bar showing what fraction of idle power is base, context, VRAM."""
    fig, ax = plt.subplots(figsize=(7, 4))

    # H100 minimum power (no CUDA context): ~72W average from bare idle
    bare_power = prof_df[prof_df.sm_clock < 1000]["power_mean"].mean()
    cuda_active = prof_df[prof_df.sm_clock > 1000]

    categories = []
    base_vals = []
    context_vals = []
    vram_vals = []

    # Sort CUDA-active by VRAM
    for _, row in cuda_active.sort_values("vram_gb").iterrows():
        label = f"GPU{row.gpu_slot}\n({row.workload_type[:6]})\n{row.vram_gb:.0f}GB"
        categories.append(label)
        base_vals.append(bare_power)

        # Marginal VRAM effect estimate (~0.04 W/GB from within-host regression)
        vram_contribution = 0.04 * row.vram_gb
        context_contribution = row.power_mean - bare_power - vram_contribution
        if context_contribution < 0:
            context_contribution = row.power_mean - bare_power
            vram_contribution = 0

        context_vals.append(context_contribution)
        vram_vals.append(vram_contribution)

    x = np.arange(len(categories))
    width = 0.6

    ax.bar(x, base_vals, width, label=f"Base Idle ({bare_power:.0f}W)", color="#94a3b8")
    ax.bar(x, context_vals, width, bottom=base_vals,
           label="CUDA Context Overhead", color=C_CUDA, alpha=0.8)
    ax.bar(x, vram_vals, width,
           bottom=[b + c for b, c in zip(base_vals, context_vals)],
           label="VRAM Allocation (~0.04 W/GB)", color=C_VRAM_HIGH, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylabel("Idle Power (W)")
    ax.set_title("Figure 4: Idle Power Decomposition — The Parking Tax Breakdown",
                fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    # Annotate total for each
    for i, (b, c, v) in enumerate(zip(base_vals, context_vals, vram_vals)):
        total = b + c + v
        ax.text(i, total + 1, f"{total:.0f}W", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_parking_tax_decomposition.pdf")
    fig.savefig(FIGURES_DIR / "fig4_parking_tax_decomposition.png")
    plt.close(fig)
    print("  Saved fig4_parking_tax_decomposition")


# ============================================================================
# FIGURE 5: Temporal Stability
# ============================================================================
def fig5_temporal_stability(idle, prof_df):
    """Daily power time series per GPU, showing measurement stability."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    idle_copy = idle.copy()
    idle_copy["date"] = idle_copy["timestamp"].dt.date

    # Bare idle GPUs
    bare_uuids = prof_df[prof_df.sm_clock < 1000]["UUID"]
    for uuid in bare_uuids:
        row = prof_df[prof_df.UUID == uuid].iloc[0]
        g = idle_copy[idle_copy.UUID == uuid]
        daily = g.groupby("date")["DCGM_FI_DEV_POWER_USAGE"].mean()
        ax1.plot(daily.index, daily.values, "o-", markersize=3,
                label=f"GPU{row.gpu_slot} ({row.hostname[:10]}, {row.vram_gb:.0f}GB)",
                alpha=0.8)
    ax1.set_ylabel("Power (W)")
    ax1.set_title("(a) Bare Idle GPUs — Daily Mean Power", fontweight="bold")
    ax1.legend(fontsize=6, ncol=2)

    # CUDA active GPUs
    cuda_uuids = prof_df[prof_df.sm_clock > 1000]["UUID"]
    for uuid in cuda_uuids:
        row = prof_df[prof_df.UUID == uuid].iloc[0]
        g = idle_copy[idle_copy.UUID == uuid]
        daily = g.groupby("date")["DCGM_FI_DEV_POWER_USAGE"].mean()
        ax2.plot(daily.index, daily.values, "o-", markersize=3,
                label=f"GPU{row.gpu_slot} ({row.hostname[:10]}, {row.vram_gb:.0f}GB)",
                alpha=0.8)
    ax2.set_ylabel("Power (W)")
    ax2.set_title("(b) CUDA-Active GPUs — Daily Mean Power", fontweight="bold")
    ax2.legend(fontsize=6, ncol=2)

    plt.xticks(rotation=45)
    fig.suptitle("Figure 5: Temporal Stability of Idle Power Readings (18 Days)",
                fontweight="bold", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_temporal_stability.pdf")
    fig.savefig(FIGURES_DIR / "fig5_temporal_stability.png")
    plt.close(fig)
    print("  Saved fig5_temporal_stability")


# ============================================================================
# FIGURE 6: Thermal Profile
# ============================================================================
def fig6_thermal_profile(idle, prof_df):
    """GPU temp vs HBM temp scatter, sized by VRAM."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for _, row in prof_df.iterrows():
        color = C_BARE if row.sm_clock < 1000 else C_CUDA
        size = max(30, row.vram_gb * 2)
        ax.scatter(row.temp_mean, row.mem_temp_mean, s=size, c=color,
                  edgecolors="black", linewidths=0.5, alpha=0.8, zorder=5)
        ax.annotate(f"GPU{row.gpu_slot}\n{row.vram_gb:.0f}GB",
                   xy=(row.temp_mean, row.mem_temp_mean),
                   xytext=(6, 6), textcoords="offset points", fontsize=7)

    # Identity line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "--", color="gray", alpha=0.3, zorder=0)

    # Delta = HBM - GPU offset lines
    for delta in [4, 6, 8]:
        ax.plot(lims, [l + delta for l in lims], ":", color="gray", alpha=0.2)
        ax.text(lims[1] - 1, lims[1] + delta - 0.5, f"+{delta}°C",
               fontsize=7, color="gray", alpha=0.5)

    ax.scatter([], [], c=C_BARE, s=60, edgecolors="black", label="Bare Idle")
    ax.scatter([], [], c=C_CUDA, s=60, edgecolors="black", label="CUDA Active")
    ax.legend()

    ax.set_xlabel("GPU Die Temperature (°C)")
    ax.set_ylabel("HBM Temperature (°C)")
    ax.set_title("Figure 6: GPU vs HBM Temperature by Power State\n(marker size ∝ VRAM allocation)",
                fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_thermal_profile.pdf")
    fig.savefig(FIGURES_DIR / "fig6_thermal_profile.png")
    plt.close(fig)
    print("  Saved fig6_thermal_profile")


# ============================================================================
# FIGURE 7: Cold-Start Breakeven Chart
# ============================================================================
def fig7_cold_start_breakeven():
    """Breakeven analysis: parking cost vs cold-start cost over time."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    parking_w = 70.86  # CUDA context overhead
    electricity_rate = 0.08  # $/kWh

    # Cold start scenarios
    scenarios = {
        "Llama-70B\n(PyTorch, 45s)": {"load_s": 45, "gpu_cost_hr": 2.49},
        "Llama-70B\n(ServerlessLLM, 8s)": {"load_s": 8, "gpu_cost_hr": 2.49},
        "Qwen-30B\n(~your deploy, 25s)": {"load_s": 25, "gpu_cost_hr": 2.49},
        "Llama-8B\n(Run:ai, 5s)": {"load_s": 5, "gpu_cost_hr": 2.49},
    }

    # X axis: hours idle before next request
    hours = np.linspace(0, 24, 200)

    # Parking cost accumulates linearly
    parking_cost = parking_w / 1000 * electricity_rate * hours
    ax.plot(hours, parking_cost, "-", color=C_CUDA, linewidth=2.5,
           label=f"Parking cost ({parking_w:.0f}W × ${electricity_rate}/kWh)")

    colors = ["#8b5cf6", "#22c55e", "#f59e0b", "#06b6d4"]
    for (name, params), color in zip(scenarios.items(), colors):
        cold_cost = params["gpu_cost_hr"] * (params["load_s"] / 3600)
        ax.axhline(cold_cost, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax.text(24.5, cold_cost, f"{name}\n${cold_cost:.4f}", fontsize=7,
               va="center", color=color)

        # Find breakeven
        breakeven_h = cold_cost / (parking_w / 1000 * electricity_rate)
        if breakeven_h < 24:
            ax.axvline(breakeven_h, color=color, linestyle=":", alpha=0.3)
            ax.scatter([breakeven_h], [cold_cost], color=color, s=40, zorder=5)

    ax.set_xlabel("Hours Idle Before Next Request")
    ax.set_ylabel("Cost ($)")
    ax.set_title("Figure 7: Keep-Warm vs Cold-Start Cost Comparison\n(energy cost only, CoreWeave H100 @ $2.49/hr)",
                fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 24)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_cold_start_breakeven.pdf")
    fig.savefig(FIGURES_DIR / "fig7_cold_start_breakeven.png")
    plt.close(fig)
    print("  Saved fig7_cold_start_breakeven")


# ============================================================================
# FIGURE 8: Effect Size Summary (Forest Plot)
# ============================================================================
def fig8_effect_summary(idle):
    """Forest plot of effect sizes for all metrics."""
    fig, ax = plt.subplots(figsize=(7, 5))

    bare = idle[idle.power_state == "Bare Idle\n(345 MHz)"]
    cuda = idle[idle.power_state == "CUDA Active\n(1980 MHz)"]

    metrics = [
        ("DCGM_FI_DEV_POWER_USAGE", "Power (W)"),
        ("DCGM_FI_DEV_GPU_TEMP", "GPU Temp (°C)"),
        ("DCGM_FI_DEV_MEMORY_TEMP", "HBM Temp (°C)"),
        ("DCGM_FI_DEV_SM_CLOCK", "SM Clock (MHz)"),
        ("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", "Energy (J)"),
    ]

    effects = []
    labels = []
    for col, label in metrics:
        b = bare[col].dropna()
        c = cuda[col].dropna()
        diff = c.mean() - b.mean()
        se = np.sqrt(c.std()**2 / len(c) + b.std()**2 / len(b))
        pooled = np.sqrt((b.std()**2 + c.std()**2) / 2)
        d = diff / pooled if pooled > 0 else 0
        effects.append({"label": label, "diff": diff, "se": se, "d": d,
                       "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se})
        labels.append(label)

    y_pos = list(range(len(effects)))
    for i, eff in enumerate(effects):
        color = C_CUDA if eff["d"] > 0.5 else "#94a3b8"
        ax.barh(i, eff["d"], color=color, alpha=0.7, height=0.6)
        ax.text(eff["d"] + 0.2, i, f"d={eff['d']:.2f}, Δ={eff['diff']:+.1f}",
               va="center", fontsize=8)

    ax.axvline(0.8, color="red", linestyle="--", alpha=0.3, label="Large effect (|d|>0.8)")
    ax.axvline(0.5, color="orange", linestyle="--", alpha=0.3, label="Medium effect (|d|>0.5)")
    ax.axvline(0.2, color="green", linestyle="--", alpha=0.3, label="Small effect (|d|>0.2)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (Bare Idle → CUDA Active)")
    ax.set_title("Figure 8: Effect Sizes — CUDA Context Impact on Idle Metrics",
                fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_effect_summary.pdf")
    fig.savefig(FIGURES_DIR / "fig8_effect_summary.png")
    plt.close(fig)
    print("  Saved fig8_effect_summary")


# ============================================================================
# FIGURE 9: Industry Extrapolation
# ============================================================================
def fig9_industry_extrapolation():
    """Bar chart of industry-scale parking tax under different assumptions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    parking_w = 70.86
    hours_yr = 8760

    # Scenarios: number of idle GPUs
    scenarios = {
        "Conservative\n(500K GPUs)": 500_000,
        "Moderate\n(1M GPUs)": 1_000_000,
        "NVIDIA 2023\n(1.1M, 30%)": 1_128_000,
        "Aggressive\n(2M GPUs)": 2_000_000,
    }

    # Energy in GWh/yr
    gwh = []
    labels = []
    for label, n in scenarios.items():
        e = parking_w * n * hours_yr / 1e9
        gwh.append(e)
        labels.append(label)

    bars = ax1.bar(labels, gwh, color=[C_BARE, "#f59e0b", C_CUDA, C_VRAM_HIGH],
                  alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, gwh):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 20,
                f"{val:.0f}\nGWh/yr", ha="center", fontsize=8, fontweight="bold")
    ax1.set_ylabel("Energy (GWh/year)")
    ax1.set_title("(a) Annual Energy: Model Parking Tax", fontweight="bold")

    # CO2 equivalent
    carbon_kg_per_kwh = 0.39  # US average
    co2_kt = [g * 1000 * carbon_kg_per_kwh / 1000 for g in gwh]
    bars2 = ax2.bar(labels, co2_kt, color=[C_BARE, "#f59e0b", C_CUDA, C_VRAM_HIGH],
                   alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars2, co2_kt):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 5,
                f"{val:.0f}\nkT CO₂", ha="center", fontsize=8, fontweight="bold")
    ax2.set_ylabel("CO₂ Emissions (kilotons/year)")
    ax2.set_title("(b) Carbon Footprint (US avg grid)", fontweight="bold")

    fig.suptitle("Figure 9: Industry-Scale Model Parking Tax Extrapolation",
                fontweight="bold", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig9_industry_extrapolation.pdf")
    fig.savefig(FIGURES_DIR / "fig9_industry_extrapolation.png")
    plt.close(fig)
    print("  Saved fig9_industry_extrapolation")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Generating publication figures...")
    idle, prof_df = load_data()

    fig1_power_state_violin(idle, prof_df)
    fig2_vram_dose_response(idle, prof_df)
    fig3_per_gpu_power_boxes(idle, prof_df)
    fig4_parking_tax_decomposition(prof_df)
    fig5_temporal_stability(idle, prof_df)
    fig6_thermal_profile(idle, prof_df)
    fig7_cold_start_breakeven()
    fig8_effect_summary(idle)
    fig9_industry_extrapolation()

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Formats: PDF (vector) + PNG (300 DPI)")


if __name__ == "__main__":
    main()
