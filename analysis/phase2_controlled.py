#!/usr/bin/env python3
"""
Phase 2 Analysis: Controlled VRAM Loading Experiment.

Loads experiment data from Phase 2, segments by phase, computes the
dose-response curve (VRAM GB → idle power W), runs statistical tests,
and generates the definitive figures for the paper.

Usage:
    # Analyze a single experiment run
    python phase2_controlled.py --experiment data/experiments/h100_dose_response.jsonl

    # Analyze all experiment runs
    python phase2_controlled.py --all

    # Combine with Phase 1 observational data
    python phase2_controlled.py --experiment ... --with-phase1
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as t_dist
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# STYLE
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
})

FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("analysis/results")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

C_BARE = "#3b82f6"
C_CUDA = "#ef4444"
C_FIT = "#8b5cf6"


# ============================================================================
# DATA LOADING
# ============================================================================
def load_experiment(exp_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load experiment JSONL and manifest."""
    experiment_log = exp_dir / "experiment.jsonl"
    manifest_file = exp_dir / "manifest.json"

    with open(manifest_file) as f:
        manifest = json.load(f)

    rows = []
    with open(experiment_log) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"Loaded {len(df)} samples from {exp_dir}")
    print(f"  GPU: {manifest.get('gpu_info', {}).get('gpu_name', 'unknown')}")
    print(f"  UUID: {manifest.get('gpu_info', {}).get('uuid', 'unknown')}")
    print(f"  Phases: {len(manifest.get('phases', []))}")
    print(f"  VRAM levels: {manifest.get('vram_levels_gb', [])}")

    return df, manifest


def load_all_experiments() -> list[tuple[pd.DataFrame, dict, Path]]:
    """Load all experiment runs."""
    exp_base = Path("data/experiments")
    experiments = []
    for exp_dir in sorted(exp_base.glob("phase2_*")):
        if (exp_dir / "manifest.json").exists():
            df, manifest = load_experiment(exp_dir)
            experiments.append((df, manifest, exp_dir))
    return experiments


# ============================================================================
# ANALYSIS
# ============================================================================
def analyze_experiment(df: pd.DataFrame, manifest: dict) -> dict:
    """Analyze a single experiment run."""
    print("\n" + "=" * 70)
    print("PHASE 2 ANALYSIS: CONTROLLED VRAM LOADING")
    print("=" * 70)

    gpu_id = manifest.get("gpu_id", 0)
    gpu_name = manifest.get("gpu_info", {}).get("gpu_name", "H100")

    # Group by phase
    phases = df.groupby("phase")
    results = {"gpu_id": gpu_id, "gpu_name": gpu_name, "phases": {}}

    print(f"\n{'Phase':<25} {'VRAM(GB)':>9} {'N':>5} {'Power':>12} {'Temp':>10} "
          f"{'MemTemp':>10} {'SMClk':>10} {'MemUsed(MB)':>12}")
    print("-" * 100)

    phase_stats = []
    for phase_name, group in phases:
        # Skip pre/post snapshots
        if "_pre" in str(phase_name) or "_post" in str(phase_name):
            continue

        target_vram = group["target_vram_gb"].iloc[0] if "target_vram_gb" in group else 0
        n = len(group)

        pwr = group["power_w"].dropna()
        tmp = group["gpu_temp_c"].dropna()
        mtmp = group["mem_temp_c"].dropna()
        clk = group["sm_clock_mhz"].dropna()
        mem_used = group["mem_used_mb"].dropna()

        stat = {
            "phase": phase_name,
            "target_vram_gb": float(target_vram),
            "n_samples": n,
            "power_mean": pwr.mean(),
            "power_std": pwr.std(),
            "power_median": pwr.median(),
            "power_p5": pwr.quantile(0.05),
            "power_p95": pwr.quantile(0.95),
            "temp_mean": tmp.mean(),
            "temp_std": tmp.std(),
            "mem_temp_mean": mtmp.mean() if len(mtmp) > 0 else np.nan,
            "mem_temp_std": mtmp.std() if len(mtmp) > 0 else np.nan,
            "sm_clock_mean": clk.mean(),
            "mem_used_mean": mem_used.mean(),
            "mem_used_std": mem_used.std(),
        }
        phase_stats.append(stat)
        results["phases"][phase_name] = stat

        print(f"{phase_name:<25} {target_vram:>8.0f}G {n:>5} "
              f"{pwr.mean():>6.2f}±{pwr.std():>4.2f}W "
              f"{tmp.mean():>5.1f}±{tmp.std():.1f}°C "
              f"{mtmp.mean():>5.1f}±{mtmp.std():.1f}°C " if len(mtmp) > 0 else "" +
              f"{clk.mean():>7.0f}MHz "
              f"{mem_used.mean():>8.0f}±{mem_used.std():.0f}")

    stats_df = pd.DataFrame(phase_stats).sort_values("target_vram_gb")

    # ---- Dose-Response Analysis ----
    print("\n" + "-" * 70)
    print("DOSE-RESPONSE: VRAM (GB) → Idle Power (W)")
    print("-" * 70)

    # Separate bare idle from CUDA phases
    cuda_phases = stats_df[stats_df["phase"] != "bare_idle"].copy()
    bare_phase = stats_df[stats_df["phase"] == "bare_idle"]

    if len(bare_phase) > 0:
        bare_power = bare_phase.iloc[0]["power_mean"]
        print(f"\n  Bare idle baseline: {bare_power:.2f}W")

        # CUDA context overhead
        if len(cuda_phases) > 0:
            context_only = cuda_phases[cuda_phases["target_vram_gb"] == 0]
            if len(context_only) > 0:
                context_power = context_only.iloc[0]["power_mean"]
                context_overhead = context_power - bare_power
                print(f"  CUDA context (0 GB): {context_power:.2f}W")
                print(f"  CUDA context overhead: +{context_overhead:.2f}W")
                results["cuda_context_overhead_w"] = context_overhead
            else:
                context_overhead = 0
    else:
        bare_power = None
        context_overhead = 0

    # Linear regression on CUDA phases: Power = a + b * VRAM_GB
    if len(cuda_phases) >= 3:
        slope, intercept, r_val, p_val, se = stats.linregress(
            cuda_phases["target_vram_gb"], cuda_phases["power_mean"]
        )
        print(f"\n  Linear regression (CUDA phases):")
        print(f"    Power = {intercept:.2f} + {slope:.4f} × VRAM_GB")
        print(f"    Marginal cost: {slope:.4f} W/GB")
        print(f"    R² = {r_val**2:.4f}, p = {p_val:.4e}")
        print(f"    SE(slope) = {se:.4f}")
        print(f"    95% CI(slope): [{slope - 1.96*se:.4f}, {slope + 1.96*se:.4f}] W/GB")

        results["regression"] = {
            "intercept": intercept,
            "slope_w_per_gb": slope,
            "r_squared": r_val**2,
            "p_value": p_val,
            "slope_se": se,
            "slope_ci_low": slope - 1.96 * se,
            "slope_ci_high": slope + 1.96 * se,
        }

    # ---- TOST Equivalence Test ----
    # Formally demonstrate β ≈ 0 rather than just failing to reject β = 0.
    # Equivalence bound: ±0.1 W/GB (a 64GB model adding <6.4W would be negligible
    # compared to the 26-66W CUDA context overhead).
    if "regression" in results:
        reg = results["regression"]
        beta = reg["slope_w_per_gb"]
        se_beta = reg["slope_se"]
        n_points = len(cuda_phases)
        df_resid = n_points - 2  # degrees of freedom for simple linear regression
        equiv_bound = 0.1  # W/GB — physically meaningful threshold

        # Two One-Sided Tests (TOST)
        # H0_1: β <= -Δ  vs  H1_1: β > -Δ   (lower test)
        # H0_2: β >= +Δ  vs  H1_2: β < +Δ   (upper test)
        t_lower = (beta - (-equiv_bound)) / se_beta
        t_upper = (beta - equiv_bound) / se_beta
        p_lower = 1 - t_dist.cdf(t_lower, df_resid)  # one-sided: P(T > t)
        p_upper = t_dist.cdf(t_upper, df_resid)       # one-sided: P(T < t)
        p_tost = max(p_lower, p_upper)  # TOST p-value

        print(f"\n  TOST Equivalence Test (bound = ±{equiv_bound} W/GB):")
        print(f"    β = {beta:.4f} W/GB, SE = {se_beta:.4f}, df = {df_resid}")
        print(f"    Lower test (β > -{equiv_bound}): t = {t_lower:.3f}, p = {p_lower:.4e}")
        print(f"    Upper test (β < +{equiv_bound}): t = {t_upper:.3f}, p = {p_upper:.4e}")
        print(f"    TOST p-value: {p_tost:.4e}")
        if p_tost < 0.05:
            print(f"    ✓ EQUIVALENCE ESTABLISHED: |β| < {equiv_bound} W/GB (p = {p_tost:.4e})")
        else:
            print(f"    ✗ Cannot establish equivalence at α = 0.05 (p = {p_tost:.4e})")
            print(f"      (May need more samples or a wider equivalence bound)")

        results["tost"] = {
            "equivalence_bound_w_per_gb": equiv_bound,
            "t_lower": t_lower,
            "t_upper": t_upper,
            "p_lower": p_lower,
            "p_upper": p_upper,
            "p_tost": p_tost,
            "equivalent": p_tost < 0.05,
            "df": df_resid,
        }

    # ---- Pairwise Tests ----
    print("\n  Pairwise comparisons (adjacent VRAM levels):")
    cuda_sorted = cuda_phases.sort_values("target_vram_gb")
    pairwise = []
    for i in range(len(cuda_sorted) - 1):
        a = cuda_sorted.iloc[i]
        b = cuda_sorted.iloc[i + 1]

        # Get raw data for these phases
        a_data = df[df["phase"] == a["phase"]]["power_w"].dropna()
        b_data = df[df["phase"] == b["phase"]]["power_w"].dropna()

        diff = b["power_mean"] - a["power_mean"]
        vram_diff = b["target_vram_gb"] - a["target_vram_gb"]
        per_gb = diff / vram_diff if vram_diff > 0 else 0

        u_stat, p = stats.mannwhitneyu(b_data, a_data, alternative="two-sided")
        pooled = np.sqrt((a_data.std()**2 + b_data.std()**2) / 2)
        d = diff / pooled if pooled > 0 else 0

        pair = {
            "from_gb": a["target_vram_gb"],
            "to_gb": b["target_vram_gb"],
            "power_diff_w": diff,
            "w_per_gb": per_gb,
            "p_value": p,
            "cohens_d": d,
        }
        pairwise.append(pair)
        print(f"    {a['target_vram_gb']:.0f}GB → {b['target_vram_gb']:.0f}GB: "
              f"Δ={diff:+.3f}W, {per_gb:.4f} W/GB, p={p:.2e}, d={d:.3f}")

    results["pairwise"] = pairwise

    # ---- Thermal Dose-Response ----
    if "mem_temp_mean" in cuda_phases.columns:
        print("\n  HBM Temperature dose-response:")
        for _, row in cuda_sorted.iterrows():
            if not np.isnan(row["mem_temp_mean"]):
                print(f"    {row['target_vram_gb']:>5.0f} GB: "
                      f"{row['mem_temp_mean']:.1f}±{row['mem_temp_std']:.1f}°C")

    return results, stats_df


# ============================================================================
# FIGURES
# ============================================================================
def plot_dose_response(stats_df: pd.DataFrame, results: dict, exp_label: str = ""):
    """The money figure: VRAM GB vs idle power W."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Separate bare idle
    bare = stats_df[stats_df["phase"] == "bare_idle"]
    cuda = stats_df[stats_df["phase"] != "bare_idle"].sort_values("target_vram_gb")

    # (a) Power dose-response
    ax = axes[0]
    if len(bare) > 0:
        ax.axhline(bare.iloc[0]["power_mean"], color=C_BARE, linestyle="--",
                   alpha=0.7, label=f"Bare idle ({bare.iloc[0]['power_mean']:.1f}W)")

    ax.errorbar(cuda["target_vram_gb"], cuda["power_mean"],
               yerr=cuda["power_std"], fmt="o-", color=C_CUDA,
               markersize=8, capsize=4, linewidth=2,
               label="CUDA active", markeredgecolor="black", markeredgewidth=0.5)

    # Regression line
    if "regression" in results:
        reg = results["regression"]
        x_fit = np.linspace(0, cuda["target_vram_gb"].max() + 5, 100)
        y_fit = reg["intercept"] + reg["slope_w_per_gb"] * x_fit
        ax.plot(x_fit, y_fit, "--", color=C_FIT, alpha=0.7,
               label=f"OLS: {reg['slope_w_per_gb']:.3f} W/GB (R²={reg['r_squared']:.3f})")

    ax.set_xlabel("VRAM Allocation (GB)")
    ax.set_ylabel("Idle Power (W)")
    ax.set_title("(a) Power Dose-Response", fontweight="bold")
    ax.legend(fontsize=7)

    # (b) GPU Temperature
    ax = axes[1]
    if len(bare) > 0:
        ax.axhline(bare.iloc[0]["temp_mean"], color=C_BARE, linestyle="--",
                   alpha=0.7, label="Bare idle")
    ax.errorbar(cuda["target_vram_gb"], cuda["temp_mean"],
               yerr=cuda["temp_std"], fmt="s-", color="#f59e0b",
               markersize=7, capsize=4, linewidth=2,
               markeredgecolor="black", markeredgewidth=0.5)
    ax.set_xlabel("VRAM Allocation (GB)")
    ax.set_ylabel("GPU Temperature (°C)")
    ax.set_title("(b) Thermal Dose-Response", fontweight="bold")
    ax.legend(fontsize=7)

    # (c) HBM Temperature
    ax = axes[2]
    if len(bare) > 0 and not np.isnan(bare.iloc[0]["mem_temp_mean"]):
        ax.axhline(bare.iloc[0]["mem_temp_mean"], color=C_BARE, linestyle="--",
                   alpha=0.7, label="Bare idle")

    valid_mem = cuda.dropna(subset=["mem_temp_mean"])
    if len(valid_mem) > 0:
        ax.errorbar(valid_mem["target_vram_gb"], valid_mem["mem_temp_mean"],
                   yerr=valid_mem["mem_temp_std"], fmt="D-", color="#06b6d4",
                   markersize=7, capsize=4, linewidth=2,
                   markeredgecolor="black", markeredgewidth=0.5)
    ax.set_xlabel("VRAM Allocation (GB)")
    ax.set_ylabel("HBM Temperature (°C)")
    ax.set_title("(c) HBM Thermal Dose-Response", fontweight="bold")
    ax.legend(fontsize=7)

    fig.suptitle(f"Phase 2: Controlled VRAM Loading — Dose-Response Curves {exp_label}",
                fontweight="bold", fontsize=12, y=1.03)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig10_phase2_dose_response{exp_label}.pdf")
    fig.savefig(FIGURES_DIR / f"fig10_phase2_dose_response{exp_label}.png")
    plt.close(fig)
    print(f"  Saved fig10_phase2_dose_response{exp_label}")


def plot_timeseries(df: pd.DataFrame, exp_label: str = ""):
    """Full time-series of the experiment showing phase transitions."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    df_sorted = df.sort_values("timestamp")

    # Color by phase
    phases = df_sorted["phase"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))
    phase_colors = dict(zip(phases, colors))

    for phase_name in phases:
        mask = df_sorted["phase"] == phase_name
        subset = df_sorted[mask]
        vram = subset["target_vram_gb"].iloc[0] if "target_vram_gb" in subset else "?"
        label = f"{phase_name} ({vram}GB)"

        axes[0].scatter(subset["timestamp"], subset["power_w"],
                       s=5, c=[phase_colors[phase_name]], alpha=0.7, label=label)
        axes[1].scatter(subset["timestamp"], subset["gpu_temp_c"],
                       s=5, c=[phase_colors[phase_name]], alpha=0.7)
        if "mem_temp_c" in subset:
            axes[2].scatter(subset["timestamp"], subset["mem_temp_c"],
                           s=5, c=[phase_colors[phase_name]], alpha=0.7)

    axes[0].set_ylabel("Power (W)")
    axes[0].set_title("Power Draw", fontweight="bold")
    axes[0].legend(fontsize=6, ncol=3, loc="upper right")

    axes[1].set_ylabel("GPU Temp (°C)")
    axes[1].set_title("GPU Temperature", fontweight="bold")

    axes[2].set_ylabel("HBM Temp (°C)")
    axes[2].set_title("HBM Temperature", fontweight="bold")
    axes[2].set_xlabel("Time")

    fig.suptitle(f"Phase 2: Experiment Timeline {exp_label}",
                fontweight="bold", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig11_phase2_timeseries{exp_label}.pdf")
    fig.savefig(FIGURES_DIR / f"fig11_phase2_timeseries{exp_label}.png")
    plt.close(fig)
    print(f"  Saved fig11_phase2_timeseries{exp_label}")


def plot_parking_tax_combined(stats_df_phase2: pd.DataFrame, results_phase2: dict):
    """Combined figure: Phase 1 observational + Phase 2 controlled."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Try to load Phase 1 results
    phase1_file = RESULTS_DIR / "gpu_summary.csv"
    if phase1_file.exists():
        phase1 = pd.read_csv(phase1_file)
        ax.scatter(phase1["vram_gb"], phase1["power_mean"],
                  s=80, c="#94a3b8", marker="^", edgecolors="black",
                  linewidths=0.5, alpha=0.7, zorder=3,
                  label="Phase 1 (observational)")

    # Phase 2 controlled data
    bare = stats_df_phase2[stats_df_phase2["phase"] == "bare_idle"]
    cuda = stats_df_phase2[stats_df_phase2["phase"] != "bare_idle"].sort_values("target_vram_gb")

    if len(bare) > 0:
        ax.axhline(bare.iloc[0]["power_mean"], color=C_BARE, linestyle="--",
                   alpha=0.5, label=f"Bare idle ({bare.iloc[0]['power_mean']:.1f}W)")

    ax.errorbar(cuda["target_vram_gb"], cuda["power_mean"],
               yerr=cuda["power_std"], fmt="o-", color=C_CUDA,
               markersize=10, capsize=5, linewidth=2.5, zorder=5,
               label="Phase 2 (controlled)", markeredgecolor="black", markeredgewidth=0.5)

    # Regression
    if "regression" in results_phase2:
        reg = results_phase2["regression"]
        x_fit = np.linspace(-2, 70, 100)
        y_fit = reg["intercept"] + reg["slope_w_per_gb"] * x_fit
        ax.plot(x_fit, y_fit, "--", color=C_FIT, alpha=0.7, linewidth=1.5,
               label=f"OLS: {reg['slope_w_per_gb']:.3f} W/GB "
                     f"(95% CI: [{reg.get('slope_ci_low', 0):.3f}, {reg.get('slope_ci_high', 0):.3f}])")

    ax.set_xlabel("VRAM Allocation (GB)", fontsize=11)
    ax.set_ylabel("Idle Power (W)", fontsize=11)
    ax.set_title("The Model Parking Tax: VRAM Allocation vs. Idle Power\n"
                "H100 80GB HBM3 — Controlled Experiment + Observational Validation",
                fontweight="bold", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlim(-3, 75)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig12_combined_parking_tax.pdf")
    fig.savefig(FIGURES_DIR / "fig12_combined_parking_tax.png")
    plt.close(fig)
    print("  Saved fig12_combined_parking_tax")


# ============================================================================
# SUMMARY REPORT
# ============================================================================
def print_summary(results: dict, stats_df: pd.DataFrame):
    """Print a publication-ready summary of findings."""
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY — KEY FINDINGS")
    print("=" * 70)

    bare = stats_df[stats_df["phase"] == "bare_idle"]
    cuda = stats_df[stats_df["phase"] != "bare_idle"].sort_values("target_vram_gb")

    if len(bare) > 0:
        print(f"\n  1. BARE IDLE BASELINE: {bare.iloc[0]['power_mean']:.2f}W")

    if "cuda_context_overhead_w" in results:
        print(f"\n  2. CUDA CONTEXT OVERHEAD: +{results['cuda_context_overhead_w']:.2f}W")
        print(f"     (This is the fixed cost of having any inference server running)")

    if "regression" in results:
        reg = results["regression"]
        print(f"\n  3. MARGINAL VRAM COST: {reg['slope_w_per_gb']:.4f} W/GB")
        print(f"     95% CI: [{reg.get('slope_ci_low', 0):.4f}, {reg.get('slope_ci_high', 0):.4f}] W/GB")
        print(f"     R² = {reg['r_squared']:.4f}")

        if abs(reg['slope_w_per_gb']) < 0.1 and reg['r_squared'] < 0.3:
            print(f"\n  CONCLUSION: VRAM allocation has NEGLIGIBLE effect on idle power.")
            print(f"  The parking tax is dominated by the CUDA context ({results.get('cuda_context_overhead_w', '?'):.0f}W),")
            print(f"  not by model size. A 64GB model costs the same to park as a 1GB model.")
        else:
            total_cost_64gb = reg['slope_w_per_gb'] * 64
            print(f"\n  CONCLUSION: VRAM allocation adds {reg['slope_w_per_gb']:.3f}W per GB.")
            print(f"  A 64GB model adds {total_cost_64gb:.1f}W beyond the CUDA context cost.")

    if "tost" in results:
        tost = results["tost"]
        print(f"\n  4. EQUIVALENCE TEST (TOST):")
        print(f"     Bound: ±{tost['equivalence_bound_w_per_gb']} W/GB")
        print(f"     p(TOST) = {tost['p_tost']:.4e}")
        if tost['equivalent']:
            print(f"     ✓ Formally equivalent to zero: |β| < {tost['equivalence_bound_w_per_gb']} W/GB")
        else:
            print(f"     ✗ Cannot formally establish equivalence (insufficient power)")

    # Per-phase summary table (for paper)
    print(f"\n  Publication-ready table:")
    print(f"  {'VRAM (GB)':>10} {'Power (W)':>12} {'GPU T (°C)':>12} {'HBM T (°C)':>12} {'N':>5}")
    print(f"  {'-'*55}")
    if len(bare) > 0:
        b = bare.iloc[0]
        print(f"  {'bare':>10} {b['power_mean']:>7.2f}±{b['power_std']:.2f} "
              f"{b['temp_mean']:>7.1f}±{b['temp_std']:.1f} "
              f"{b['mem_temp_mean']:>7.1f}±{b['mem_temp_std']:.1f} "
              f"{b['n_samples']:>5}")
    for _, row in cuda.iterrows():
        mt = f"{row['mem_temp_mean']:>7.1f}±{row['mem_temp_std']:.1f}" if not np.isnan(row["mem_temp_mean"]) else "      N/A"
        print(f"  {row['target_vram_gb']:>10.0f} {row['power_mean']:>7.2f}±{row['power_std']:.2f} "
              f"{row['temp_mean']:>7.1f}±{row['temp_std']:.1f} "
              f"{mt} "
              f"{row['n_samples']:>5}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 2 Analysis")
    parser.add_argument("--experiment", type=str, default=None,
                       help="Path to experiment directory")
    parser.add_argument("--all", action="store_true",
                       help="Analyze all experiments")
    parser.add_argument("--with-phase1", action="store_true",
                       help="Include Phase 1 observational data in combined figure")
    args = parser.parse_args()

    if args.all:
        experiments = load_all_experiments()
        if not experiments:
            print("No experiments found in data/experiments/")
            print("Run the experiment first:")
            print("  python experiments/run_phase2.py --gpu 0 --phase-duration 1200")
            return

        for df, manifest, exp_dir in experiments:
            label = f"_gpu{manifest['gpu_id']}"
            results, stats_df = analyze_experiment(df, manifest)
            print_summary(results, stats_df)
            plot_dose_response(stats_df, results, label)
            plot_timeseries(df, label)

            if args.with_phase1:
                plot_parking_tax_combined(stats_df, results)

            # Save results
            output_file = RESULTS_DIR / f"phase2_results{label}.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Results saved to {output_file}")

    elif args.experiment:
        exp_dir = Path(args.experiment)
        df, manifest = load_experiment(exp_dir)
        results, stats_df = analyze_experiment(df, manifest)
        print_summary(results, stats_df)
        plot_dose_response(stats_df, results)
        plot_timeseries(df)

        if args.with_phase1:
            plot_parking_tax_combined(stats_df, results)

        output_file = RESULTS_DIR / "phase2_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")

    else:
        # Try to find the most recent experiment
        experiments = load_all_experiments()
        if experiments:
            df, manifest, exp_dir = experiments[-1]
            print(f"Using most recent experiment: {exp_dir}")
            results, stats_df = analyze_experiment(df, manifest)
            print_summary(results, stats_df)
            plot_dose_response(stats_df, results)
            plot_timeseries(df)
            plot_parking_tax_combined(stats_df, results)
        else:
            print("No experiments found. Run the experiment first:")
            print("  python experiments/dose_response.py --gpu 0")
            print("\nOr specify an experiment file:")
            print("  python analysis/phase2_controlled.py --experiment data/experiments/h100_dose_response.jsonl")


if __name__ == "__main__":
    main()
