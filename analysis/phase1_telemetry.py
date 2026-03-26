#!/usr/bin/env python3
"""
Phase 1 Analysis: The Model Parking Tax
========================================
Quantifying idle power cost by VRAM allocation on H100 GPUs.

Analyzes production telemetry to measure:
1. Power draw difference between bare-idle and model-loaded-idle GPUs
2. Marginal energy cost per GB of VRAM allocation
3. Thermal and clock behavior across deployment configurations
4. Statistical significance and effect sizes
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from pathlib import Path
import json
from itertools import combinations

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = Path("data/processed/telemetry_wide.parquet")
OUTPUT_DIR = Path("analysis/results")
FIGURES_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Metrics to analyze
POWER_COL = "DCGM_FI_DEV_POWER_USAGE"
GPU_TEMP_COL = "DCGM_FI_DEV_GPU_TEMP"
MEM_TEMP_COL = "DCGM_FI_DEV_MEMORY_TEMP"
SM_CLOCK_COL = "DCGM_FI_DEV_SM_CLOCK"
MEM_CLOCK_COL = "DCGM_FI_DEV_MEM_CLOCK"
GPU_UTIL_COL = "DCGM_FI_DEV_GPU_UTIL"
FB_USED_COL = "DCGM_FI_DEV_FB_USED"
ENERGY_COL = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"

KEY_METRICS = [POWER_COL, GPU_TEMP_COL, MEM_TEMP_COL, SM_CLOCK_COL]


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================
def load_and_prepare():
    """Load telemetry and create analysis-ready dataset."""
    print("=" * 70)
    print("PHASE 1: THE MODEL PARKING TAX — IDLE POWER ANALYSIS")
    print("=" * 70)

    df = pq.read_table(DATA_PATH).to_pandas()
    print(f"\nLoaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"Date range: {df.timestamp.min()} → {df.timestamp.max()}")
    print(f"GPUs: {df.UUID.nunique()}, Hosts: {df.Hostname.nunique()}")

    # --- Filter to pure idle (GPU util == 0) ---
    idle = df[df[GPU_UTIL_COL] == 0].copy()
    active_rows = len(df) - len(idle)
    print(f"\nFiltered to idle-only: {len(idle):,} rows ({len(idle)/len(df)*100:.2f}%)")
    print(f"Excluded {active_rows:,} rows with GPU_UTIL > 0")

    # --- Create GPU deployment profile ---
    gpu_profiles = []
    for uuid, g in idle.groupby("UUID"):
        profile = {
            "UUID": uuid,
            "gpu_slot": int(g["gpu"].mode().iloc[0]),
            "hostname": g["Hostname"].mode().iloc[0],
            "workload_type": g["workload_type"].mode().iloc[0],
            "vram_mb": g[FB_USED_COL].median(),
            "vram_gb": g[FB_USED_COL].median() / 1024,
            "sm_clock_median": g[SM_CLOCK_COL].median(),
            "power_median": g[POWER_COL].median(),
            "power_mean": g[POWER_COL].mean(),
            "power_std": g[POWER_COL].std(),
            "gpu_temp_median": g[GPU_TEMP_COL].median(),
            "mem_temp_median": g[MEM_TEMP_COL].median(),
            "n_samples": len(g),
        }
        gpu_profiles.append(profile)

    profiles_df = pd.DataFrame(gpu_profiles).sort_values(
        ["hostname", "gpu_slot"]
    )

    # --- Classify power state ---
    # Key insight: GPUs fall into two distinct clock states
    # 345 MHz = low-power idle (no CUDA context)
    # 1980 MHz = active idle (CUDA context loaded)
    profiles_df["power_state"] = np.where(
        profiles_df["sm_clock_median"] > 1000, "cuda_active", "bare_idle"
    )

    # --- Create VRAM tier ---
    def vram_tier(mb):
        if mb < 100:
            return "empty (< 0.1 GB)"
        elif mb < 5000:
            return "small (< 5 GB)"
        elif mb < 40000:
            return "medium (5-40 GB)"
        else:
            return "large (40+ GB)"

    profiles_df["vram_tier"] = profiles_df["vram_mb"].apply(vram_tier)

    # Add power state and VRAM tier back to idle df
    uuid_to_state = profiles_df.set_index("UUID")["power_state"].to_dict()
    uuid_to_tier = profiles_df.set_index("UUID")["vram_tier"].to_dict()
    uuid_to_vram = profiles_df.set_index("UUID")["vram_gb"].to_dict()
    idle["power_state"] = idle["UUID"].map(uuid_to_state)
    idle["vram_tier"] = idle["UUID"].map(uuid_to_tier)
    idle["vram_gb"] = idle["UUID"].map(uuid_to_vram)

    return df, idle, profiles_df


# ============================================================================
# ANALYSIS 1: GPU DEPLOYMENT MAP
# ============================================================================
def print_deployment_map(profiles_df):
    """Print the per-GPU deployment configuration table."""
    print("\n" + "=" * 70)
    print("TABLE 1: GPU DEPLOYMENT MAP")
    print("=" * 70)

    fmt = "{:<6} {:<22} {:<16} {:>10} {:>8} {:>6} {:>7} {:>8} {:>12}"
    print(fmt.format(
        "GPU#", "Host", "Workload", "VRAM(MB)", "Power", "Temp", "MemT",
        "SMClk", "PowerState"
    ))
    print("-" * 100)

    for _, row in profiles_df.iterrows():
        print(fmt.format(
            row["gpu_slot"],
            row["hostname"][:22],
            row["workload_type"][:16],
            f"{row['vram_mb']:.0f}",
            f"{row['power_median']:.1f}W",
            f"{row['gpu_temp_median']:.0f}°C",
            f"{row['mem_temp_median']:.0f}°C",
            f"{row['sm_clock_median']:.0f}",
            row["power_state"],
        ))

    print(f"\n  Bare idle (345 MHz): {(profiles_df.power_state == 'bare_idle').sum()} GPUs")
    print(f"  CUDA active (1980 MHz): {(profiles_df.power_state == 'cuda_active').sum()} GPUs")


# ============================================================================
# ANALYSIS 2: POWER STATE COMPARISON (bare idle vs CUDA active)
# ============================================================================
def analyze_power_states(idle, profiles_df):
    """Compare power between bare-idle and CUDA-active-idle GPUs."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: POWER STATE EFFECT (CUDA Context Overhead)")
    print("=" * 70)

    bare = idle[idle["power_state"] == "bare_idle"]
    cuda = idle[idle["power_state"] == "cuda_active"]

    results = {}

    for metric, label, unit in [
        (POWER_COL, "Power Usage", "W"),
        (GPU_TEMP_COL, "GPU Temperature", "°C"),
        (MEM_TEMP_COL, "Memory Temperature", "°C"),
    ]:
        bare_vals = bare[metric].dropna()
        cuda_vals = cuda[metric].dropna()

        # Mann-Whitney U test (non-parametric, doesn't assume normality)
        u_stat, p_val = stats.mannwhitneyu(
            cuda_vals, bare_vals, alternative="two-sided"
        )

        # Cohen's d effect size
        pooled_std = np.sqrt(
            (bare_vals.std()**2 + cuda_vals.std()**2) / 2
        )
        cohens_d = (cuda_vals.mean() - bare_vals.mean()) / pooled_std

        # 95% CI for mean difference (bootstrap-free, using CLT)
        diff_mean = cuda_vals.mean() - bare_vals.mean()
        diff_se = np.sqrt(
            cuda_vals.std()**2 / len(cuda_vals)
            + bare_vals.std()**2 / len(bare_vals)
        )
        ci_low = diff_mean - 1.96 * diff_se
        ci_high = diff_mean + 1.96 * diff_se

        print(f"\n  {label} ({unit}):")
        print(f"    Bare idle  (n={len(bare_vals):,}): "
              f"mean={bare_vals.mean():.2f}, median={bare_vals.median():.2f}, "
              f"std={bare_vals.std():.2f}")
        print(f"    CUDA active (n={len(cuda_vals):,}): "
              f"mean={cuda_vals.mean():.2f}, median={cuda_vals.median():.2f}, "
              f"std={cuda_vals.std():.2f}")
        print(f"    Δ = {diff_mean:+.2f} {unit}  "
              f"[95% CI: {ci_low:.2f} – {ci_high:.2f}]")
        print(f"    Mann-Whitney U = {u_stat:.0f}, p = {p_val:.2e}")
        print(f"    Cohen's d = {cohens_d:.3f} "
              f"({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")

        results[metric] = {
            "bare_mean": bare_vals.mean(),
            "bare_std": bare_vals.std(),
            "cuda_mean": cuda_vals.mean(),
            "cuda_std": cuda_vals.std(),
            "diff_mean": diff_mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_val,
            "cohens_d": cohens_d,
            "n_bare": len(bare_vals),
            "n_cuda": len(cuda_vals),
        }

    return results


# ============================================================================
# ANALYSIS 3: WITHIN CUDA-ACTIVE — VRAM EFFECT ON POWER
# ============================================================================
def analyze_vram_effect(idle, profiles_df):
    """Within CUDA-active GPUs, does VRAM allocation affect power?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: VRAM ALLOCATION EFFECT (Within CUDA-Active GPUs)")
    print("=" * 70)
    print("  Controlling for clock state: only GPUs at 1980 MHz SM clock")

    cuda_idle = idle[idle["power_state"] == "cuda_active"].copy()

    # Per-GPU summary for the CUDA-active group
    gpu_summary = []
    for uuid, g in cuda_idle.groupby("UUID"):
        prof = profiles_df[profiles_df.UUID == uuid].iloc[0]
        gpu_summary.append({
            "UUID": uuid[:16],
            "host": prof.hostname[:20],
            "gpu": prof.gpu_slot,
            "workload": prof.workload_type,
            "vram_gb": prof.vram_gb,
            "power_mean": g[POWER_COL].mean(),
            "power_std": g[POWER_COL].std(),
            "power_p5": g[POWER_COL].quantile(0.05),
            "power_p95": g[POWER_COL].quantile(0.95),
            "temp_mean": g[GPU_TEMP_COL].mean(),
            "mem_temp_mean": g[MEM_TEMP_COL].mean(),
            "n": len(g),
        })

    summary_df = pd.DataFrame(gpu_summary).sort_values("vram_gb")

    print("\n  CUDA-Active GPUs by VRAM allocation:")
    fmt = "    GPU {gpu:>2} | {host:<20} | {workload:<16} | VRAM={vram_gb:>6.1f}GB | Power={power_mean:>6.1f}W ±{power_std:.2f} | Temp={temp_mean:.1f}°C | MemT={mem_temp_mean:.1f}°C"
    for _, r in summary_df.iterrows():
        print(fmt.format(**r.to_dict()))

    # --- Split by host to control for node-level differences ---
    results = {}
    for host, host_df in cuda_idle.groupby("Hostname"):
        host_gpus = profiles_df[
            (profiles_df.hostname == host) & (profiles_df.power_state == "cuda_active")
        ]
        if len(host_gpus) < 2:
            continue

        print(f"\n  --- Host: {host} ({len(host_gpus)} CUDA-active GPUs) ---")

        for _, gpu in host_gpus.sort_values("vram_gb").iterrows():
            gpu_data = cuda_idle[cuda_idle.UUID == gpu.UUID]
            power = gpu_data[POWER_COL]
            print(f"    GPU {gpu.gpu_slot}: VRAM={gpu.vram_gb:.1f}GB, "
                  f"Power={power.mean():.2f}W ±{power.std():.2f}, "
                  f"[p5={power.quantile(0.05):.1f}, p95={power.quantile(0.95):.1f}]")

        # Pairwise comparisons within host
        host_gpu_list = host_gpus.sort_values("vram_gb")
        if len(host_gpu_list) >= 2:
            print(f"\n    Pairwise power comparisons (within {host[:20]}):")
            for (_, gpu_a), (_, gpu_b) in combinations(
                host_gpu_list.iterrows(), 2
            ):
                a_power = cuda_idle[cuda_idle.UUID == gpu_a.UUID][POWER_COL].dropna()
                b_power = cuda_idle[cuda_idle.UUID == gpu_b.UUID][POWER_COL].dropna()

                diff = b_power.mean() - a_power.mean()
                vram_diff = gpu_b.vram_gb - gpu_a.vram_gb

                u_stat, p_val = stats.mannwhitneyu(
                    b_power, a_power, alternative="two-sided"
                )
                pooled = np.sqrt((a_power.std()**2 + b_power.std()**2) / 2)
                d = diff / pooled if pooled > 0 else 0

                per_gb = diff / vram_diff if vram_diff != 0 else float("nan")

                print(f"    GPU{gpu_a.gpu_slot}({gpu_a.vram_gb:.0f}GB) vs "
                      f"GPU{gpu_b.gpu_slot}({gpu_b.vram_gb:.0f}GB): "
                      f"ΔPower={diff:+.2f}W, ΔVRAM={vram_diff:+.1f}GB, "
                      f"~{per_gb:.4f}W/GB, p={p_val:.2e}, d={d:.3f}")

                key = f"{host[:15]}_gpu{gpu_a.gpu_slot}_vs_gpu{gpu_b.gpu_slot}"
                results[key] = {
                    "host": host,
                    "gpu_a": gpu_a.gpu_slot,
                    "gpu_b": gpu_b.gpu_slot,
                    "vram_a_gb": gpu_a.vram_gb,
                    "vram_b_gb": gpu_b.vram_gb,
                    "power_a": a_power.mean(),
                    "power_b": b_power.mean(),
                    "power_diff_w": diff,
                    "vram_diff_gb": vram_diff,
                    "w_per_gb": per_gb,
                    "p_value": p_val,
                    "cohens_d": d,
                }

    # --- Correlation: VRAM vs Power across all CUDA-active GPUs ---
    print("\n  --- Correlation: VRAM Allocation vs Idle Power ---")
    r, p = stats.pearsonr(summary_df["vram_gb"], summary_df["power_mean"])
    rho, p_rho = stats.spearmanr(summary_df["vram_gb"], summary_df["power_mean"])
    print(f"    Pearson r = {r:.4f}, p = {p:.4e}")
    print(f"    Spearman ρ = {rho:.4f}, p = {p_rho:.4e}")

    # Linear regression: Power = a + b * VRAM_GB
    slope, intercept, r_val, p_lm, se = stats.linregress(
        summary_df["vram_gb"], summary_df["power_mean"]
    )
    print(f"    Linear fit: Power = {intercept:.2f} + {slope:.4f} × VRAM_GB")
    print(f"      → Marginal cost: {slope:.4f} W per GB")
    print(f"      → R² = {r_val**2:.4f}, p = {p_lm:.4e}, SE(slope) = {se:.4f}")

    results["regression"] = {
        "intercept": intercept,
        "slope_w_per_gb": slope,
        "r_squared": r_val**2,
        "p_value": p_lm,
        "slope_se": se,
        "pearson_r": r,
        "spearman_rho": rho,
    }

    return results, summary_df


# ============================================================================
# ANALYSIS 4: THERMAL ANALYSIS
# ============================================================================
def analyze_thermal(idle, profiles_df):
    """Analyze thermal differences: does VRAM loading affect HBM temperature?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: THERMAL BEHAVIOR")
    print("=" * 70)

    results = {}

    for state_label, state_val in [("bare_idle", "bare_idle"), ("cuda_active", "cuda_active")]:
        subset = idle[idle.power_state == state_val]
        print(f"\n  {state_label} GPUs:")
        for uuid, g in subset.groupby("UUID"):
            prof = profiles_df[profiles_df.UUID == uuid].iloc[0]
            gpu_t = g[GPU_TEMP_COL].dropna()
            mem_t = g[MEM_TEMP_COL].dropna()
            delta = mem_t.mean() - gpu_t.mean()
            print(f"    GPU{prof.gpu_slot} ({prof.hostname[:15]}, {prof.vram_gb:.0f}GB): "
                  f"GPU={gpu_t.mean():.1f}°C, HBM={mem_t.mean():.1f}°C, "
                  f"Δ(HBM-GPU)={delta:+.1f}°C")

    # Does HBM temp correlate with VRAM across all GPUs?
    cuda_gpus = profiles_df[profiles_df.power_state == "cuda_active"]
    if len(cuda_gpus) >= 3:
        r, p = stats.pearsonr(cuda_gpus["vram_gb"], cuda_gpus["mem_temp_median"])
        print(f"\n  VRAM vs HBM Temp (CUDA-active): Pearson r = {r:.4f}, p = {p:.4e}")
        results["vram_vs_hbm_temp"] = {"pearson_r": r, "p_value": p}

    return results


# ============================================================================
# ANALYSIS 5: TEMPORAL STABILITY
# ============================================================================
def analyze_temporal_stability(idle, profiles_df):
    """Check if power readings are stable over time or drift."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: TEMPORAL STABILITY OF IDLE POWER")
    print("=" * 70)

    idle = idle.copy()
    idle["date"] = idle["timestamp"].dt.date

    print("\n  Daily mean power by power state:")
    daily = idle.groupby(["date", "power_state"])[POWER_COL].agg(["mean", "std", "count"])
    for (date, state), row in daily.iterrows():
        print(f"    {date} | {state:12s} | {row['mean']:.2f}W ± {row['std']:.2f} | n={row['count']:,}")

    # Check for temporal drift in specific GPUs
    print("\n  Per-GPU daily power stability (std of daily means):")
    for uuid, g in idle.groupby("UUID"):
        prof = profiles_df[profiles_df.UUID == uuid].iloc[0]
        daily_means = g.groupby("date")[POWER_COL].mean()
        drift = daily_means.std()
        print(f"    GPU{prof.gpu_slot} ({prof.hostname[:15]}, {prof.vram_gb:.0f}GB): "
              f"daily mean range = {daily_means.min():.1f}–{daily_means.max():.1f}W, "
              f"σ(daily) = {drift:.3f}W")


# ============================================================================
# ANALYSIS 6: ECONOMIC MODEL
# ============================================================================
def economic_model(power_state_results, vram_results, profiles_df):
    """Calculate the parking tax in dollars and carbon."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: ECONOMIC & ENVIRONMENTAL MODEL")
    print("=" * 70)

    cuda_overhead_w = power_state_results[POWER_COL]["diff_mean"]
    cuda_overhead_ci = (
        power_state_results[POWER_COL]["ci_low"],
        power_state_results[POWER_COL]["ci_high"],
    )

    # Get slope from regression if available
    if "regression" in vram_results:
        marginal_w_per_gb = vram_results["regression"]["slope_w_per_gb"]
    else:
        marginal_w_per_gb = 0.0

    # Electricity rates ($/kWh)
    rates = {
        "cheap (Iowa/Oregon)": 0.04,
        "average US": 0.08,
        "California": 0.12,
        "Europe average": 0.15,
    }

    # Carbon intensity (kg CO2/kWh)
    carbon_intensities = {
        "US average": 0.39,
        "California": 0.21,
        "Germany": 0.35,
        "Renewables-heavy": 0.05,
    }

    hours_per_year = 8760

    print(f"\n  CUDA Context Overhead: {cuda_overhead_w:.2f}W "
          f"[95% CI: {cuda_overhead_ci[0]:.2f} – {cuda_overhead_ci[1]:.2f}W]")
    print(f"  Marginal VRAM cost: {marginal_w_per_gb:.4f} W/GB")

    # --- Per-GPU annual parking tax ---
    print("\n  --- Per-GPU Annual Parking Tax (CUDA context only) ---")
    kwh_per_year = cuda_overhead_w * hours_per_year / 1000
    for label, rate in rates.items():
        cost = kwh_per_year * rate
        print(f"    {label:25s}: {kwh_per_year:.1f} kWh/yr → ${cost:.2f}/yr per GPU")

    # --- For a 62GB LLM deployment ---
    total_parking_w = cuda_overhead_w + (marginal_w_per_gb * 62)
    total_kwh = total_parking_w * hours_per_year / 1000
    print(f"\n  --- 62GB LLM Deployment Parking Tax ---")
    print(f"    Total idle overhead: {total_parking_w:.2f}W ({cuda_overhead_w:.2f}W context + {marginal_w_per_gb * 62:.2f}W VRAM)")
    for label, rate in rates.items():
        cost = total_kwh * rate
        print(f"    {label:25s}: {total_kwh:.1f} kWh/yr → ${cost:.2f}/yr per GPU")

    # --- Fleet-level (your 14 GPUs) ---
    n_cuda_active = len(profiles_df[profiles_df.power_state == "cuda_active"])
    n_bare = len(profiles_df[profiles_df.power_state == "bare_idle"])
    fleet_overhead_w = cuda_overhead_w * n_cuda_active
    fleet_kwh = fleet_overhead_w * hours_per_year / 1000
    print(f"\n  --- Your Fleet (14 GPUs: {n_cuda_active} CUDA-active, {n_bare} bare-idle) ---")
    for label, rate in rates.items():
        cost = fleet_kwh * rate
        print(f"    {label:25s}: {fleet_kwh:.1f} kWh/yr → ${cost:.2f}/yr")

    # --- Industry extrapolation ---
    print("\n  --- Industry Scale Extrapolation ---")
    # NVIDIA shipped ~3.76M datacenter GPUs in 2023
    # Estimate 30-40% idle at any given time (from Flexera cloud waste reports)
    for idle_pct, n_gpus_total in [(0.3, 3_760_000), (0.4, 5_000_000)]:
        n_idle = int(n_gpus_total * idle_pct)
        ind_kwh = cuda_overhead_w * n_idle * hours_per_year / 1000
        ind_gwh = ind_kwh / 1e6
        for region, intensity in carbon_intensities.items():
            co2_tons = ind_kwh * intensity / 1000
            co2_kt = co2_tons / 1000
            print(f"    {n_idle:,} idle GPUs × {cuda_overhead_w:.0f}W × "
                  f"{region:20s}: {ind_gwh:.1f} GWh/yr, "
                  f"{co2_kt:.1f} kT CO₂/yr")

    return {
        "cuda_overhead_w": cuda_overhead_w,
        "marginal_w_per_gb": marginal_w_per_gb,
        "annual_kwh_per_gpu": kwh_per_year,
    }


# ============================================================================
# ANALYSIS 7: COLD-START BREAKEVEN
# ============================================================================
def cold_start_breakeven(econ_results):
    """At what request rate does keep-warm beat cold-start?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: COLD-START BREAKEVEN CALCULATOR")
    print("=" * 70)

    parking_w = econ_results["cuda_overhead_w"]

    # Cold start costs (from literature)
    cold_starts = {
        "Llama-70B (NVMe, standard PyTorch)": {"load_time_s": 45, "model_gb": 140},
        "Llama-70B (ServerlessLLM optimized)": {"load_time_s": 8, "model_gb": 140},
        "Llama-8B (Run:ai Model Streamer)": {"load_time_s": 5, "model_gb": 16},
        "Qwen-30B (~your deployment)": {"load_time_s": 25, "model_gb": 62},
    }

    # GPU cost ($/hr) for various providers
    gpu_costs_hr = {
        "on-prem (depreciation only)": 1.50,
        "AWS p5.xlarge": 4.50,
        "CoreWeave H100": 2.49,
        "Lambda H100": 2.49,
    }

    electricity_rate = 0.08  # $/kWh

    print(f"\n  Parking power cost: {parking_w:.2f}W = "
          f"${parking_w / 1000 * electricity_rate:.6f}/hr electricity")

    for scenario, params in cold_starts.items():
        load_s = params["load_time_s"]
        print(f"\n  Scenario: {scenario}")
        print(f"    Cold start latency: {load_s}s")

        # Energy cost of parking for 1 hour
        parking_cost_hr = parking_w / 1000 * electricity_rate  # $/hr

        for provider, gpu_hr_cost in gpu_costs_hr.items():
            # Cost of one cold start = GPU cost during load time
            cold_start_cost = gpu_hr_cost * (load_s / 3600)
            # How many hours of parking = 1 cold start cost?
            breakeven_hours = cold_start_cost / parking_cost_hr if parking_cost_hr > 0 else float("inf")

            print(f"    {provider:30s}: cold start = ${cold_start_cost:.4f}, "
                  f"breakeven = {breakeven_hours:.0f}h ({breakeven_hours/24:.1f} days)")

    print("\n  Interpretation: if you expect fewer than 1 request per [breakeven] hours,")
    print("  cold-starting is cheaper than keeping the model warm (energy cost only).")
    print("  Note: This only considers energy. GPU rental cost dominates in cloud settings.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    # Load data
    df, idle, profiles_df = load_and_prepare()

    # Run analyses
    print_deployment_map(profiles_df)

    power_state_results = analyze_power_states(idle, profiles_df)

    vram_results, gpu_summary = analyze_vram_effect(idle, profiles_df)

    thermal_results = analyze_thermal(idle, profiles_df)

    analyze_temporal_stability(idle, profiles_df)

    econ = economic_model(power_state_results, vram_results, profiles_df)

    cold_start_breakeven(econ)

    # --- Save results ---
    all_results = {
        "power_state_comparison": {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()}
            for k, v in power_state_results.items()
        },
        "vram_effect": {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in vram_results.items()
        },
        "economic": {k: float(v) for k, v in econ.items()},
    }

    output_file = OUTPUT_DIR / "phase1_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nResults saved to {output_file}")

    # Save GPU summary table
    gpu_summary.to_csv(OUTPUT_DIR / "gpu_summary.csv", index=False)
    profiles_df.to_csv(OUTPUT_DIR / "gpu_profiles.csv", index=False)
    print(f"GPU summary saved to {OUTPUT_DIR / 'gpu_summary.csv'}")

    # Save idle dataset for plotting
    print(f"\n{'=' * 70}")
    print("PHASE 1 ANALYSIS COMPLETE")
    print(f"{'=' * 70}")

    return idle, profiles_df, power_state_results, vram_results, gpu_summary, econ


if __name__ == "__main__":
    main()
