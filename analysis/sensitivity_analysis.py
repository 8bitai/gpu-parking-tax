#!/usr/bin/env python3
"""
Sensitivity Analysis for Industry-Scale Impact Estimate (Section 6).

Varies key parameters (ρ, fleet size, P_park) to produce ranges
instead of point estimates. Generates a tornado chart and parameter
sweep table for the paper.

Usage:
    python analysis/sensitivity_analysis.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("analysis/results")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

HOURS_PER_YEAR = 8760

# ============================================================================
# BASE CASE PARAMETERS
# ============================================================================
BASE = {
    "N_gpus": 3_760_000,        # NVIDIA datacenter GPU shipments 2023
    "rho": 0.65,                # utilization (35% idle)
    "P_park_w": 40,             # fleet-weighted avg parking tax (W)
    "carbon_kg_per_kwh": 0.39,  # US grid average
}

# Parameter ranges for sensitivity sweep
RANGES = {
    "N_gpus": {
        "label": "Fleet size (M GPUs)",
        "low": 2_000_000, "high": 6_000_000,
        "low_label": "2.0M", "high_label": "6.0M",
    },
    "rho": {
        "label": "Utilization rate",
        "low": 0.50, "high": 0.80,
        "low_label": "50%", "high_label": "80%",
    },
    "P_park_w": {
        "label": "Avg parking tax (W)",
        "low": 26.3, "high": 66.4,     # A100 low to L40S high
        "low_label": "26.3 (A100)", "high_label": "66.4 (L40S)",
    },
    "carbon_kg_per_kwh": {
        "label": "Carbon intensity (kg/kWh)",
        "low": 0.05, "high": 0.50,
        "low_label": "0.05 (renewables)", "high_label": "0.50 (coal-heavy)",
    },
}


def compute_impact(N_gpus, rho, P_park_w, carbon_kg_per_kwh):
    """Compute annual energy and carbon impact."""
    n_idle = N_gpus * (1 - rho)
    energy_gwh = n_idle * P_park_w * HOURS_PER_YEAR / 1e9
    carbon_kt = energy_gwh * 1e6 * carbon_kg_per_kwh / 1e6  # GWh→kWh * kg/kWh → tons / 1000
    return energy_gwh, carbon_kt


def sensitivity_sweep():
    """One-at-a-time sensitivity analysis."""
    base_energy, base_carbon = compute_impact(**BASE)
    print(f"Base case: {base_energy:.0f} GWh/yr, {base_carbon:.0f} kT CO2/yr")

    results = {"base": {"energy_gwh": base_energy, "carbon_kt": base_carbon, **BASE}}
    tornado_data = []

    for param, info in RANGES.items():
        # Low scenario
        params_low = {**BASE, param: info["low"]}
        e_low, c_low = compute_impact(**params_low)

        # High scenario
        params_high = {**BASE, param: info["high"]}
        e_high, c_high = compute_impact(**params_high)

        # Note: for rho, higher utilization means LESS idle power, so the
        # relationship is inverted. We keep low/high as defined by parameter
        # value, not by impact direction.
        tornado_data.append({
            "param": param,
            "label": info["label"],
            "energy_low": e_low,
            "energy_high": e_high,
            "energy_range": abs(e_high - e_low),
            "carbon_low": c_low,
            "carbon_high": c_high,
            "low_label": info["low_label"],
            "high_label": info["high_label"],
        })

        results[param] = {
            "low_val": info["low"],
            "high_val": info["high"],
            "energy_low_gwh": e_low,
            "energy_high_gwh": e_high,
            "carbon_low_kt": c_low,
            "carbon_high_kt": c_high,
        }

        print(f"\n  {info['label']}:")
        print(f"    Low  ({info['low_label']}): {e_low:.0f} GWh, {c_low:.0f} kT CO2")
        print(f"    Base:           {base_energy:.0f} GWh, {base_carbon:.0f} kT CO2")
        print(f"    High ({info['high_label']}): {e_high:.0f} GWh, {c_high:.0f} kT CO2")

    # Overall range (all parameters at extremes)
    params_best = {
        "N_gpus": RANGES["N_gpus"]["low"],
        "rho": RANGES["rho"]["high"],          # high util = less idle
        "P_park_w": RANGES["P_park_w"]["low"],
        "carbon_kg_per_kwh": RANGES["carbon_kg_per_kwh"]["low"],
    }
    params_worst = {
        "N_gpus": RANGES["N_gpus"]["high"],
        "rho": RANGES["rho"]["low"],           # low util = more idle
        "P_park_w": RANGES["P_park_w"]["high"],
        "carbon_kg_per_kwh": RANGES["carbon_kg_per_kwh"]["high"],
    }
    e_best, c_best = compute_impact(**params_best)
    e_worst, c_worst = compute_impact(**params_worst)

    print(f"\n  Combined extremes:")
    print(f"    Best case:  {e_best:.0f} GWh, {c_best:.0f} kT CO2")
    print(f"    Worst case: {e_worst:.0f} GWh, {c_worst:.0f} kT CO2")

    results["combined"] = {
        "energy_best_gwh": e_best,
        "energy_worst_gwh": e_worst,
        "carbon_best_kt": c_best,
        "carbon_worst_kt": c_worst,
    }

    return results, tornado_data, base_energy


def plot_tornado(tornado_data, base_energy):
    """Tornado chart showing parameter sensitivity."""
    # Sort by energy range (most sensitive first)
    tornado_data = sorted(tornado_data, key=lambda x: x["energy_range"])

    fig, ax = plt.subplots(figsize=(8, 3.5))

    y_pos = np.arange(len(tornado_data))
    labels = [d["label"] for d in tornado_data]

    for i, d in enumerate(tornado_data):
        left = min(d["energy_low"], d["energy_high"])
        width = abs(d["energy_high"] - d["energy_low"])

        bar = ax.barh(i, width, left=left, height=0.6,
                      color="#3b82f6", alpha=0.7, edgecolor="black", linewidth=0.5)

        # Annotate low/high values
        ax.text(d["energy_low"], i, f" {d['low_label']}",
                ha="right" if d["energy_low"] > base_energy else "left",
                va="center", fontsize=7, color="#666")
        ax.text(d["energy_high"], i, f" {d['high_label']}",
                ha="left" if d["energy_high"] > base_energy else "right",
                va="center", fontsize=7, color="#666")

    ax.axvline(base_energy, color="red", linestyle="--", linewidth=1.5,
               label=f"Base case ({base_energy:.0f} GWh/yr)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Annual Parking Energy (GWh/yr)")
    ax.set_title("Sensitivity Analysis: Industry-Scale Parking Tax Estimate",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_sensitivity_tornado.pdf")
    fig.savefig(FIGURES_DIR / "fig_sensitivity_tornado.png")
    plt.close(fig)
    print(f"\nSaved tornado chart to {FIGURES_DIR / 'fig_sensitivity_tornado.png'}")


def plot_2d_sweep():
    """2D heatmap: utilization vs fleet size → energy."""
    rhos = np.linspace(0.50, 0.85, 15)
    fleet_sizes = np.linspace(2e6, 6e6, 15)

    energy_grid = np.zeros((len(rhos), len(fleet_sizes)))
    for i, rho in enumerate(rhos):
        for j, n in enumerate(fleet_sizes):
            e, _ = compute_impact(n, rho, BASE["P_park_w"], BASE["carbon_kg_per_kwh"])
            energy_grid[i, j] = e

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(energy_grid, origin="lower", aspect="auto",
                   extent=[fleet_sizes[0]/1e6, fleet_sizes[-1]/1e6,
                           rhos[0]*100, rhos[-1]*100],
                   cmap="YlOrRd")

    cbar = fig.colorbar(im, ax=ax, label="Annual Energy (GWh/yr)")

    # Mark base case
    ax.plot(BASE["N_gpus"]/1e6, BASE["rho"]*100, "k*", markersize=15,
            markeredgecolor="white", markeredgewidth=1.5,
            label=f"Base case ({compute_impact(**BASE)[0]:.0f} GWh)")

    ax.set_xlabel("Fleet Size (M GPUs)")
    ax.set_ylabel("Utilization Rate (%)")
    ax.set_title("Parking Tax Energy: Utilization vs Fleet Size\n"
                 f"(P_park = {BASE['P_park_w']}W fleet average)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_sensitivity_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "fig_sensitivity_heatmap.png")
    plt.close(fig)
    print(f"Saved heatmap to {FIGURES_DIR / 'fig_sensitivity_heatmap.png'}")


def main():
    print("=" * 70)
    print("SENSITIVITY ANALYSIS: INDUSTRY-SCALE IMPACT ESTIMATE")
    print("=" * 70)

    results, tornado_data, base_energy = sensitivity_sweep()
    plot_tornado(tornado_data, base_energy)
    plot_2d_sweep()

    # Save results
    output_file = RESULTS_DIR / "sensitivity_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    # Print paper-ready summary
    combined = results["combined"]
    print(f"\n{'=' * 70}")
    print("PAPER-READY SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Base estimate: {results['base']['energy_gwh']:.0f} GWh/yr")
    print(f"  Plausible range: {combined['energy_best_gwh']:.0f}–{combined['energy_worst_gwh']:.0f} GWh/yr")
    print(f"  Carbon range: {combined['carbon_best_kt']:.0f}–{combined['carbon_worst_kt']:.0f} kT CO2/yr")


if __name__ == "__main__":
    main()
