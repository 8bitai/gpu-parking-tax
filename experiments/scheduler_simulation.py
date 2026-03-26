#!/usr/bin/env python3
"""
Proof-of-Concept Parking Tax Scheduler.

Simulates keep-warm/evict decisions using the breakeven model T* from
the paper. Compares three policies against synthetic and real-world
request traces:

  1. Always-On:  Models stay loaded forever (baseline — industry default)
  2. Fixed-TTL:  Evict after a fixed idle timeout (e.g., 5 min, 15 min)
  3. Breakeven:  Evict when expected idle time > T* (our model)

Generates energy savings figures for the paper.

Usage:
    python experiments/parking_scheduler.py
    python experiments/parking_scheduler.py --trace data/traces/azure_llm.csv
    python experiments/parking_scheduler.py --output-dir figures
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path

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


# ============================================================================
# GPU CONFIGURATIONS (from our measurements)
# ============================================================================
GPU_CONFIGS = {
    "H100": {
        "P_base": 71.8,        # bare idle (W)
        "P_park": 49.9,        # CUDA context overhead (W)
        "P_idle": 121.7,       # P_base + P_park
        "P_load": 300,         # estimated loading power (W)
        "t_load_pytorch": 45,  # standard PyTorch load time for 70B (s)
        "t_load_fast": 8,      # ServerlessLLM load time (s)
        "tdp": 700,
    },
    "A100": {
        "P_base": 53.7,
        "P_park": 26.3,
        "P_idle": 80.0,
        "P_load": 250,
        "t_load_pytorch": 45,
        "t_load_fast": 8,
        "tdp": 300,
    },
    "L40S": {
        "P_base": 35.6,
        "P_park": 66.4,
        "P_idle": 102.0,
        "P_load": 280,
        "t_load_pytorch": 45,
        "t_load_fast": 8,
        "tdp": 350,
    },
}

# Model catalog (for multi-model scenarios)
MODEL_CATALOG = {
    "llama-70b":  {"size_gb": 140, "t_load": 45, "P_load": 300},
    "llama-30b":  {"size_gb": 62,  "t_load": 25, "P_load": 300},
    "llama-8b":   {"size_gb": 16,  "t_load": 8,  "P_load": 200},
    "mistral-7b": {"size_gb": 14,  "t_load": 7,  "P_load": 200},
    "qwen-72b":   {"size_gb": 144, "t_load": 48, "P_load": 300},
}


# ============================================================================
# TRACE GENERATION
# ============================================================================
def generate_poisson_trace(
    rate_per_hour: float,
    duration_hours: float,
    n_models: int = 1,
    seed: int = 42,
) -> list[dict]:
    """Generate Poisson request arrivals for n models."""
    rng = np.random.default_rng(seed)
    trace = []

    model_names = list(MODEL_CATALOG.keys())[:n_models]

    for model_name in model_names:
        # Each model gets its own arrival rate (split evenly or weighted)
        model_rate = rate_per_hour / n_models
        n_requests = rng.poisson(model_rate * duration_hours)
        arrivals = np.sort(rng.uniform(0, duration_hours * 3600, n_requests))

        for t in arrivals:
            trace.append({
                "timestamp": float(t),
                "model": model_name,
                "duration_s": float(rng.exponential(10)),  # inference duration
            })

    trace.sort(key=lambda x: x["timestamp"])
    return trace


def generate_bursty_trace(
    base_rate: float,
    burst_rate: float,
    burst_fraction: float,
    duration_hours: float,
    n_models: int = 1,
    seed: int = 42,
) -> list[dict]:
    """Generate bursty traffic: alternating quiet and busy periods."""
    rng = np.random.default_rng(seed)
    trace = []
    model_names = list(MODEL_CATALOG.keys())[:n_models]

    total_seconds = duration_hours * 3600
    t = 0

    while t < total_seconds:
        # Decide if this period is a burst
        is_burst = rng.random() < burst_fraction
        period_duration = rng.exponential(600)  # ~10 min periods
        rate = burst_rate if is_burst else base_rate

        # Generate arrivals in this period
        period_end = min(t + period_duration, total_seconds)
        n_arrivals = rng.poisson(rate * (period_end - t) / 3600)

        for _ in range(n_arrivals):
            arrival = rng.uniform(t, period_end)
            model = rng.choice(model_names)
            trace.append({
                "timestamp": float(arrival),
                "model": model,
                "duration_s": float(rng.exponential(10)),
            })

        t = period_end

    trace.sort(key=lambda x: x["timestamp"])
    return trace


def generate_diurnal_trace(
    peak_rate: float,
    duration_hours: float,
    n_models: int = 1,
    seed: int = 42,
) -> list[dict]:
    """Generate diurnal traffic pattern (sinusoidal, peaks during business hours)."""
    rng = np.random.default_rng(seed)
    trace = []
    model_names = list(MODEL_CATALOG.keys())[:n_models]

    # Use thinning algorithm for non-homogeneous Poisson process
    total_seconds = duration_hours * 3600
    t = 0

    while t < total_seconds:
        # Rate varies sinusoidally: peak at hour 14 (2 PM), trough at hour 2 (2 AM)
        hour_of_day = (t / 3600) % 24
        rate = peak_rate * (0.2 + 0.8 * max(0, np.sin(np.pi * (hour_of_day - 6) / 12)))

        # Next candidate arrival (using peak rate as upper bound)
        dt = rng.exponential(3600 / peak_rate)
        t += dt

        if t >= total_seconds:
            break

        # Accept/reject (thinning)
        hour_of_day = (t / 3600) % 24
        current_rate = peak_rate * (0.2 + 0.8 * max(0, np.sin(np.pi * (hour_of_day - 6) / 12)))
        if rng.random() < current_rate / peak_rate:
            model = rng.choice(model_names)
            trace.append({
                "timestamp": float(t),
                "model": model,
                "duration_s": float(rng.exponential(10)),
            })

    trace.sort(key=lambda x: x["timestamp"])
    return trace


# ============================================================================
# SCHEDULER POLICIES
# ============================================================================
@dataclass
class SchedulerMetrics:
    """Track energy and latency for a scheduling policy."""
    total_energy_wh: float = 0.0
    total_cold_starts: int = 0
    total_requests: int = 0
    total_idle_time_s: float = 0.0
    total_warm_time_s: float = 0.0
    total_cold_start_latency_s: float = 0.0
    cold_start_energy_wh: float = 0.0
    idle_energy_wh: float = 0.0


def simulate_always_on(trace, gpu_config, duration_s):
    """Always-on policy: model stays loaded for entire duration."""
    metrics = SchedulerMetrics()
    metrics.total_requests = len(trace)
    metrics.total_warm_time_s = duration_s

    # Energy = P_idle * total_time (model always loaded)
    metrics.idle_energy_wh = gpu_config["P_idle"] * duration_s / 3600
    metrics.total_energy_wh = metrics.idle_energy_wh
    metrics.total_cold_starts = 1  # initial load only

    return metrics


def simulate_fixed_ttl(trace, gpu_config, duration_s, ttl_s):
    """Fixed TTL: evict model after ttl_s of idle time."""
    metrics = SchedulerMetrics()
    metrics.total_requests = len(trace)

    is_loaded = False
    last_activity = 0
    t = 0

    for req in trace:
        t = req["timestamp"]

        # Check if model was evicted due to TTL
        if is_loaded and (t - last_activity) > ttl_s:
            # Model was evicted at last_activity + ttl_s
            evict_time = last_activity + ttl_s
            idle_duration = ttl_s
            metrics.idle_energy_wh += gpu_config["P_idle"] * idle_duration / 3600
            # After eviction, GPU at bare idle
            bare_duration = t - evict_time
            metrics.idle_energy_wh += gpu_config["P_base"] * bare_duration / 3600
            is_loaded = False

        if not is_loaded:
            # Cold start
            metrics.total_cold_starts += 1
            load_time = gpu_config["t_load_pytorch"]
            metrics.cold_start_energy_wh += gpu_config["P_load"] * load_time / 3600
            metrics.total_cold_start_latency_s += load_time
            is_loaded = True
        else:
            # Was warm — count idle time since last request
            idle_duration = t - last_activity
            metrics.idle_energy_wh += gpu_config["P_idle"] * idle_duration / 3600

        # Process request
        last_activity = t + req["duration_s"]

    # After last request until end of simulation
    if is_loaded:
        remaining = duration_s - last_activity
        if remaining > ttl_s:
            metrics.idle_energy_wh += gpu_config["P_idle"] * ttl_s / 3600
            metrics.idle_energy_wh += gpu_config["P_base"] * (remaining - ttl_s) / 3600
        else:
            metrics.idle_energy_wh += gpu_config["P_idle"] * remaining / 3600
    else:
        metrics.idle_energy_wh += gpu_config["P_base"] * (duration_s - last_activity) / 3600

    metrics.total_energy_wh = metrics.idle_energy_wh + metrics.cold_start_energy_wh
    return metrics


def simulate_breakeven(trace, gpu_config, duration_s, use_fast_loader=False):
    """Breakeven policy: evict when expected idle > T*.

    For memoryless (Poisson) arrivals, the optimal policy is deterministic:
    evict after T* seconds of idle time, where:
        T* = (P_load * t_load) / P_park
    """
    t_load = gpu_config["t_load_fast"] if use_fast_loader else gpu_config["t_load_pytorch"]
    T_star = (gpu_config["P_load"] * t_load) / gpu_config["P_park"]

    # The breakeven policy is actually a fixed-TTL with T* as the TTL!
    # The insight is that T* is derived from our measurements, not guessed.
    metrics = simulate_fixed_ttl(trace, gpu_config, duration_s, T_star)
    return metrics, T_star


# ============================================================================
# SIMULATION RUNNER
# ============================================================================
def run_simulation(
    trace_name: str,
    trace: list[dict],
    gpu_name: str = "H100",
    duration_hours: float = 24,
):
    """Run all policies on a trace and compare."""
    gpu = GPU_CONFIGS[gpu_name]
    duration_s = duration_hours * 3600

    print(f"\n{'=' * 70}")
    print(f"SIMULATION: {trace_name} | GPU: {gpu_name} | Duration: {duration_hours}h")
    print(f"  Requests: {len(trace)}")
    if trace:
        avg_gap = duration_s / len(trace)
        print(f"  Avg inter-arrival: {avg_gap:.1f}s ({avg_gap/60:.1f} min)")
    print(f"{'=' * 70}")

    results = {}

    # Always-on baseline
    ao = simulate_always_on(trace, gpu, duration_s)
    results["always_on"] = ao
    print(f"\n  Always-On:")
    print(f"    Energy: {ao.total_energy_wh:.1f} Wh")
    print(f"    Cold starts: {ao.total_cold_starts}")

    # Fixed TTLs
    for ttl_min in [1, 2, 5, 10, 15, 30, 60]:
        ft = simulate_fixed_ttl(trace, gpu, duration_s, ttl_min * 60)
        results[f"ttl_{ttl_min}min"] = ft
        savings_pct = (1 - ft.total_energy_wh / ao.total_energy_wh) * 100 if ao.total_energy_wh > 0 else 0
        print(f"  TTL {ttl_min:>2}min: {ft.total_energy_wh:>8.1f} Wh | "
              f"Cold starts: {ft.total_cold_starts:>4} | "
              f"Savings: {savings_pct:>5.1f}% | "
              f"Avg latency added: {ft.total_cold_start_latency_s/max(len(trace),1):.2f}s/req")

    # Breakeven (standard loader)
    be, T_star = simulate_breakeven(trace, gpu, duration_s, use_fast_loader=False)
    results["breakeven_standard"] = be
    savings_pct = (1 - be.total_energy_wh / ao.total_energy_wh) * 100 if ao.total_energy_wh > 0 else 0
    print(f"\n  Breakeven (T*={T_star:.0f}s = {T_star/60:.1f}min, standard loader):")
    print(f"    Energy: {be.total_energy_wh:.1f} Wh | Savings: {savings_pct:.1f}%")
    print(f"    Cold starts: {be.total_cold_starts} | "
          f"Avg latency: {be.total_cold_start_latency_s/max(len(trace),1):.2f}s/req")

    # Breakeven (fast loader)
    be_fast, T_star_fast = simulate_breakeven(trace, gpu, duration_s, use_fast_loader=True)
    results["breakeven_fast"] = be_fast
    savings_fast = (1 - be_fast.total_energy_wh / ao.total_energy_wh) * 100 if ao.total_energy_wh > 0 else 0
    print(f"\n  Breakeven (T*={T_star_fast:.0f}s = {T_star_fast/60:.1f}min, fast loader):")
    print(f"    Energy: {be_fast.total_energy_wh:.1f} Wh | Savings: {savings_fast:.1f}%")
    print(f"    Cold starts: {be_fast.total_cold_starts} | "
          f"Avg latency: {be_fast.total_cold_start_latency_s/max(len(trace),1):.2f}s/req")

    return results, T_star, T_star_fast


# ============================================================================
# FIGURES
# ============================================================================
def plot_energy_comparison(all_results: dict):
    """Bar chart comparing energy across policies and traffic patterns."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax_idx, (trace_name, sim_results) in enumerate(all_results.items()):
        ax = axes[ax_idx]

        policies = []
        energies = []
        cold_starts = []
        colors = []

        # Always-on
        ao = sim_results["always_on"]
        policies.append("Always-On")
        energies.append(ao.total_energy_wh)
        cold_starts.append(ao.total_cold_starts)
        colors.append("#94a3b8")

        # Selected TTLs
        for ttl_min, color in [(5, "#fbbf24"), (15, "#f59e0b")]:
            key = f"ttl_{ttl_min}min"
            if key in sim_results:
                ft = sim_results[key]
                policies.append(f"TTL {ttl_min}m")
                energies.append(ft.total_energy_wh)
                cold_starts.append(ft.total_cold_starts)
                colors.append(color)

        # Breakeven
        be = sim_results["breakeven_standard"]
        policies.append("Breakeven\n(standard)")
        energies.append(be.total_energy_wh)
        cold_starts.append(be.total_cold_starts)
        colors.append("#3b82f6")

        be_fast = sim_results["breakeven_fast"]
        policies.append("Breakeven\n(fast load)")
        energies.append(be_fast.total_energy_wh)
        cold_starts.append(be_fast.total_cold_starts)
        colors.append("#06b6d4")

        x = np.arange(len(policies))
        bars = ax.bar(x, energies, color=colors, edgecolor="black", linewidth=0.5,
                      width=0.7)

        # Annotate savings vs always-on
        for i, (e, cs) in enumerate(zip(energies, cold_starts)):
            if i == 0:
                continue
            savings = (1 - e / energies[0]) * 100
            ax.text(i, e + max(energies) * 0.02,
                    f"-{savings:.0f}%\n({cs} cs)",
                    ha="center", va="bottom", fontsize=7, color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels(policies, fontsize=8)
        ax.set_ylabel("Energy (Wh)" if ax_idx == 0 else "")
        ax.set_title(trace_name, fontweight="bold")

    fig.suptitle("Energy Savings: Scheduling Policies vs Always-On (H100, 24h)",
                 fontweight="bold", fontsize=12, y=1.03)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_scheduler_comparison.pdf")
    fig.savefig(FIGURES_DIR / "fig_scheduler_comparison.png")
    plt.close(fig)
    print(f"\nSaved scheduler comparison to {FIGURES_DIR / 'fig_scheduler_comparison.png'}")


def plot_rate_sweep(gpu_name="H100"):
    """Show energy savings as a function of request rate."""
    gpu = GPU_CONFIGS[gpu_name]
    duration_hours = 24
    duration_s = duration_hours * 3600

    rates = np.logspace(-1, 2, 30)  # 0.1 to 100 req/hr
    savings_standard = []
    savings_fast = []
    savings_ttl5 = []

    for rate in rates:
        trace = generate_poisson_trace(rate, duration_hours, n_models=1, seed=42)
        ao = simulate_always_on(trace, gpu, duration_s)

        be, _ = simulate_breakeven(trace, gpu, duration_s, use_fast_loader=False)
        be_f, _ = simulate_breakeven(trace, gpu, duration_s, use_fast_loader=True)
        ttl5 = simulate_fixed_ttl(trace, gpu, duration_s, 5 * 60)

        s_std = (1 - be.total_energy_wh / ao.total_energy_wh) * 100 if ao.total_energy_wh > 0 else 0
        s_fast = (1 - be_f.total_energy_wh / ao.total_energy_wh) * 100 if ao.total_energy_wh > 0 else 0
        s_ttl = (1 - ttl5.total_energy_wh / ao.total_energy_wh) * 100 if ao.total_energy_wh > 0 else 0

        savings_standard.append(s_std)
        savings_fast.append(s_fast)
        savings_ttl5.append(s_ttl)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(rates, savings_standard, "o-", color="#3b82f6", label="Breakeven (standard loader)", markersize=4)
    ax.semilogx(rates, savings_fast, "s-", color="#06b6d4", label="Breakeven (fast loader)", markersize=4)
    ax.semilogx(rates, savings_ttl5, "^--", color="#f59e0b", label="Fixed TTL (5 min)", markersize=4, alpha=0.7)

    # Mark the critical rate λ*
    lambda_star = gpu["P_park"] / (gpu["P_load"] * gpu["t_load_pytorch"]) * 3600
    ax.axvline(lambda_star, color="#ef4444", linestyle=":", linewidth=1.5,
               label=f"$\\lambda^*$ = {lambda_star:.1f} req/hr")

    ax.set_xlabel("Request Rate (req/hr)")
    ax.set_ylabel("Energy Savings vs Always-On (%)")
    ax.set_title(f"Breakeven Scheduler: Energy Savings by Traffic Rate ({gpu_name})",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(-5, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_scheduler_rate_sweep.pdf")
    fig.savefig(FIGURES_DIR / "fig_scheduler_rate_sweep.png")
    plt.close(fig)
    print(f"Saved rate sweep to {FIGURES_DIR / 'fig_scheduler_rate_sweep.png'}")


def plot_cross_architecture():
    """Compare breakeven savings across GPU architectures."""
    rates = np.logspace(-1, 2, 25)
    duration_hours = 24
    duration_s = duration_hours * 3600

    fig, ax = plt.subplots(figsize=(7, 4.5))
    gpu_colors = {"H100": "#3b82f6", "A100": "#10b981", "L40S": "#f59e0b"}

    for gpu_name, color in gpu_colors.items():
        gpu = GPU_CONFIGS[gpu_name]
        savings = []
        for rate in rates:
            trace = generate_poisson_trace(rate, duration_hours, n_models=1, seed=42)
            ao = simulate_always_on(trace, gpu, duration_s)
            be, _ = simulate_breakeven(trace, gpu, duration_s, use_fast_loader=False)
            s = (1 - be.total_energy_wh / ao.total_energy_wh) * 100 if ao.total_energy_wh > 0 else 0
            savings.append(s)

        lambda_star = gpu["P_park"] / (gpu["P_load"] * gpu["t_load_pytorch"]) * 3600
        ax.semilogx(rates, savings, "o-", color=color, markersize=4,
                     label=f"{gpu_name} ($\\lambda^*$={lambda_star:.1f}/hr)")

    ax.set_xlabel("Request Rate (req/hr)")
    ax.set_ylabel("Energy Savings vs Always-On (%)")
    ax.set_title("Breakeven Scheduler Savings Across GPU Architectures",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(-5, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_scheduler_cross_arch.pdf")
    fig.savefig(FIGURES_DIR / "fig_scheduler_cross_arch.png")
    plt.close(fig)
    print(f"Saved cross-architecture comparison to {FIGURES_DIR / 'fig_scheduler_cross_arch.png'}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Parking Tax Scheduler Simulator")
    parser.add_argument("--gpu", type=str, default="H100", choices=list(GPU_CONFIGS.keys()))
    parser.add_argument("--duration", type=float, default=24, help="Simulation duration in hours")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    global FIGURES_DIR
    FIGURES_DIR = Path(args.output_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PARKING TAX SCHEDULER — PROOF OF CONCEPT")
    print("=" * 70)

    all_results = {}

    # Scenario 1: Steady low traffic (5 req/hr — below λ* for most GPUs)
    trace_low = generate_poisson_trace(5, args.duration, n_models=1, seed=42)
    results_low, T_star, T_star_fast = run_simulation(
        "Low Traffic (5 req/hr)", trace_low, args.gpu, args.duration
    )
    all_results["Low Traffic\n(5 req/hr)"] = results_low

    # Scenario 2: Bursty traffic
    trace_bursty = generate_bursty_trace(
        base_rate=2, burst_rate=60, burst_fraction=0.3,
        duration_hours=args.duration, n_models=1, seed=42
    )
    results_bursty, _, _ = run_simulation(
        "Bursty Traffic", trace_bursty, args.gpu, args.duration
    )
    all_results["Bursty Traffic\n(2/60 req/hr)"] = results_bursty

    # Scenario 3: Diurnal pattern
    trace_diurnal = generate_diurnal_trace(
        peak_rate=30, duration_hours=args.duration, n_models=1, seed=42
    )
    results_diurnal, _, _ = run_simulation(
        "Diurnal Pattern (peak 30 req/hr)", trace_diurnal, args.gpu, args.duration
    )
    all_results["Diurnal Pattern\n(peak 30 req/hr)"] = results_diurnal

    # Generate figures
    print(f"\n{'=' * 70}")
    print("GENERATING FIGURES")
    print(f"{'=' * 70}")

    plot_energy_comparison(all_results)
    plot_rate_sweep(args.gpu)
    plot_cross_architecture()

    # Save numerical results
    def metrics_to_dict(m):
        return {
            "total_energy_wh": m.total_energy_wh,
            "cold_starts": m.total_cold_starts,
            "requests": m.total_requests,
            "cold_start_energy_wh": m.cold_start_energy_wh,
            "idle_energy_wh": m.idle_energy_wh,
            "avg_cold_start_latency_s": m.total_cold_start_latency_s / max(m.total_requests, 1),
        }

    json_results = {}
    for trace_name, sim_results in all_results.items():
        trace_key = trace_name.replace("\n", " ").strip()
        json_results[trace_key] = {
            policy: metrics_to_dict(m) if isinstance(m, SchedulerMetrics) else {
                "metrics": metrics_to_dict(m[0]) if isinstance(m, tuple) else metrics_to_dict(m)
            }
            for policy, m in sim_results.items()
        }

    json_results["breakeven_times"] = {
        "T_star_standard_s": T_star,
        "T_star_fast_s": T_star_fast,
        "T_star_standard_min": T_star / 60,
        "T_star_fast_min": T_star_fast / 60,
    }

    output_file = RESULTS_DIR / "scheduler_results.json"
    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    # Print paper-ready summary
    print(f"\n{'=' * 70}")
    print("PAPER-READY SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Breakeven time T* (standard): {T_star:.0f}s ({T_star/60:.1f} min)")
    print(f"  Breakeven time T* (fast):     {T_star_fast:.0f}s ({T_star_fast/60:.1f} min)")

    for trace_name, sim_results in all_results.items():
        ao_energy = sim_results["always_on"].total_energy_wh
        be_energy = sim_results["breakeven_standard"].total_energy_wh
        savings = (1 - be_energy / ao_energy) * 100 if ao_energy > 0 else 0
        cs = sim_results["breakeven_standard"].total_cold_starts
        print(f"\n  {trace_name.replace(chr(10), ' ')}:")
        print(f"    Always-on: {ao_energy:.0f} Wh | Breakeven: {be_energy:.0f} Wh")
        print(f"    Savings: {savings:.1f}% | Cold starts: {cs}")


if __name__ == "__main__":
    main()
