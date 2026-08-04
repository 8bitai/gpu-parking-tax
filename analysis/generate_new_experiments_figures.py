#!/usr/bin/env python3
"""
Generate figures for the L40S/B200 flag and H100 throughput experiments.

Data files:
  - H100 SXM flag:  data/raw/perf_boost_h100_sxm.jsonl   (condition schema)
  - A100 SXM4 flag: data/raw/perf_boost_a100_sxm4.jsonl  (condition schema)
  - L40S baseline:  data/raw/l40s_dose_response.jsonl     (phase schema)
  - L40S flag:      data/raw/l40s_dose_response_flag.jsonl
  - B200 baseline:  data/raw/b200_dose_response.jsonl
  - B200 flag:      data/raw/b200_dose_response_flag.jsonl
  - H100 tput:      data/raw/throughput_h100_sxm.json

Produces:
  1. flag_recovery_across_architectures.png — bare / context-off / context-on, x4 GPUs
  2. b200_dose_response.png                  — Blackwell tax + VRAM-flatness
  3. throughput_flag_comparison.png          — H100 tok/s and load SM clock, flag off vs on

Note: matplotlib text is NOT in usetex mode, so annotation strings use plain
spaces (no LaTeX "\\," thin-spaces) and mathtext only for $-$/$+$/$\\Delta$.
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style: match generate_paper_figures.py / generate_perf_boost_figures.py ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

RAW = Path("data/raw")
FIGURES_DIR = Path("figures")
PAPER_DIR = Path("paper")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GPU_COLORS = {"H100": "#3b82f6", "A100": "#10b981",
              "L40S": "#f59e0b", "B200": "#8b5cf6"}
BARE_COLOR = "#94a3b8"


def load_jsonl(path):
    rows = []
    with open(RAW / path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean_by(rows, field, value, key="power_w", agg=np.mean):
    vals = [r[key] for r in rows if r.get(field) == value and r.get(key) is not None]
    return float(agg(vals)) if vals else None


def get(spec, key="power_w", agg=np.mean):
    """spec is either ('condition', file, cond) or ('phase', file, phase).

    agg defaults to mean; pass np.median for idle-power bars so transient DVFS
    spikes (a stray sample at full clock) do not distort the settled value.
    """
    schema, fname, sel = spec
    rows = load_jsonl(fname)
    field = "condition" if schema == "condition" else "phase"
    return mean_by(rows, field, sel, key, agg=agg)


# ── Figure 1: flag recovery across four architectures ───────────────────────
ARCHES = [
    ("H100", ("condition", "perf_boost_h100_sxm.jsonl", "bare"),
             ("condition", "perf_boost_h100_sxm.jsonl", "baseline"),
             ("condition", "perf_boost_h100_sxm.jsonl", "flag_on")),
    ("A100", ("condition", "perf_boost_a100_sxm4.jsonl", "bare"),
             ("condition", "perf_boost_a100_sxm4.jsonl", "baseline"),
             ("condition", "perf_boost_a100_sxm4.jsonl", "flag_on")),
    ("L40S", ("phase", "l40s_dose_response.jsonl", "bare_idle"),
             ("phase", "l40s_dose_response.jsonl", "phase1_0gb"),
             ("phase", "l40s_dose_response_flag.jsonl", "vram_0gb")),
    ("B200", ("phase", "b200_dose_response.jsonl", "bare_idle"),
             ("phase", "b200_dose_response.jsonl", "vram_0gb"),
             ("phase", "b200_dose_response_flag.jsonl", "vram_0gb")),
]


def flag_recovery_across_architectures():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    group_w = 0.72
    bw = group_w / 3.0
    xs = np.arange(len(ARCHES))

    for gi, (name, s_bare, s_off, s_on) in enumerate(ARCHES):
        bare = get(s_bare, agg=np.median)
        off = get(s_off, agg=np.median)
        on = get(s_on, agg=np.median)
        c = GPU_COLORS[name]
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        light = f"#{min(r+80,255):02x}{min(g+80,255):02x}{min(b+80,255):02x}"
        x0 = gi - bw
        vals = [bare, off, on]
        cols = [BARE_COLOR, c, light]
        hatches = ["", "", "//"]
        for k, (v, col, h) in enumerate(zip(vals, cols, hatches)):
            ax.bar(x0 + k * bw, v, bw * 0.92, color=col, edgecolor="black",
                   linewidth=0.5, alpha=0.9, hatch=h, zorder=5,
                   label={0: "Bare idle", 1: "CUDA ctx (default)",
                          2: "CUDA ctx (flag on)"}[k] if gi == 0 else None)
            ax.text(x0 + k * bw, v + 4, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        # recovery annotation
        tax = off - bare
        resid = on - bare
        recovered = 100 * (tax - resid) / tax if tax else 0
        ax.text(gi, off + 26, f"$-${off-on:.0f} W\n{recovered:.0f}% recovered",
                ha="center", va="bottom", fontsize=11, color="red",
                fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([a[0] for a in ARCHES], fontweight="bold")
    ax.set_ylabel("Idle Power (W)")
    ax.set_ylim(0, 330)
    ax.set_title("The flag returns context-active idle power toward the bare-idle floor "
                 "on every architecture", fontweight="bold", fontsize=14)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "flag_recovery_across_architectures.png")
    fig.savefig(PAPER_DIR / "flag_recovery_across_architectures.png")
    plt.close(fig)
    print("  Saved flag_recovery_across_architectures")


# ── Figure 2: B200 dose-response (tax + VRAM-flatness) ──────────────────────
def b200_dose_response():
    rows = load_jsonl("b200_dose_response.jsonl")
    bare = mean_by(rows, "phase", "bare_idle")
    vram_phases = sorted({r["phase"] for r in rows if r["phase"].startswith("vram_")},
                         key=lambda p: int(p.split("_")[1].replace("gb", "")))
    gbs = [int(p.split("_")[1].replace("gb", "")) for p in vram_phases]
    powers = [mean_by(rows, "phase", p) for p in vram_phases]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    c = GPU_COLORS["B200"]
    ax.axhline(bare, ls="--", color=BARE_COLOR, lw=1.5, zorder=2)
    ax.scatter([-4], [bare], s=140, color=BARE_COLOR, edgecolor="black",
               zorder=6, label=f"Bare idle ({bare:.0f} W, 120 MHz)")
    ax.plot(gbs, powers, "-o", color=c, markersize=9, lw=2.2, zorder=5,
            markeredgecolor="black", label="CUDA ctx active (1965 MHz)")
    tax = np.mean(powers) - bare
    spread = max(powers) - min(powers)
    ax.annotate("", xy=(-4, np.mean(powers)), xytext=(-4, bare),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.8), zorder=7)
    ax.text(-2.5, (np.mean(powers) + bare) / 2, f"$+${tax:.0f} W\ncontext\ntax",
            ha="left", va="center", color="red", fontsize=11, fontweight="bold")
    ax.text(np.mean(gbs), np.mean(powers) + 6,
            f"flat across VRAM ($\\Delta$ {spread:.1f} W over 0–32 GB)",
            ha="center", fontsize=11)
    ax.set_xlabel("Allocated VRAM (GB)")
    ax.set_ylabel("Idle Power (W)")
    ax.set_ylim(bare - 20, max(powers) + 30)
    ax.set_xlim(-6, 34)
    ax.set_title("Blackwell B200 (HBM3e): context step, VRAM-independent",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "b200_dose_response.png")
    fig.savefig(PAPER_DIR / "b200_dose_response.png")
    plt.close(fig)
    print("  Saved b200_dose_response")


# ── Figure 3: H100 throughput, flag off vs on ───────────────────────────────
def throughput_flag_comparison():
    with open(RAW / "throughput_h100_sxm.json") as f:
        d = json.load(f)
    b = d["results"]["baseline"]
    fl = d["results"]["flag_on"]
    c = GPU_COLORS["H100"]
    r, g, bl = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    light = f"#{min(r+80,255):02x}{min(g+80,255):02x}{min(bl+80,255):02x}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4))

    # throughput
    tp = [b["throughput_tok_s"], fl["throughput_tok_s"]]
    ax1.bar([0, 1], tp, 0.55, color=[c, light], edgecolor="black", linewidth=0.5,
            hatch=["", "//"], alpha=0.9, zorder=5)
    for i, v in enumerate(tp):
        ax1.text(i, v + 60, f"{v:.0f}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
    dpct = 100 * (tp[1] - tp[0]) / tp[0]
    ax1.text(0.5, max(tp) * 0.5, f"{dpct:+.1f}%\n(single run)", ha="center",
             va="center", fontsize=12, fontweight="bold", color="red",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="red", alpha=0.95))
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Flag off\n(default)", "Flag on"])
    ax1.set_ylabel("Decode throughput (tok/s)")
    ax1.set_ylim(0, max(tp) * 1.18)
    ax1.set_title("Throughput under load", fontweight="bold", fontsize=13)

    # load SM clock
    clk = [b["load_sm_clock_mean_mhz"], fl["load_sm_clock_mean_mhz"]]
    ax2.bar([0, 1], clk, 0.55, color=[c, light], edgecolor="black", linewidth=0.5,
            hatch=["", "//"], alpha=0.9, zorder=5)
    for i, v in enumerate(clk):
        ax2.text(i, v + 30, f"{v:.0f}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Flag off\n(default)", "Flag on"])
    ax2.set_ylabel("Load SM clock (MHz)")
    ax2.set_ylim(0, max(clk) * 1.22)
    ax2.set_title("Clock reached under load", fontweight="bold", fontsize=13)
    ax2.text(0.5, max(clk) * 0.5, "identical\n(full boost)", ha="center",
             va="center", fontsize=12, fontweight="bold", color="red",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="red", alpha=0.95))

    fig.suptitle("H100 SXM: the flag suppresses only idle boost, not active clocks "
                 "(Qwen2.5-7B, concurrency 64)", fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "throughput_flag_comparison.png")
    fig.savefig(PAPER_DIR / "throughput_flag_comparison.png")
    plt.close(fig)
    print("  Saved throughput_flag_comparison")


def main():
    print("Generating new-experiment figures...")
    flag_recovery_across_architectures()
    b200_dose_response()
    throughput_flag_comparison()
    print(f"\nAll figures saved to {FIGURES_DIR}/ and {PAPER_DIR}/")


if __name__ == "__main__":
    main()
