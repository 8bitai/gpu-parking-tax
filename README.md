# The Model Parking Tax

Companion repository for *"The Model Parking Tax: Quantifying the Hidden Energy Cost of Always-On GPU Model Deployment"*.

## Overview

This paper presents the first cross-architecture measurement of idle GPU power as a function of VRAM allocation. We combine 18 days of production telemetry (335,267 samples, 14 H100 GPUs) with controlled dose-response experiments on three GPU architectures: NVIDIA H100 (HBM3), A100 (HBM2e), and L40S (GDDR6).

**Key finding:** Idle GPU power is *piecewise constant* across all three architectures. The CUDA context forces a discrete DVFS transition (+26--66 W), while the marginal VRAM effect is bounded below measurement relevance (|beta| < 0.02 W/GB). The CUDA context accounts for >98% of the parking tax regardless of memory technology.

## Repository Structure

```
paper/                                  LaTeX source and figures
  parking_tax.tex                       Main paper
  cross_architecture_dose_response.png  Figure 1: three GPUs, dose-response curves
  parking_tax_decomposition.png         Figure 2: base/context/VRAM bar chart
  vram_regression_detail.png            Figure 3: zoomed regression per GPU

scraper/                                Phase 1: production telemetry collection
  scrape.py                             DCGM metric collection daemon (Prometheus)
  preprocess.py                         Raw CSV to Parquet pipeline
  workload_classifier.py                K8s metadata to workload labels
  config.yaml                           Metric definitions and classification rules
  validate.py                           Data validation checks
  quick_test.py                         Scraper connectivity test

experiments/                            Phase 2: controlled experiments
  dose_response.py                      VRAM dose-response (auto-detects GPU arch)
  model_validation.py                   Real model validation (Qwen2.5-7B on all GPUs)
  scheduler_simulation.py               Breakeven scheduler simulation (Section 6)

analysis/                               Analysis and figure generation
  phase1_telemetry.py                   Phase 1 production telemetry analysis
  phase2_controlled.py                  Phase 2 controlled experiment analysis
  generate_paper_figures.py             Generate all three paper figures
  supplementary_figures.py              Phase 1 supplementary figures
  sensitivity_analysis.py               Industry-scale sensitivity analysis
  results/                              Precomputed results (JSON, CSV)

data/
  telemetry/                            Phase 1 raw CSVs (not included, see below)
  experiments/                          Phase 2 experiment data
    h100_dose_response.jsonl            H100 canonical run (paper)
    a100_dose_response.jsonl            A100 canonical run (paper)
    l40s_dose_response.jsonl            L40S canonical run (paper)
    h100_quick_test.jsonl               H100 quick validation
    a100_exploratory.jsonl              A100 earlier run
    l40s_exploratory.jsonl              L40S earlier run
    h100_model_validation.jsonl         Qwen2.5-7B idle power validation
    h100_cold_start_traces.jsonl        Cold-start power traces (1 Hz)
    h100_validation_manifest.json       Validation experiment metadata
```

## Reproducing Results

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- GPU access required only for running experiments (not for analysis)

### Setup

```bash
uv sync
cp .env.example .env  # only needed for scraper
```

### Running Analysis

The analysis scripts operate on data in `data/experiments/` (Phase 2) and `analysis/results/` (precomputed Phase 1 results):

```bash
# Regenerate all paper figures
uv run python analysis/generate_paper_figures.py

# Phase 2 analysis (controlled experiments)
uv run python analysis/phase2_controlled.py

# Sensitivity analysis (industry-scale estimates)
uv run python analysis/sensitivity_analysis.py
```

### Running Experiments (requires GPU access)

The dose-response script auto-detects GPU architecture and selects appropriate VRAM levels:

```bash
# Quick test (~45 min, 5 VRAM levels)
uv run python experiments/dose_response.py --gpu 0 --quick

# Full experiment (~3.5 hours, 8+ VRAM levels, 20 min per phase)
uv run python experiments/dose_response.py --gpu 0

# Custom VRAM levels
uv run python experiments/dose_response.py --gpu 0 --levels 0,8,32,64
```

Real model validation:

```bash
uv run python experiments/model_validation.py --gpu 0
```

Scheduler simulation (no GPU needed):

```bash
uv run python experiments/scheduler_simulation.py
```

### Collecting Telemetry (requires Prometheus/DCGM infrastructure)

```bash
# Edit .env with your Prometheus/DCGM endpoints
uv run python scraper/scrape.py --daemon
uv run python scraper/preprocess.py
```

## Phase 1 Telemetry Data

The raw Phase 1 telemetry (18 days, 335,267 idle samples from 14 H100 GPUs) totals ~6.8 GB of daily CSV files and is not included in this repository. The precomputed Phase 1 results in `analysis/results/phase1_results.json` contain all statistics reported in the paper. Contact the authors if you need access to the raw telemetry.

## License

MIT
