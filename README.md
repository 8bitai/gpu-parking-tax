# The Model Parking Tax

[![arXiv](https://img.shields.io/badge/arXiv-2605.23918-b31b1b.svg)](https://arxiv.org/abs/2605.23918)

Companion repository for *"The Model Parking Tax: Quantifying the Hidden Energy Cost of Always-On GPU Model Deployment"* ([arXiv:2605.23918](https://arxiv.org/abs/2605.23918)).

## Overview

This paper presents the first cross-architecture measurement of idle GPU power as a function of VRAM allocation. We combine 18 days of production telemetry (335,267 samples, 14 H100 GPUs) with controlled dose-response experiments on three GPU architectures: NVIDIA H100 (HBM3), A100 (HBM2e), and L40S (GDDR6).

**Key finding:** Idle GPU power is *piecewise constant* across all three architectures. The CUDA context forces a discrete DVFS transition (+26--66 W), while the marginal VRAM effect is bounded below measurement relevance (|beta| < 0.02 W/GB). NVIDIA's `CUDA_DISABLE_PERF_BOOST` flag eliminates the parking tax within measurement noise on both Hopper and Ampere. On a 4xH100 node, the tax multiplies linearly at ~52 W/GPU across TP, DP, and mixed configurations.

## Repository Structure

```
paper/
  parking_tax.tex                       Paper source (HotCarbon 2026)
  *.png                                 Generated figures

experiments/                            Experiment runner scripts
  dose_response.py                      VRAM dose-response (auto-detects GPU arch)
  model_validation.py                   Real model validation (Qwen2.5-7B)
  perf_boost.py                         CUDA_DISABLE_PERF_BOOST evaluation
  multi_gpu.py                          Multi-GPU TP/DP scaling (4xH100)
  latency_retest.py                     Focused latency retest protocol
  scheduler_simulation.py               Breakeven scheduler simulation
  utils.py                              Shared helpers (nvidia-smi, vLLM lifecycle)

analysis/                               Analysis and figure generation
  phase1_telemetry.py                   Phase 1 production telemetry analysis
  phase2_controlled.py                  Phase 2 controlled experiment analysis
  generate_paper_figures.py             Phase 2 dose-response figures
  generate_perf_boost_figures.py        Perf-boost power/latency figures
  generate_multi_gpu_figures.py         Multi-GPU scaling figures
  sensitivity_analysis.py              Industry-scale sensitivity analysis
  supplementary_figures.py              Phase 1 supplementary figures
  results/                              Precomputed results (JSON, CSV)

data/
  telemetry/                            Phase 1 raw CSVs (18 days, 14 H100s)
  raw/                                  Experiment data (JSONL)
    h100_dose_response.jsonl            H100 dose-response (paper)
    a100_dose_response.jsonl            A100 dose-response (paper)
    l40s_dose_response.jsonl            L40S dose-response (paper)
    h100_model_validation.jsonl         Qwen2.5-7B idle power validation
    h100_cold_start_traces.jsonl        Cold-start power traces (1 Hz)
    perf_boost_h100_sxm.jsonl           Flag evaluation, H100 SXM
    perf_boost_a100_sxm4.jsonl          Flag evaluation, A100 SXM4
    latency_retest_h100_sxm.json        Latency retest, H100
    latency_retest_a100_sxm4.json       Latency retest, A100
    multi_gpu.jsonl                     4xH100 TP/DP scaling experiment

scraper/                                Phase 1: production telemetry collection
  scrape.py                             DCGM metric collection daemon (Prometheus)
  preprocess.py                         Raw CSV pipeline
  workload_classifier.py                K8s metadata to workload labels
  config.yaml                           Metric definitions and classification rules
  validate.py                           Data validation checks
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

```bash
# Regenerate all paper figures
uv run python analysis/generate_paper_figures.py
uv run python analysis/generate_perf_boost_figures.py
uv run python analysis/generate_multi_gpu_figures.py

# Phase 2 analysis (controlled experiments)
uv run python analysis/phase2_controlled.py

# Sensitivity analysis (industry-scale estimates)
uv run python analysis/sensitivity_analysis.py
```

### Running Experiments (requires GPU access)

```bash
# Dose-response (auto-detects GPU, ~3.5 hours full / ~45 min quick)
uv run python experiments/dose_response.py --gpu 0
uv run python experiments/dose_response.py --gpu 0 --quick

# CUDA_DISABLE_PERF_BOOST (requires driver >= 580.105.08)
uv run python experiments/perf_boost.py --gpu 0

# Multi-GPU (requires 4xH100 SXM with NVLink)
uv run python experiments/multi_gpu.py          # full (~6.5 hours)
uv run python experiments/multi_gpu.py --quick  # quick (~3.5 hours)
uv run python experiments/multi_gpu.py --smoke-test

# Real model validation
uv run python experiments/model_validation.py --gpu 0

# Scheduler simulation (no GPU needed)
uv run python experiments/scheduler_simulation.py
```

### Collecting Telemetry (requires Prometheus/DCGM infrastructure)

```bash
# Edit .env with your Prometheus/DCGM endpoints
uv run python scraper/scrape.py --daemon
uv run python scraper/preprocess.py
```

## Phase 1 Telemetry Data

The raw Phase 1 telemetry (18 days, 335,267 idle samples from 14 H100 GPUs) totals ~6.8 GB of daily CSV files. The precomputed Phase 1 results in `analysis/results/phase1_results.json` contain all statistics reported in the paper. Contact the authors for raw telemetry access.

## Citation

If you use this work, please cite:

```bibtex
@article{vadari2026parkingtax,
  title={The Model Parking Tax: Quantifying the Hidden Energy Cost of Always-On GPU Model Deployment},
  author={Vadari, Sai Sathvik},
  journal={arXiv preprint arXiv:2605.23918},
  year={2026}
}
```

## License

MIT
