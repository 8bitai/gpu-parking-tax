# GPU Behavioral Profiling

Workload-stratified GPU telemetry analysis — scraping, preprocessing, and interactive EDA dashboard for DCGM metrics across multi-node GPU clusters.

## What This Does

Collects 41 DCGM telemetry metrics from GPU clusters, classifies active workloads by Kubernetes pod metadata, and serves an 8-page interactive dashboard for exploratory data analysis. Designed for understanding GPU behavior patterns before building ML models (anomaly detection, workload fingerprinting).

### Key Capabilities

- **Workload classification** — Automatic labeling of GPU activity by workload type (LLM inference, embedding, voice AI, computer vision) based on K8s container/pod patterns
- **Derived features** — 7 computed metrics: tensor dominance, FP16/FP32 ratio, compute-memory ratio, clock throttling, thermal headroom, power efficiency, memory utilization
- **Pre-aggregated analytics** — Hourly rollups, per-workload statistics, correlation matrices, quantile tables
- **Interactive EDA dashboard** — 8 pages of Plotly charts with dark mode, workload-colored visualizations, and metric selectors

## Architecture

```
Prometheus/DCGM Exporter
        |
   scraper/ ──→ data/raw/ (daily CSVs)
        |
  preprocess.py ──→ data/processed/ (5 Parquet files)
        |
     api/ (FastAPI) ──→ dashboard/ (Next.js + Plotly)
```

**Why this stack?**
- **FastAPI + PyArrow**: Multi-GB Parquet files need predicate pushdown. Python handles this natively.
- **Plotly**: Only charting lib with native violin plots, heatmaps, radar charts, and 3D scatter — all needed for EDA.
- **Pre-aggregation**: Most endpoints serve from pre-computed Parquets (<100ms). Distribution/scatter endpoints downsample to max 5K points.

## Project Structure

```
├── scraper/
│   ├── scrape.py              # DCGM metric collection daemon
│   ├── preprocess.py          # CSV → Parquet pipeline + aggregations
│   ├── workload_classifier.py # K8s metadata → workload labels
│   └── config.yaml            # Metrics, labels, classification rules
├── api/
│   ├── main.py                # FastAPI app with CORS + lifespan
│   ├── data_loader.py         # Parquet cache + numpy-safe serialization
│   └── routers/               # 8 route modules (26 endpoints)
├── dashboard/
│   └── src/
│       ├── app/               # 8 Next.js pages
│       ├── components/        # PlotlyChart, Sidebar, Card, Providers
│       └── lib/               # API client, zustand store, color config
├── pyproject.toml
└── .env.example
```

## Dashboard Pages

| Page | What It Shows |
|---|---|
| **Overview** | KPIs, workload breakdown, utilization timeline (stacked area), GPU×Day heatmap |
| **Workload Profiles** | Radar fingerprints, violin distributions, pipe breakdown, comparison table |
| **GPU Health** | Per-GPU cards (util/temp/power), selectable metric heatmap, error summary |
| **Temporal** | Diurnal heatmap (hour×workload), weekly patterns, rolling mean/std drift |
| **Correlations** | Full correlation matrix, interactive X/Y scatter, top correlated pairs |
| **Distributions** | Violin+box plots, overlaid histograms with bin slider, quantile table |
| **Anomalies** | XID error timeline, violation stacked bars, ECC error trends |
| **Profiling** | 2D PCA scatter (workload separability), explained variance, feature loadings |

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Node.js 18+ and [Bun](https://bun.sh/)

### Installation

```bash
# Clone
git clone <repo-url> && cd gpu-behavioral-profiling

# Python dependencies
uv sync

# Dashboard dependencies
cd dashboard && bun install && cd ..

# Environment
cp .env.example .env
# Edit .env with your Prometheus/DCGM exporter URLs
```

### Running

**1. Collect data** (requires access to your Prometheus/DCGM endpoints):

```bash
uv run python scraper/scrape.py --daemon
```

**2. Preprocess** (generates Parquet files from raw CSVs):

```bash
uv run python scraper/preprocess.py
```

This produces 5 files in `data/processed/`:

| File | Contents |
|---|---|
| `telemetry_wide.parquet` | Full wide-format data (one row per timestamp×GPU) |
| `hourly_agg.parquet` | Hourly means by workload/host/GPU |
| `stats_by_workload.parquet` | Per-workload descriptive stats |
| `correlation_matrix.parquet` | Pearson correlations of all numeric metrics |
| `quantiles_by_workload.parquet` | p1/p5/p25/p50/p75/p95/p99 per workload per metric |

**3. Start the API**:

```bash
uv run uvicorn api.main:app --reload
# Swagger docs at http://localhost:8000/docs
```

**4. Start the dashboard**:

```bash
cd dashboard && bun dev
# Opens at http://localhost:3000
```

## API Endpoints

26 endpoints across 8 route groups:

- `/api/overview/` — summary, KPIs, utilization timeline, GPU heatmap
- `/api/workloads/` — profiles, distributions, comparison, pipe breakdown
- `/api/gpus/` — inventory, health summary, selectable heatmap
- `/api/temporal/` — diurnal, weekly, drift detection
- `/api/correlations/` — matrix, scatter, top pairs
- `/api/distributions/` — violin, histogram, quantiles, metric list
- `/api/anomalies/` — XID events, violations, ECC trends
- `/api/profiling/` — PCA scatter, fingerprints

## DCGM Metrics Collected

41 metrics across 7 categories: performance/utilization, profiling (pipe activity), thermal/power, reliability/ECC, violations/throttling, interconnect (NVLink), and licensing. See `scraper/config.yaml` for the full list.

## Configuration

All sensitive configuration is in `.env`. See `.env.example` for required variables:

- `PROMETHEUS_URL` — Prometheus server for range queries
- `DCGM_EXPORTER_URL` — Direct DCGM exporter endpoint (fallback)
- `CORS_ORIGINS` — Allowed origins for the API
- `NEXT_PUBLIC_API_URL` — API base URL for the dashboard

## License

Private.
