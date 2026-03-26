#!/usr/bin/env python3
"""
Preprocessor: converts raw daily CSVs (long format) into wide-format Parquet files
suitable for analysis.

Long format: one row per (timestamp, metric, GPU)
Wide format: one row per (timestamp, GPU), metrics as columns

Also computes derived features:
- power_efficiency: GPU_UTIL / POWER_USAGE
- thermal_headroom: max rated temp - current temp
- memory_utilization_pct: FB_USED / (FB_USED + FB_FREE) * 100
- clock_ratio: SM_CLOCK / max_sm_clock (measures throttling)
- tensor_dominance: TENSOR_ACTIVE / SM_ACTIVE (LLM vs non-LLM fingerprint)
- fp16_fp32_ratio: FP16_ACTIVE / (FP16_ACTIVE + FP32_ACTIVE) (precision mix)
- compute_memory_ratio: SM_ACTIVE / DRAM_ACTIVE (compute vs memory bound)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# H100 specs for derived features
H100_MAX_SM_CLOCK = 1980  # MHz (boost)
H100_MAX_TEMP = 83  # °C (throttle point)
H100_TDP = 700  # W

L40S_MAX_SM_CLOCK = 2520  # MHz (boost)
L40S_MAX_TEMP = 90  # °C
L40S_TDP = 350  # W


def load_raw_csvs(raw_dir: str, date_range: tuple[str, str] = None) -> pd.DataFrame:
    """Load all raw CSV files, optionally filtered by date range."""
    raw_path = Path(raw_dir)
    csv_files = sorted(raw_path.glob("*.csv"))

    if date_range:
        start, end = date_range
        csv_files = [f for f in csv_files if start <= f.stem <= end]

    if not csv_files:
        log.warning(f"No CSV files found in {raw_dir}")
        return pd.DataFrame()

    log.info(f"Loading {len(csv_files)} CSV files from {raw_dir}")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"])
            dfs.append(df)
            log.info(f"  {f.name}: {len(df)} rows")
        except Exception as e:
            log.error(f"  {f.name}: failed to load - {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot from long format (one row per metric) to wide format (metrics as columns)."""
    # Identify the index columns (everything except metric and value)
    id_cols = ["timestamp", "UUID", "gpu", "device", "modelName", "Hostname",
               "namespace", "pod", "container", "workload_type",
               "DCGM_FI_DRIVER_VERSION", "DCGM_FI_DEV_VBIOS_VERSION",
               "DCGM_FI_DEV_SERIAL", "pci_bus_id"]

    # Keep only columns that exist in the dataframe
    id_cols = [c for c in id_cols if c in df.columns]

    # Pivot
    wide = df.pivot_table(
        index=id_cols,
        columns="metric",
        values="value",
        aggfunc="first",  # Should be unique after dedup, but just in case
    ).reset_index()

    # Flatten column names
    wide.columns.name = None

    return wide


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed features useful for analysis."""
    # Determine GPU model for spec lookups
    is_h100 = df["modelName"].str.contains("H100", case=False, na=False)

    max_sm = is_h100.map({True: H100_MAX_SM_CLOCK, False: L40S_MAX_SM_CLOCK})
    max_temp = is_h100.map({True: H100_MAX_TEMP, False: L40S_MAX_TEMP})

    # Clock throttling ratio (1.0 = full boost, <1.0 = throttled)
    if "DCGM_FI_DEV_SM_CLOCK" in df.columns:
        df["clock_ratio"] = df["DCGM_FI_DEV_SM_CLOCK"] / max_sm

    # Thermal headroom (degrees below throttle point)
    if "DCGM_FI_DEV_GPU_TEMP" in df.columns:
        df["thermal_headroom"] = max_temp - df["DCGM_FI_DEV_GPU_TEMP"]

    # Power efficiency (utilization per watt)
    if "DCGM_FI_DEV_GPU_UTIL" in df.columns and "DCGM_FI_DEV_POWER_USAGE" in df.columns:
        df["power_efficiency"] = df["DCGM_FI_DEV_GPU_UTIL"] / df["DCGM_FI_DEV_POWER_USAGE"].clip(lower=1)

    # Memory utilization percentage
    if "DCGM_FI_DEV_FB_USED" in df.columns and "DCGM_FI_DEV_FB_FREE" in df.columns:
        total_fb = df["DCGM_FI_DEV_FB_USED"] + df["DCGM_FI_DEV_FB_FREE"]
        df["memory_utilization_pct"] = (df["DCGM_FI_DEV_FB_USED"] / total_fb.clip(lower=1)) * 100

    # Tensor dominance: how much of SM activity is tensor core work
    # High = LLM/transformer workload, Low = traditional compute
    if "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE" in df.columns and "DCGM_FI_PROF_SM_ACTIVE" in df.columns:
        df["tensor_dominance"] = (
            df["DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"] / df["DCGM_FI_PROF_SM_ACTIVE"].clip(lower=0.001)
        )

    # FP16/FP32 ratio: precision mix fingerprint
    if "DCGM_FI_PROF_PIPE_FP16_ACTIVE" in df.columns and "DCGM_FI_PROF_PIPE_FP32_ACTIVE" in df.columns:
        fp_total = df["DCGM_FI_PROF_PIPE_FP16_ACTIVE"] + df["DCGM_FI_PROF_PIPE_FP32_ACTIVE"]
        df["fp16_fp32_ratio"] = df["DCGM_FI_PROF_PIPE_FP16_ACTIVE"] / fp_total.clip(lower=0.001)

    # Compute vs memory bound indicator
    if "DCGM_FI_PROF_SM_ACTIVE" in df.columns and "DCGM_FI_PROF_DRAM_ACTIVE" in df.columns:
        df["compute_memory_ratio"] = (
            df["DCGM_FI_PROF_SM_ACTIVE"] / df["DCGM_FI_PROF_DRAM_ACTIVE"].clip(lower=0.001)
        )

    return df


def process(raw_dir: str, output_dir: str, date_range: tuple[str, str] = None):
    """Full preprocessing pipeline."""
    df = load_raw_csvs(raw_dir, date_range)
    if df.empty:
        log.warning("No data to process")
        return

    log.info(f"Total raw rows: {len(df)}")
    log.info(f"Metrics: {df['metric'].nunique()}")
    log.info(f"GPUs: {df['UUID'].nunique()}")
    log.info(f"Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    log.info(f"Workload types: {df['workload_type'].value_counts().to_dict()}")

    # Pivot to wide format
    wide = long_to_wide(df)
    log.info(f"Wide format: {len(wide)} rows × {len(wide.columns)} columns")

    # Add derived features
    wide = add_derived_features(wide)

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    parquet_path = out_path / "telemetry_wide.parquet"
    wide.to_parquet(parquet_path, index=False, engine="pyarrow")
    log.info(f"Saved wide-format data to {parquet_path}")

    # Also save a summary stats file
    summary = compute_summary(wide)
    summary_path = out_path / "summary_stats.csv"
    summary.to_csv(summary_path)
    log.info(f"Saved summary stats to {summary_path}")

    # Pre-aggregated outputs for the dashboard API
    log.info("Computing pre-aggregated outputs for dashboard...")

    hourly = compute_hourly_agg(wide)
    hourly.to_parquet(out_path / "hourly_agg.parquet", index=False, engine="pyarrow")
    log.info(f"Saved hourly_agg.parquet ({len(hourly)} rows)")

    stats = compute_stats_by_workload(wide)
    stats.to_parquet(out_path / "stats_by_workload.parquet", index=False, engine="pyarrow")
    log.info(f"Saved stats_by_workload.parquet ({len(stats)} rows)")

    corr = compute_correlation_matrix(wide)
    corr.to_parquet(out_path / "correlation_matrix.parquet", engine="pyarrow")
    log.info(f"Saved correlation_matrix.parquet ({corr.shape})")

    quantiles = compute_quantiles_by_workload(wide)
    quantiles.to_parquet(out_path / "quantiles_by_workload.parquet", index=False, engine="pyarrow")
    log.info(f"Saved quantiles_by_workload.parquet ({len(quantiles)} rows)")


def get_numeric_metric_cols(df: pd.DataFrame) -> list[str]:
    """Return all numeric metric column names (DCGM_ + derived features)."""
    derived = [
        "clock_ratio", "thermal_headroom", "power_efficiency",
        "memory_utilization_pct", "tensor_dominance", "fp16_fp32_ratio",
        "compute_memory_ratio",
    ]
    numeric_dtypes = df.select_dtypes(include="number").columns
    return [
        c for c in df.columns
        if (c.startswith("DCGM_") or c in derived) and c in numeric_dtypes
    ]


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-workload-type summary statistics for all numeric metrics."""
    metric_cols = get_numeric_metric_cols(df)
    numeric_df = df[["workload_type"] + metric_cols]
    return numeric_df.groupby("workload_type").describe()


def compute_hourly_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly means grouped by (workload_type, Hostname, gpu, hour)."""
    metric_cols = get_numeric_metric_cols(df)
    df = df.copy()
    df["hour"] = df["timestamp"].dt.floor("h")
    group_cols = ["workload_type", "Hostname", "gpu", "hour"]
    group_cols = [c for c in group_cols if c in df.columns]
    agg = df.groupby(group_cols, observed=True)[metric_cols].mean().reset_index()
    return agg


def compute_stats_by_workload(df: pd.DataFrame) -> pd.DataFrame:
    """Per-workload descriptive stats (mean, std, min, max, count) for all metrics."""
    metric_cols = get_numeric_metric_cols(df)
    rows = []
    for wl, grp in df.groupby("workload_type", observed=True):
        for col in metric_cols:
            s = grp[col].dropna()
            if len(s) == 0:
                continue
            rows.append({
                "workload_type": wl,
                "metric": col,
                "count": len(s),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "max": s.max(),
            })
    return pd.DataFrame(rows)


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of numeric columns."""
    metric_cols = get_numeric_metric_cols(df)
    corr = df[metric_cols].corr()
    return corr


def compute_quantiles_by_workload(df: pd.DataFrame) -> pd.DataFrame:
    """Quantiles (p1/p5/p25/p50/p75/p95/p99) per workload per metric."""
    metric_cols = get_numeric_metric_cols(df)
    quantile_levels = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    rows = []
    for wl, grp in df.groupby("workload_type", observed=True):
        for col in metric_cols:
            s = grp[col].dropna()
            if len(s) == 0:
                continue
            qs = s.quantile(quantile_levels)
            row = {"workload_type": wl, "metric": col}
            for q, v in zip(quantile_levels, qs):
                row[f"p{int(q*100)}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw telemetry CSVs")
    parser.add_argument("--raw-dir", default="../data/telemetry", help="Raw CSV directory")
    parser.add_argument("--output-dir", default="../data/processed", help="Output directory")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    if not Path(raw_dir).is_absolute():
        raw_dir = str(Path(__file__).parent / raw_dir)

    output_dir = args.output_dir
    if not Path(output_dir).is_absolute():
        output_dir = str(Path(__file__).parent / output_dir)

    date_range = None
    if args.start and args.end:
        date_range = (args.start, args.end)

    process(raw_dir, output_dir, date_range)


if __name__ == "__main__":
    main()
