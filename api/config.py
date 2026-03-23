from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

PARQUET_FILES = {
    "wide": DATA_DIR / "telemetry_wide.parquet",
    "hourly": DATA_DIR / "hourly_agg.parquet",
    "stats": DATA_DIR / "stats_by_workload.parquet",
    "correlation": DATA_DIR / "correlation_matrix.parquet",
    "quantiles": DATA_DIR / "quantiles_by_workload.parquet",
}

MAX_SCATTER_POINTS = 5000

WORKLOAD_COLORS = {
    "llm_inference": "#8b5cf6",
    "embedding": "#06b6d4",
    "voice_ai": "#f59e0b",
    "computer_vision": "#10b981",
    "other": "#6b7280",
    "idle": "#374151",
}
