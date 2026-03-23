import logging
import math
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from api.config import PARQUET_FILES

log = logging.getLogger(__name__)

_cache: dict[str, pd.DataFrame] = {}


def load_all():
    """Load all parquet files into memory cache on startup."""
    for name, path in PARQUET_FILES.items():
        if path.exists():
            _cache[name] = pd.read_parquet(path)
            log.info(f"Loaded {name}: {_cache[name].shape}")
        else:
            log.warning(f"Missing parquet: {path}")


def get(name: str) -> pd.DataFrame | None:
    """Get a cached dataframe by name."""
    return _cache.get(name)


def get_wide_sample(n: int = 5000, workload: str | None = None) -> pd.DataFrame:
    """Get a downsampled slice of the wide dataframe."""
    df = _cache.get("wide")
    if df is None:
        return pd.DataFrame()
    if workload:
        df = df[df["workload_type"] == workload]
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    return df


def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric metric column names."""
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


def jsonify(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {jsonify(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj
