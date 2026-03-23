from fastapi import APIRouter, Query

from api import data_loader

router = APIRouter()


@router.get("/violin")
def violin(metric: str = Query(...)):
    """Violin/box plot data: sampled values per workload for a metric."""
    df = data_loader.get_wide_sample(n=5000)
    if df.empty:
        return {"error": "Data not loaded"}

    if metric not in df.columns:
        return {"error": f"Metric {metric} not found"}

    result = {}
    for wl, grp in df.groupby("workload_type", observed=True):
        vals = grp[metric].dropna()
        result[wl] = [round(v, 4) for v in vals.tolist()]
    return data_loader.jsonify({"metric": metric, "data": result})


@router.get("/histogram")
def histogram(metric: str = Query(...), bins: int = Query(default=50)):
    """Histogram data per workload type."""
    df = data_loader.get_wide_sample(n=5000)
    if df.empty:
        return {"error": "Data not loaded"}

    if metric not in df.columns:
        return {"error": f"Metric {metric} not found"}

    import numpy as np

    all_vals = df[metric].dropna()
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    result = {}
    for wl, grp in df.groupby("workload_type", observed=True):
        vals = grp[metric].dropna()
        counts, _ = np.histogram(vals, bins=bin_edges)
        result[wl] = counts.tolist()

    return data_loader.jsonify({
        "metric": metric,
        "bin_edges": [round(b, 4) for b in bin_edges.tolist()],
        "data": result,
    })


@router.get("/quantiles")
def quantiles(metric: str = Query(default=None)):
    """Quantile table from pre-computed data."""
    df = data_loader.get("quantiles")
    if df is None:
        return {"error": "Data not loaded"}

    if metric:
        df = df[df["metric"] == metric]

    return data_loader.jsonify(df.to_dict(orient="records"))


@router.get("/metrics")
def available_metrics():
    """List available numeric metrics."""
    df = data_loader.get("wide")
    if df is None:
        return {"error": "Data not loaded"}
    return data_loader.jsonify(data_loader.get_numeric_cols(df))
