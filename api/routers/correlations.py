from fastapi import APIRouter, Query

from api import data_loader
from api.config import MAX_SCATTER_POINTS

router = APIRouter()


@router.get("/matrix")
def matrix():
    """Full correlation matrix."""
    df = data_loader.get("correlation")
    if df is None:
        return {"error": "Data not loaded"}

    metrics = df.index.tolist()
    matrix = df.values.round(3).tolist()
    return data_loader.jsonify({"metrics": metrics, "matrix": matrix})


@router.get("/scatter")
def scatter(
    x: str = Query(..., description="X-axis metric"),
    y: str = Query(..., description="Y-axis metric"),
):
    """Scatter plot data for two metrics, colored by workload type."""
    df = data_loader.get_wide_sample(n=MAX_SCATTER_POINTS)
    if df.empty:
        return {"error": "Data not loaded"}

    for col in [x, y]:
        if col not in df.columns:
            return {"error": f"Metric {col} not found"}

    subset = df[[x, y, "workload_type"]].dropna()
    result = {}
    for wl, grp in subset.groupby("workload_type", observed=True):
        result[wl] = {
            "x": grp[x].round(4).tolist(),
            "y": grp[y].round(4).tolist(),
        }
    return data_loader.jsonify({"x_metric": x, "y_metric": y, "data": result})


@router.get("/top-pairs")
def top_pairs(n: int = Query(default=20)):
    """Top N most correlated metric pairs."""
    df = data_loader.get("correlation")
    if df is None:
        return {"error": "Data not loaded"}

    pairs = []
    metrics = df.columns.tolist()
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i < j:
                val = df.iloc[i, j]
                if val is not None and not (isinstance(val, float) and (val != val)):
                    pairs.append({"metric_1": m1, "metric_2": m2, "correlation": round(val, 4)})

    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return data_loader.jsonify(pairs[:n])
