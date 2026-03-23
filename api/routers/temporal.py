from fastapi import APIRouter, Query

from api import data_loader

router = APIRouter()


@router.get("/diurnal")
def diurnal(metric: str = Query(default="DCGM_FI_DEV_GPU_UTIL")):
    """Diurnal heatmap: hour-of-day × workload type."""
    df = data_loader.get("hourly")
    if df is None:
        return {"error": "Data not loaded"}

    if metric not in df.columns:
        return {"error": f"Metric {metric} not found"}

    df = df.copy()
    df["hour_of_day"] = df["hour"].dt.hour

    pivot = df.groupby(["workload_type", "hour_of_day"], observed=True)[metric].mean().reset_index()
    workloads = sorted(pivot["workload_type"].unique())
    hours = list(range(24))

    matrix = []
    for wl in workloads:
        row = []
        wdf = pivot[pivot["workload_type"] == wl].set_index("hour_of_day")
        for h in hours:
            row.append(round(wdf.loc[h, metric], 2) if h in wdf.index else None)
        matrix.append(row)

    return data_loader.jsonify({"workloads": workloads, "hours": hours, "matrix": matrix, "metric": metric})


@router.get("/weekly")
def weekly(metric: str = Query(default="DCGM_FI_DEV_GPU_UTIL")):
    """Weekly pattern: metric by day-of-week."""
    df = data_loader.get("hourly")
    if df is None:
        return {"error": "Data not loaded"}

    if metric not in df.columns:
        return {"error": f"Metric {metric} not found"}

    df = df.copy()
    df["day_of_week"] = df["hour"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    result = {}
    for wl, grp in df.groupby("workload_type", observed=True):
        daily = grp.groupby("day_of_week")[metric].mean()
        result[wl] = {d: round(daily.get(d, 0), 2) for d in day_order}

    return data_loader.jsonify({"days": day_order, "workloads": result, "metric": metric})


@router.get("/drift")
def drift(metric: str = Query(default="DCGM_FI_DEV_GPU_UTIL")):
    """Rolling mean/std over time for drift detection."""
    df = data_loader.get("hourly")
    if df is None:
        return {"error": "Data not loaded"}

    if metric not in df.columns:
        return {"error": f"Metric {metric} not found"}

    daily = df.groupby(df["hour"].dt.date)[metric].agg(["mean", "std"]).reset_index()
    daily.columns = ["date", "mean", "std"]
    daily["date"] = daily["date"].astype(str)

    return data_loader.jsonify({
        "dates": daily["date"].tolist(),
        "mean": daily["mean"].round(2).tolist(),
        "std": daily["std"].round(2).tolist(),
        "metric": metric,
    })
