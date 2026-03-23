from fastapi import APIRouter

from api import data_loader

router = APIRouter()


@router.get("/summary")
def summary():
    """High-level dataset summary."""
    df = data_loader.get("wide")
    if df is None:
        return {"error": "Data not loaded"}

    return data_loader.jsonify({
        "total_rows": len(df),
        "gpu_count": df["UUID"].nunique() if "UUID" in df.columns else 0,
        "node_count": df["Hostname"].nunique() if "Hostname" in df.columns else 0,
        "date_range": {
            "start": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
            "end": str(df["timestamp"].max()) if "timestamp" in df.columns else None,
        },
        "workload_counts": df["workload_type"].value_counts().to_dict() if "workload_type" in df.columns else {},
        "columns": list(df.columns),
    })


@router.get("/kpis")
def kpis():
    """Key performance indicators."""
    df = data_loader.get("wide")
    if df is None:
        return {"error": "Data not loaded"}

    result = {}
    for col, label in [
        ("DCGM_FI_DEV_GPU_UTIL", "mean_gpu_util"),
        ("DCGM_FI_DEV_POWER_USAGE", "mean_power_w"),
        ("DCGM_FI_DEV_GPU_TEMP", "mean_temp_c"),
        ("DCGM_FI_DEV_MEM_COPY_UTIL", "mean_mem_util"),
        ("memory_utilization_pct", "mean_fb_util_pct"),
    ]:
        if col in df.columns:
            result[label] = round(df[col].mean(), 2)
    return data_loader.jsonify(result)


@router.get("/utilization-timeline")
def utilization_timeline():
    """Hourly GPU utilization timeline by workload type."""
    df = data_loader.get("hourly")
    if df is None:
        return {"error": "Data not loaded"}

    util_col = "DCGM_FI_DEV_GPU_UTIL"
    if util_col not in df.columns:
        return {"error": f"{util_col} not found"}

    grouped = df.groupby(["hour", "workload_type"], observed=True)[util_col].mean().reset_index()
    grouped["hour"] = grouped["hour"].astype(str)

    result = {}
    for wl, grp in grouped.groupby("workload_type", observed=True):
        result[wl] = {
            "hours": grp["hour"].tolist(),
            "values": grp[util_col].round(2).tolist(),
        }
    return data_loader.jsonify(result)


@router.get("/gpu-heatmap")
def gpu_heatmap():
    """GPU × day heatmap of mean utilization."""
    df = data_loader.get("hourly")
    if df is None:
        return {"error": "Data not loaded"}

    util_col = "DCGM_FI_DEV_GPU_UTIL"
    if util_col not in df.columns or "hour" not in df.columns:
        return {"error": "Required columns not found"}

    df = df.copy()
    df["date"] = df["hour"].dt.date.astype(str)

    pivot = df.groupby(["gpu", "date"], observed=True)[util_col].mean().reset_index()
    gpus = sorted(pivot["gpu"].unique().tolist())
    dates = sorted(pivot["date"].unique().tolist())

    matrix = []
    for g in gpus:
        row = []
        gdf = pivot[pivot["gpu"] == g].set_index("date")
        for d in dates:
            row.append(round(float(gdf.loc[d, util_col]), 2) if d in gdf.index else None)
        matrix.append(row)

    return data_loader.jsonify({"gpus": gpus, "dates": dates, "matrix": matrix})
