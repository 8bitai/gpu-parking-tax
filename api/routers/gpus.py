from fastapi import APIRouter, Query

from api import data_loader

router = APIRouter()


@router.get("/inventory")
def inventory():
    """List all GPUs with basic info."""
    df = data_loader.get("wide")
    if df is None:
        return {"error": "Data not loaded"}

    cols = ["UUID", "gpu", "Hostname", "modelName"]
    cols = [c for c in cols if c in df.columns]
    inv = df[cols].drop_duplicates().to_dict(orient="records")
    return data_loader.jsonify(inv)


@router.get("/health-summary")
def health_summary():
    """Per-GPU health metrics: mean temp, power, ECC errors, etc."""
    df = data_loader.get("wide")
    if df is None:
        return {"error": "Data not loaded"}

    health_cols = {
        "DCGM_FI_DEV_GPU_UTIL": "mean_util",
        "DCGM_FI_DEV_GPU_TEMP": "mean_temp",
        "DCGM_FI_DEV_POWER_USAGE": "mean_power",
        "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL": "ecc_sbe_total",
        "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": "ecc_dbe_total",
        "DCGM_FI_DEV_XID_ERRORS": "xid_errors",
        "DCGM_FI_DEV_POWER_VIOLATION": "power_violations",
        "DCGM_FI_DEV_THERMAL_VIOLATION": "thermal_violations",
    }

    available = {k: v for k, v in health_cols.items() if k in df.columns}
    group_cols = [c for c in ["gpu", "Hostname", "UUID"] if c in df.columns]

    agg_funcs = {}
    for col, label in available.items():
        if "violation" in label or "ecc" in label or "xid" in label:
            agg_funcs[col] = "max"
        else:
            agg_funcs[col] = "mean"

    result = df.groupby(group_cols, observed=True).agg(agg_funcs).reset_index()
    result = result.rename(columns=available)
    result = result.round(2)
    return data_loader.jsonify(result.to_dict(orient="records"))


@router.get("/heatmap")
def heatmap(metric: str = Query(default="DCGM_FI_DEV_GPU_UTIL")):
    """GPU × time heatmap for any metric."""
    df = data_loader.get("hourly")
    if df is None:
        return {"error": "Data not loaded"}

    if metric not in df.columns:
        return {"error": f"Metric {metric} not found"}

    df = df.copy()
    df["date"] = df["hour"].dt.date.astype(str)

    pivot = df.groupby(["gpu", "date"], observed=True)[metric].mean().reset_index()
    gpus = sorted(pivot["gpu"].unique().tolist())
    dates = sorted(pivot["date"].unique().tolist())

    matrix = []
    for g in gpus:
        row = []
        gdf = pivot[pivot["gpu"] == g].set_index("date")
        for d in dates:
            row.append(round(float(gdf.loc[d, metric]), 2) if d in gdf.index else None)
        matrix.append(row)

    return data_loader.jsonify({"gpus": gpus, "dates": dates, "matrix": matrix, "metric": metric})
