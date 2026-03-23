from fastapi import APIRouter

from api import data_loader

router = APIRouter()

RADAR_FEATURES = [
    "tensor_dominance", "fp16_fp32_ratio", "compute_memory_ratio",
    "clock_ratio", "thermal_headroom", "power_efficiency", "memory_utilization_pct",
]


@router.get("/profiles")
def profiles():
    """Per-workload mean of radar features."""
    stats = data_loader.get("stats")
    if stats is None:
        return {"error": "Data not loaded"}

    result = {}
    for wl, grp in stats.groupby("workload_type"):
        radar = grp[grp["metric"].isin(RADAR_FEATURES)].set_index("metric")
        result[wl] = {m: round(radar.loc[m, "mean"], 4) if m in radar.index else 0 for m in RADAR_FEATURES}
    return data_loader.jsonify(result)


@router.get("/distributions")
def distributions():
    """Violin plot data: sampled values per workload for key metrics."""
    df = data_loader.get_wide_sample(n=3000)
    if df.empty:
        return {"error": "Data not loaded"}

    metrics = [
        "DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_POWER_USAGE",
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "DCGM_FI_PROF_SM_ACTIVE",
        "tensor_dominance", "fp16_fp32_ratio",
    ]
    metrics = [m for m in metrics if m in df.columns]

    result = {}
    for m in metrics:
        result[m] = {}
        for wl, grp in df.groupby("workload_type", observed=True):
            vals = grp[m].dropna().tolist()
            result[m][wl] = [round(v, 4) for v in vals]
    return data_loader.jsonify(result)


@router.get("/compare")
def compare():
    """Comparison table: mean/std/median per workload per metric."""
    stats = data_loader.get("stats")
    if stats is None:
        return {"error": "Data not loaded"}

    records = stats.to_dict(orient="records")
    return data_loader.jsonify(records)


@router.get("/pipe-breakdown")
def pipe_breakdown():
    """Stacked bar of tensor/fp16/fp32/fp64 pipe activity per workload."""
    stats = data_loader.get("stats")
    if stats is None:
        return {"error": "Data not loaded"}

    pipe_metrics = [
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
        "DCGM_FI_PROF_PIPE_FP16_ACTIVE",
        "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
        "DCGM_FI_PROF_PIPE_FP64_ACTIVE",
    ]

    result = {}
    for wl, grp in stats.groupby("workload_type"):
        row = {}
        for m in pipe_metrics:
            match = grp[grp["metric"] == m]
            row[m] = round(match["mean"].iloc[0], 4) if len(match) > 0 else 0
        result[wl] = row
    return data_loader.jsonify(result)
