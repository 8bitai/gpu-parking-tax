from fastapi import APIRouter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from api import data_loader

router = APIRouter()

PCA_FEATURES = [
    "DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_GPU_TEMP",
    "DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP16_ACTIVE", "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
    "DCGM_FI_PROF_DRAM_ACTIVE", "DCGM_FI_DEV_MEM_COPY_UTIL",
    "tensor_dominance", "fp16_fp32_ratio", "compute_memory_ratio",
    "clock_ratio", "power_efficiency",
]


@router.get("/pca")
def pca_scatter():
    """2D PCA scatter: GPU-hour observations colored by workload."""
    df = data_loader.get_wide_sample(n=5000)
    if df.empty:
        return {"error": "Data not loaded"}

    features = [f for f in PCA_FEATURES if f in df.columns]
    subset = df[features + ["workload_type"]].dropna()

    if len(subset) < 10:
        return {"error": "Not enough data after dropping NaN"}

    X = subset[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    result = {}
    for wl in subset["workload_type"].unique():
        mask = subset["workload_type"].values == wl
        result[wl] = {
            "pc1": coords[mask, 0].round(4).tolist(),
            "pc2": coords[mask, 1].round(4).tolist(),
        }

    loadings = {}
    for i, feat in enumerate(features):
        loadings[feat] = {
            "pc1": round(float(pca.components_[0, i]), 4),
            "pc2": round(float(pca.components_[1, i]), 4),
        }

    return data_loader.jsonify({
        "data": result,
        "explained_variance": [round(v, 4) for v in pca.explained_variance_ratio_.tolist()],
        "loadings": loadings,
    })


@router.get("/fingerprints")
def fingerprints():
    """Workload fingerprint summary (mean of key derived features)."""
    stats = data_loader.get("stats")
    if stats is None:
        return {"error": "Data not loaded"}

    fingerprint_features = [
        "tensor_dominance", "fp16_fp32_ratio", "compute_memory_ratio",
        "clock_ratio", "power_efficiency", "memory_utilization_pct",
    ]

    result = {}
    for wl, grp in stats.groupby("workload_type"):
        fp = {}
        for feat in fingerprint_features:
            match = grp[grp["metric"] == feat]
            if len(match) > 0:
                fp[feat] = {"mean": round(match["mean"].iloc[0], 4), "std": round(match["std"].iloc[0], 4)}
        result[wl] = fp
    return data_loader.jsonify(result)
