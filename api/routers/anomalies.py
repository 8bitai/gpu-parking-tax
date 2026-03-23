from fastapi import APIRouter

from api import data_loader

router = APIRouter()


@router.get("/xid-events")
def xid_events():
    """XID error timeline by GPU."""
    df = data_loader.get("wide")
    if df is None:
        return {"error": "Data not loaded"}

    xid_col = "DCGM_FI_DEV_XID_ERRORS"
    if xid_col not in df.columns:
        return {"error": "XID column not found"}

    xid_df = df[df[xid_col] > 0][["timestamp", "gpu", "Hostname", xid_col]].copy()
    xid_df["timestamp"] = xid_df["timestamp"].astype(str)
    return data_loader.jsonify(xid_df.to_dict(orient="records"))


@router.get("/violations-summary")
def violations_summary():
    """Violation counts per GPU by type."""
    df = data_loader.get("wide")
    if df is None:
        return {"error": "Data not loaded"}

    violation_cols = {
        "DCGM_FI_DEV_POWER_VIOLATION": "power",
        "DCGM_FI_DEV_THERMAL_VIOLATION": "thermal",
    }
    available = {k: v for k, v in violation_cols.items() if k in df.columns}
    if not available:
        return {"error": "No violation columns found"}

    group_cols = [c for c in ["gpu", "Hostname"] if c in df.columns]
    result = df.groupby(group_cols, observed=True)[list(available.keys())].sum().reset_index()
    result = result.rename(columns=available)
    return data_loader.jsonify(result.to_dict(orient="records"))


@router.get("/ecc-trend")
def ecc_trend():
    """ECC error accumulation over time."""
    df = data_loader.get("hourly")
    if df is None:
        return {"error": "Data not loaded"}

    ecc_cols = [c for c in df.columns if "ECC" in c]
    if not ecc_cols:
        return {"error": "No ECC columns found"}

    df = df.copy()
    df["date"] = df["hour"].dt.date.astype(str)
    daily = df.groupby("date")[ecc_cols].max().reset_index()
    return data_loader.jsonify({
        "dates": daily["date"].tolist(),
        "metrics": {col: daily[col].round(2).tolist() for col in ecc_cols},
    })
