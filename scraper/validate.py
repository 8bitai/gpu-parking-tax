#!/usr/bin/env python3
"""
Validation script — run FIRST before starting continuous collection.

Checks:
1. Prometheus connectivity
2. Available vs configured metrics
3. Label coverage
4. GPU inventory
5. Workload classification
6. Data retention (backfill potential)
7. Sample values sanity check
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import urllib3
import requests
import yaml

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from scrape import prom_get, get_all_metrics, discover_prometheus
from workload_classifier import WorkloadClassifier


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    prom_url = config["prometheus_url"]
    verify_ssl = config.get("verify_ssl", False)
    configured = set(get_all_metrics(config))
    label_keys = config["label_keys"]

    # 1. Connectivity
    print(f"\n{'='*60}")
    print(f"1. PROMETHEUS CONNECTIVITY")
    print(f"{'='*60}")
    try:
        data = prom_get(prom_url, "/api/v1/query", {"query": "up"}, verify_ssl, timeout=10)
        targets_up = len(data["result"])
        print(f"  ✓ Reachable at {prom_url}")
        print(f"  ℹ {targets_up} targets reporting")
    except Exception as e:
        print(f"  ✗ Cannot reach Prometheus: {e}")
        sys.exit(1)

    # 2. Metric availability
    print(f"\n{'='*60}")
    print(f"2. METRIC AVAILABILITY")
    print(f"{'='*60}")
    available = set(discover_prometheus(prom_url, verify_ssl))
    present = configured & available
    missing = configured - available
    extra = available - configured

    print(f"  Configured: {len(configured)} | Available: {len(available)} | Matched: {len(present)}")

    if missing:
        print(f"\n  ⚠ CONFIGURED BUT MISSING ({len(missing)}):")
        for m in sorted(missing):
            print(f"    - {m}")

    if extra:
        print(f"\n  + AVAILABLE BUT NOT CONFIGURED ({len(extra)}):")
        for m in sorted(extra):
            print(f"    + {m}")

    if not missing:
        print(f"  ✓ All {len(configured)} configured metrics are available")

    # 3-5. Use a test metric to check labels, GPUs, workloads
    test_metric = "DCGM_FI_DEV_GPU_UTIL"
    if test_metric not in present:
        test_metric = sorted(present)[0] if present else None

    if not test_metric:
        print("\n✗ No matching metrics found!")
        sys.exit(1)

    data = prom_get(prom_url, "/api/v1/query", {"query": test_metric}, verify_ssl)
    results = data["result"]

    # 3. Labels
    print(f"\n{'='*60}")
    print(f"3. LABEL COVERAGE (from {test_metric}, {len(results)} series)")
    print(f"{'='*60}")

    all_labels = set()
    for r in results:
        all_labels.update(r["metric"].keys())

    for lk in label_keys:
        coverage = sum(1 for r in results if r["metric"].get(lk)) / max(len(results), 1) * 100
        status = "✓" if coverage > 90 else "⚠" if coverage > 0 else "✗"
        print(f"  {status} {lk}: {coverage:.0f}%")

    unexpected = all_labels - set(label_keys) - {"__name__"}
    if unexpected:
        print(f"\n  ℹ Extra labels (not configured): {sorted(unexpected)}")

    # 4. GPU inventory
    print(f"\n{'='*60}")
    print(f"4. GPU INVENTORY")
    print(f"{'='*60}")

    models = {}
    hostnames = set()
    for r in results:
        lb = r["metric"]
        model = lb.get("modelName", "?")
        models[model] = models.get(model, 0) + 1
        hostnames.add(lb.get("Hostname", "?"))

    print(f"  Total GPUs: {len(results)}")
    for model, count in sorted(models.items()):
        print(f"  {model}: {count} GPUs")
    print(f"  Hostnames: {sorted(hostnames)}")

    print()
    for r in results:
        lb = r["metric"]
        val = r["value"][1]
        print(f"    GPU {lb.get('gpu', '?'):>2} | {lb.get('Hostname', '?'):>20} | "
              f"util={val:>3}% | {lb.get('container', 'none')[:25]:<25} | "
              f"ns={lb.get('namespace', 'none')[:20]}")

    # 5. Workload classification
    print(f"\n{'='*60}")
    print(f"5. WORKLOAD CLASSIFICATION")
    print(f"{'='*60}")

    classifier = WorkloadClassifier(config.get("workload_rules", []))
    by_type = {}
    for r in results:
        lb = r["metric"]
        wtype = classifier.classify(lb.get("namespace", ""), lb.get("pod", ""), lb.get("container", ""))
        by_type.setdefault(wtype, []).append(lb)

    for wtype, items in sorted(by_type.items()):
        print(f"\n  [{wtype}] — {len(items)} GPU(s)")
        for lb in items:
            print(f"    GPU {lb.get('gpu', '?'):>2} @ {lb.get('Hostname', '?')}: "
                  f"container={lb.get('container', 'none')}, pod={lb.get('pod', 'none')[:40]}")

    # 6. Data retention
    print(f"\n{'='*60}")
    print(f"6. DATA RETENTION (backfill potential)")
    print(f"{'='*60}")

    now = datetime.now(timezone.utc)
    oldest_found = None
    for days_back in [1, 3, 7, 14, 30, 60, 90]:
        start = now - timedelta(days=days_back)
        try:
            data = prom_get(prom_url, "/api/v1/query_range", {
                "query": f"count({test_metric})",
                "start": start.isoformat(),
                "end": (start + timedelta(minutes=5)).isoformat(),
                "step": "1m",
            }, verify_ssl, timeout=15)
            if data["result"] and data["result"][0]["values"]:
                print(f"  ✓ Data exists {days_back} days ago")
                oldest_found = days_back
            else:
                print(f"  ✗ No data {days_back} days ago")
                break
        except Exception:
            print(f"  ✗ Query failed for {days_back} days ago")
            break

    if oldest_found:
        print(f"\n  → You can backfill at least {oldest_found} days:")
        start_date = (now - timedelta(days=oldest_found)).strftime("%Y-%m-%dT00:00:00Z")
        end_date = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"    uv run python scraper/scrape.py --backfill --start {start_date} --end {end_date}")

    # 7. Sample values
    print(f"\n{'='*60}")
    print(f"7. SAMPLE VALUES (current snapshot)")
    print(f"{'='*60}")

    spot_check_metrics = [
        "DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_GPU_TEMP",
        "DCGM_FI_DEV_FB_USED", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "DCGM_FI_PROF_SM_ACTIVE",
    ]
    for m in spot_check_metrics:
        if m not in present:
            continue
        try:
            data = prom_get(prom_url, "/api/v1/query", {"query": m}, verify_ssl, timeout=10)
            vals = [float(r["value"][1]) for r in data["result"]]
            if vals:
                print(f"  {m}:")
                print(f"    min={min(vals):.2f}  max={max(vals):.2f}  mean={sum(vals)/len(vals):.2f}")
        except Exception:
            pass

    # Summary
    print(f"\n{'='*60}")
    print(f"NEXT STEPS")
    print(f"{'='*60}")
    if missing:
        print(f"  1. Remove {len(missing)} missing metrics from config.yaml")
    print(f"  {'2' if missing else '1'}. Review workload classification above")
    if oldest_found:
        print(f"  → Backfill {oldest_found} days of history (see command above)")
    print(f"  → Then start daemon: uv run python scraper/scrape.py --daemon")
    print()


if __name__ == "__main__":
    main()
