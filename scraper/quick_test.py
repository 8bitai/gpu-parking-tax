#!/usr/bin/env python3
"""
Quick test: fetch one sample from DCGM Exporter and print it.

Usage:
    python quick_test.py
    python quick_test.py http://198.19.44.210:9400/metrics
"""

import sys
from scrape import parse_metrics_page, fetch_endpoint


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://198.19.44.210:9400/metrics"

    print(f"Fetching {url}...")
    try:
        text = fetch_endpoint(url)
        print(f"✓ Got {len(text)} bytes\n")
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)

    samples = parse_metrics_page(text)
    dcgm = [s for s in samples if s["metric"].startswith("DCGM_")]

    metrics = sorted(set(s["metric"] for s in dcgm))
    print(f"DCGM metrics found: {len(metrics)}")
    for m in metrics:
        count = sum(1 for s in dcgm if s["metric"] == m)
        print(f"  {m} ({count} GPUs)")

    # Show GPU details from first metric
    if dcgm:
        test_metric = metrics[0]
        print(f"\nGPU details (from {test_metric}):")
        for s in dcgm:
            if s["metric"] == test_metric:
                lb = s["labels"]
                print(f"  GPU {lb.get('gpu', '?')} | {lb.get('modelName', '?')} | "
                      f"val={s['value']} | container={lb.get('container', 'none')}")


if __name__ == "__main__":
    main()
