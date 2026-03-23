#!/usr/bin/env python3
"""
GPU Telemetry Scraper — Prometheus API → CSV pipeline.

Queries Prometheus for DCGM metrics, enriches with workload labels,
and appends to daily CSV files. Also supports direct DCGM exporter
scraping as fallback.

Usage:
    # One-shot: scrape last 60 seconds
    python scrape.py

    # Continuous: poll every 30 seconds
    python scrape.py --daemon

    # Backfill: scrape a specific time range from Prometheus history
    python scrape.py --backfill --start 2026-03-01T00:00:00Z --end 2026-03-20T00:00:00Z

    # Discovery: check what metrics are available
    python scrape.py --discover

    # Direct DCGM exporter mode (no Prometheus)
    python scrape.py --direct
"""

import argparse
import csv
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import urllib3
import requests
import yaml

from workload_classifier import WorkloadClassifier

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_all_metrics(config: dict) -> list[str]:
    """Flatten metric groups into a single list."""
    metrics = []
    for group in config["metrics"].values():
        metrics.extend(group)
    return metrics


def resolve_output_dir(config: dict) -> str:
    output_dir = config["output_dir"]
    if not Path(output_dir).is_absolute():
        output_dir = str(Path(__file__).parent / output_dir)
    return output_dir


# ── Prometheus API ───────────────────────────────────────────────────

def prom_get(prom_url: str, path: str, params: dict, verify_ssl: bool = False, timeout: int = 60) -> dict:
    """Make a GET request to Prometheus API."""
    url = f"{prom_url}{path}"
    resp = requests.get(url, params=params, verify=verify_ssl, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus error: {data.get('error', 'unknown')}")
    return data["data"]


def discover_prometheus(prom_url: str, verify_ssl: bool = False) -> list[str]:
    """Discover all DCGM metrics in Prometheus."""
    data = prom_get(prom_url, "/api/v1/label/__name__/values", {}, verify_ssl)
    return sorted(m for m in data if m.startswith("DCGM_"))


def query_range(
    prom_url: str,
    metric: str,
    start: datetime,
    end: datetime,
    step: str = "30s",
    verify_ssl: bool = False,
) -> list[dict]:
    """Query Prometheus range API for one metric."""
    try:
        data = prom_get(prom_url, "/api/v1/query_range", {
            "query": metric,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "step": step,
        }, verify_ssl)
        return data["result"]
    except Exception as e:
        log.error(f"Failed to query {metric}: {e}")
        return []


def scrape_prometheus(
    prom_url: str,
    metrics: list[str],
    label_keys: list[str],
    start: datetime,
    end: datetime,
    step: str = "30s",
    verify_ssl: bool = False,
) -> list[dict]:
    """Scrape all metrics from Prometheus for a time range."""
    rows = []
    for metric in metrics:
        results = query_range(prom_url, metric, start, end, step, verify_ssl)
        if not results:
            continue

        sample_count = 0
        for series in results:
            labels = series["metric"]
            for ts, val in series["values"]:
                row = {
                    "timestamp": datetime.fromtimestamp(
                        float(ts), tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "metric": metric,
                    "value": float(val),
                }
                for lk in label_keys:
                    row[lk] = labels.get(lk, "")
                rows.append(row)
                sample_count += 1

        log.info(f"  {metric}: {len(results)} series, {sample_count} samples")

    return rows


# ── Direct DCGM Exporter ────────────────────────────────────────────

METRIC_LINE_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\{([^}]*)\}\s+(.+)$')
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')


def parse_metrics_page(text: str) -> list[dict]:
    """Parse Prometheus exposition format."""
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE_RE.match(line)
        if not match:
            continue
        labels = {m.group(1): m.group(2) for m in LABEL_RE.finditer(match.group(2))}
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        samples.append({"metric": match.group(1), "labels": labels, "value": value})
    return samples


def scrape_direct(
    endpoints: list[dict],
    target_metrics: set[str],
    label_keys: list[str],
) -> list[dict]:
    """Scrape DCGM exporter endpoints directly."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for ep in endpoints:
        url = ep["url"]
        name = ep.get("name", url)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            samples = parse_metrics_page(resp.text)
        except Exception as e:
            log.error(f"Failed to scrape {name}: {e}")
            continue

        for s in samples:
            if s["metric"] not in target_metrics:
                continue
            row = {"timestamp": now, "metric": s["metric"], "value": s["value"]}
            for lk in label_keys:
                row[lk] = s["labels"].get(lk, "")
            rows.append(row)

        log.info(f"  {name}: {sum(1 for s in samples if s['metric'] in target_metrics)} samples")
    return rows


# ── Workload enrichment & CSV writing ───────────────────────────────

def enrich_with_workload(rows: list[dict], classifier: WorkloadClassifier) -> list[dict]:
    for row in rows:
        row["workload_type"] = classifier.classify(
            row.get("namespace", ""), row.get("pod", ""), row.get("container", ""),
        )
    return rows


def write_daily_csvs(rows: list[dict], output_dir: str, label_keys: list[str]):
    """Append rows to daily CSV files with deduplication."""
    if not rows:
        log.warning("No rows to write")
        return

    os.makedirs(output_dir, exist_ok=True)
    fieldnames = ["timestamp", "metric", "value"] + label_keys + ["workload_type"]

    # Group by date
    by_date: dict[str, list[dict]] = {}
    for row in rows:
        date_str = row["timestamp"][:10]
        by_date.setdefault(date_str, []).append(row)

    for date_str, day_rows in by_date.items():
        csv_path = Path(output_dir) / f"{date_str}.csv"
        file_exists = csv_path.exists()

        # Load existing keys for dedup
        existing_keys = set()
        if file_exists:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    existing_keys.add((r.get("timestamp", ""), r.get("metric", ""), r.get("UUID", "")))

        new_rows = []
        for row in day_rows:
            key = (row["timestamp"], row["metric"], row.get("UUID", ""))
            if key not in existing_keys:
                new_rows.append(row)
                existing_keys.add(key)

        if not new_rows:
            log.info(f"  {date_str}: all duplicates, skipped")
            continue

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(new_rows)

        log.info(f"  {date_str}: wrote {len(new_rows)} new rows → {csv_path}")


# ── Run modes ────────────────────────────────────────────────────────

def run_once(config: dict, start: datetime = None, end: datetime = None, direct: bool = False):
    """Single scrape cycle."""
    metrics = get_all_metrics(config)
    label_keys = config["label_keys"]
    output_dir = resolve_output_dir(config)
    classifier = WorkloadClassifier(config.get("workload_rules", []))

    if direct:
        log.info(f"Scraping {len(config['dcgm_endpoints'])} endpoint(s) directly...")
        rows = scrape_direct(config["dcgm_endpoints"], set(metrics), label_keys)
    else:
        prom_url = config["prometheus_url"]
        verify_ssl = config.get("verify_ssl", False)
        step = config.get("query_step", "30s")
        lookback = config.get("query_lookback", 60)

        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - timedelta(seconds=lookback)

        log.info(f"Querying Prometheus: {len(metrics)} metrics, {start} → {end}")
        rows = scrape_prometheus(prom_url, metrics, label_keys, start, end, step, verify_ssl)

    log.info(f"Fetched {len(rows)} total samples")
    rows = enrich_with_workload(rows, classifier)

    # Log workload distribution
    wl_counts = {}
    for r in rows:
        wl_counts[r["workload_type"]] = wl_counts.get(r["workload_type"], 0) + 1
    if wl_counts:
        log.info(f"Workload distribution: {wl_counts}")

    write_daily_csvs(rows, output_dir, label_keys)
    return len(rows)


def run_daemon(config: dict, interval: int = 30, direct: bool = False):
    """Continuous scraping loop."""
    log.info(f"Starting daemon mode, polling every {interval}s (Ctrl+C to stop)")
    cycle = 0
    while True:
        cycle += 1
        log.info(f"--- Cycle {cycle} ---")
        try:
            run_once(config, direct=direct)
        except Exception:
            log.exception("Scrape cycle failed")
        time.sleep(interval)


def run_backfill(config: dict, start: datetime, end: datetime, chunk_hours: int = 2):
    """Backfill historical data from Prometheus in chunks."""
    total_hours = (end - start).total_seconds() / 3600
    log.info(f"Backfilling {total_hours:.1f} hours from {start} to {end} in {chunk_hours}h chunks")

    total_rows = 0
    current = start
    chunk_num = 0
    total_chunks = int(total_hours / chunk_hours) + 1

    while current < end:
        chunk_num += 1
        chunk_end = min(current + timedelta(hours=chunk_hours), end)
        log.info(f"Chunk {chunk_num}/{total_chunks}: {current} → {chunk_end}")
        rows = run_once(config, start=current, end=chunk_end)
        total_rows += rows
        current = chunk_end

    log.info(f"Backfill complete: {total_rows} total rows")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPU Telemetry Scraper")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=None, help="Poll interval (seconds)")
    parser.add_argument("--backfill", action="store_true", help="Backfill historical data")
    parser.add_argument("--start", type=str, help="Start time (ISO format)")
    parser.add_argument("--end", type=str, help="End time (ISO format)")
    parser.add_argument("--direct", action="store_true", help="Scrape DCGM exporter directly (no Prometheus)")
    parser.add_argument("--discover", action="store_true", help="Discover available metrics")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.discover:
        configured = set(get_all_metrics(config))

        if args.direct:
            from scrape import parse_metrics_page
            for ep in config["dcgm_endpoints"]:
                resp = requests.get(ep["url"], timeout=15)
                samples = parse_metrics_page(resp.text)
                available = sorted(set(s["metric"] for s in samples if s["metric"].startswith("DCGM_")))
                print(f"\nEndpoint: {ep.get('name', ep['url'])}")
                for m in available:
                    tag = "✓" if m in configured else " "
                    print(f"  [{tag}] {m}")
        else:
            prom_url = config["prometheus_url"]
            verify_ssl = config.get("verify_ssl", False)
            available = discover_prometheus(prom_url, verify_ssl)
            print(f"\nPrometheus: {prom_url}")
            print(f"Found {len(available)} DCGM metrics:\n")
            for m in available:
                tag = "✓" if m in configured else " "
                print(f"  [{tag}] {m}")

            present = configured & set(available)
            missing = configured - set(available)
            print(f"\nConfigured: {len(configured)} | Available: {len(available)} | Matched: {len(present)}")
            if missing:
                print(f"\n⚠ Configured but NOT in Prometheus:")
                for m in sorted(missing):
                    print(f"  - {m}")
        return

    interval = args.interval or config.get("poll_interval", 30)

    if args.backfill:
        if not args.start or not args.end:
            parser.error("--backfill requires --start and --end")
        start = datetime.fromisoformat(args.start)
        end = datetime.fromisoformat(args.end)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        run_backfill(config, start, end)
    elif args.daemon:
        run_daemon(config, interval=interval, direct=args.direct)
    else:
        run_once(config, direct=args.direct)


if __name__ == "__main__":
    main()
