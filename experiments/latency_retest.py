#!/usr/bin/env python3
"""
Focused latency retest for CUDA_DISABLE_PERF_BOOST.

Runs vLLM with the flag on, sends requests in two phases:
  1) Cold burst: 5 requests after a 60s idle soak (captures first-request ramp)
  2) Warm burst: 200 requests back-to-back (steady-state latency)

Then repeats without the flag as a control.

This isolates whether the H100 p99 outlier is a consistent first-request
clock-ramp effect or just noise.

Usage:
  python latency_retest.py --gpu 0
  python latency_retest.py --gpu 0 --model Qwen/Qwen2.5-7B
  python latency_retest.py --gpu 0 --hf-cache /workspace/.cache/huggingface
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-7B"
VLLM_PORT = 8193
IDLE_SOAK_S = 60       # seconds to idle before cold burst
N_COLD = 5             # requests in the cold burst
N_WARM = 50            # requests in the warm burst
OUTPUT_DIR = Path("data/raw")


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def setup_hf_cache(cache_root=None):
    root = (cache_root or "").strip() or None
    if root is None and os.environ.get("HF_HOME"):
        return None
    if root is None and os.path.isdir("/workspace"):
        root = "/workspace/.cache/huggingface"
    if root:
        os.environ["HF_HOME"] = root
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(root, "hub")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(root, "hub")
        log(f"HF cache → {root}")
        return root
    return None


def get_gpu_info(gpu_id):
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu_id}",
         "--query-gpu=name,uuid,driver_version",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    parts = result.stdout.strip().split(", ")
    return {"name": parts[0], "uuid": parts[1], "driver": parts[2]}


def start_vllm(gpu_id, model, env_flag=False):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if env_flag:
        env["CUDA_DISABLE_PERF_BOOST"] = "1"
    else:
        env.pop("CUDA_DISABLE_PERF_BOOST", None)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--dtype", "float16",
        "--port", str(VLLM_PORT),
        "--max-model-len", "4096",
        "--no-enable-log-requests",
    ]
    log(f"  Starting vLLM (flag={'ON' if env_flag else 'OFF'})")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def wait_for_vllm(timeout=600):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"http://localhost:{VLLM_PORT}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            time.sleep(5)
    return False


def stop_vllm(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def send_request(model):
    import urllib.request
    body = json.dumps({
        "model": model,
        "prompt": "Explain the concept of GPU power management in one paragraph.",
        "max_tokens": 128,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{VLLM_PORT}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def measure_burst(model, n, label=""):
    latencies = []
    for i in range(n):
        t0 = time.time()
        try:
            send_request(model)
            elapsed = time.time() - t0
            latencies.append(elapsed)
            if (i + 1) % 25 == 0 or i < 5:
                log(f"    [{label}] req {i+1}/{n}: {elapsed*1000:.1f}ms")
        except Exception as e:
            log(f"    [{label}] req {i+1} FAILED: {e}")
            latencies.append(None)
    valid = [l for l in latencies if l is not None]
    return latencies, valid


def percentile(data, p):
    s = sorted(data)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def summarize(valid):
    import statistics
    return {
        "n": len(valid),
        "mean": statistics.mean(valid),
        "p50": percentile(valid, 50),
        "p95": percentile(valid, 95),
        "p99": percentile(valid, 99),
        "min": min(valid),
        "max": max(valid),
        "stdev": statistics.stdev(valid) if len(valid) > 1 else 0,
    }


def run_condition(gpu_id, model, env_flag, results):
    label = "flag_on" if env_flag else "baseline"
    log(f"\n{'='*60}")
    log(f"Condition: {label}")
    log(f"{'='*60}")

    proc = start_vllm(gpu_id, model, env_flag=env_flag)
    try:
        log("  Waiting for vLLM to be ready...")
        if not wait_for_vllm():
            log("  ERROR: vLLM failed to start")
            return
        log("  vLLM ready.")

        # Warmup: 3 throwaway requests to ensure model is loaded and KV cache primed
        log("  Warmup (3 requests)...")
        for _ in range(3):
            send_request(model)
        log("  Warmup done.")

        # Idle soak: let GPU settle at idle clocks
        log(f"  Idle soak: {IDLE_SOAK_S}s...")
        time.sleep(IDLE_SOAK_S)

        # Phase 1: Cold burst (first requests after idle)
        log(f"  Cold burst: {N_COLD} requests after {IDLE_SOAK_S}s idle...")
        cold_all, cold_valid = measure_burst(model, N_COLD, label=f"{label}/cold")
        if cold_valid:
            s = summarize(cold_valid)
            log(f"  Cold: mean={s['mean']*1000:.1f}ms, "
                f"min={s['min']*1000:.1f}ms, max={s['max']*1000:.1f}ms")

        # Phase 2: Warm burst (back-to-back, no idle gap)
        log(f"  Warm burst: {N_WARM} requests back-to-back...")
        warm_all, warm_valid = measure_burst(model, N_WARM, label=f"{label}/warm")
        if warm_valid:
            s = summarize(warm_valid)
            log(f"  Warm: mean={s['mean']*1000:.1f}ms, p50={s['p50']*1000:.1f}ms, "
                f"p95={s['p95']*1000:.1f}ms, p99={s['p99']*1000:.1f}ms")

        results.append({
            "condition": label,
            "env_flag": env_flag,
            "idle_soak_s": IDLE_SOAK_S,
            "cold": {"all": cold_all, "stats": summarize(cold_valid) if cold_valid else None},
            "warm": {"all": warm_all, "stats": summarize(warm_valid) if warm_valid else None},
        })
    finally:
        log("  Stopping vLLM...")
        stop_vllm(proc)
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Latency retest for CUDA_DISABLE_PERF_BOOST")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-cache", default=None)
    args = parser.parse_args()

    setup_hf_cache(args.hf_cache)
    gpu_info = get_gpu_info(args.gpu)
    log(f"GPU {args.gpu}: {gpu_info['name']} ({gpu_info['uuid']})")
    log(f"Driver: {gpu_info['driver']}")
    log(f"Model: {args.model}")

    results = []

    # Run flag_on first (the one with the outlier), then baseline as control
    run_condition(args.gpu, args.model, env_flag=True, results=results)
    run_condition(args.gpu, args.model, env_flag=False, results=results)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"latency_retest_{ts}.json"

    output = {
        "experiment": "latency_retest",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_info,
        "model": args.model,
        "idle_soak_s": IDLE_SOAK_S,
        "n_cold": N_COLD,
        "n_warm": N_WARM,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {out_path}")

    # Print summary
    log("\n" + "="*60)
    log("SUMMARY")
    log("="*60)
    for r in results:
        cond = r["condition"]
        if r["cold"] and r["cold"]["stats"]:
            cs = r["cold"]["stats"]
            log(f"  {cond} cold ({cs['n']} reqs):  "
                f"mean={cs['mean']*1000:.1f}ms  min={cs['min']*1000:.1f}ms  max={cs['max']*1000:.1f}ms")
        if r["warm"] and r["warm"]["stats"]:
            ws = r["warm"]["stats"]
            log(f"  {cond} warm ({ws['n']} reqs):  "
                f"mean={ws['mean']*1000:.1f}ms  p50={ws['p50']*1000:.1f}ms  "
                f"p95={ws['p95']*1000:.1f}ms  p99={ws['p99']*1000:.1f}ms  "
                f"max={ws['max']*1000:.1f}ms")


if __name__ == "__main__":
    main()
