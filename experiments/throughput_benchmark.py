#!/usr/bin/env python3
"""Sustained-throughput benchmark for CUDA_DISABLE_PERF_BOOST.

Measures aggregate decode throughput (output tokens/sec) under high concurrency
with the flag OFF (baseline) vs ON. This tests the claim that the flag imposes
"no steady-state penalty": if the flag lowered the clock ceiling *under load*
(not only at idle), throughput would regress here. A background sampler records
GPU power and SM clock during the load so we can confirm the GPU still boosts to
full clocks under active work with the flag set.

Runs both conditions back-to-back on one GPU and prints/saves the comparison.

Usage:
  uv run python experiments/throughput_benchmark.py --gpu 0
  uv run python experiments/throughput_benchmark.py --gpu 0 --concurrency 128 --n-requests 1024
  uv run python experiments/throughput_benchmark.py --gpu 0 --hf-cache /workspace/.cache/huggingface
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from utils import (
    DEFAULT_MODEL, log, percentile, query_nvidia_smi, send_vllm_request,
    setup_hf_cache, start_vllm_server, stop_vllm, wait_for_vllm,
)

PORT = 8200
OUTPUT_DIR = Path("data/raw")
PROMPT = ("Write a detailed, multi-paragraph technical explanation of how modern "
          "GPUs manage power, voltage, and clock frequency under varying load.")


def run_load(model, port, gpu_id, concurrency, n_requests, max_tokens):
    """Fire n_requests through a bounded thread pool; measure aggregate tok/s.

    A daemon thread samples nvidia-smi every 2 s during the load so we can
    report the power/SM-clock the GPU actually runs at under sustained work.
    """
    smi_samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            s = query_nvidia_smi(gpu_id)
            if s:
                smi_samples.append(s)
            time.sleep(2)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    def one_request():
        t0 = time.time()
        r = send_vllm_request(model, port, prompt=PROMPT, max_tokens=max_tokens)
        dt = time.time() - t0
        usage = r.get("usage") or {}
        return dt, usage.get("completion_tokens")

    latencies = []
    out_tokens = []
    failures = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one_request) for _ in range(n_requests)]
        for f in as_completed(futs):
            try:
                dt, ct = f.result()
                latencies.append(dt)
                if ct:
                    out_tokens.append(ct)
            except Exception as e:
                failures += 1
                if failures <= 5:
                    log(f"    request failed: {e}")
    wall = time.time() - t_start

    stop.set()
    th.join(timeout=5)

    powers = [s["power_w"] for s in smi_samples if s.get("power_w") is not None]
    clocks = [s["sm_clock_mhz"] for s in smi_samples
              if s.get("sm_clock_mhz") is not None]
    total_out = sum(out_tokens)
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "max_tokens": max_tokens,
        "completed": len(latencies),
        "failed": failures,
        "wall_s": wall,
        "total_output_tokens": total_out,
        "throughput_tok_s": (total_out / wall) if wall > 0 else None,
        "request_throughput_s": (len(latencies) / wall) if wall > 0 else None,
        "latency_mean_s": (sum(latencies) / len(latencies)) if latencies else None,
        "latency_p50_s": percentile(latencies, 50) if latencies else None,
        "latency_p99_s": percentile(latencies, 99) if latencies else None,
        "load_power_mean_w": (sum(powers) / len(powers)) if powers else None,
        "load_power_max_w": max(powers) if powers else None,
        "load_sm_clock_mean_mhz": (sum(clocks) / len(clocks)) if clocks else None,
        "load_sm_clock_max_mhz": max(clocks) if clocks else None,
        "n_smi_samples": len(powers),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Throughput comparison for CUDA_DISABLE_PERF_BOOST")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--n-requests", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--hf-cache", default=None)
    args = ap.parse_args()

    setup_hf_cache(args.hf_cache)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gpu = query_nvidia_smi(args.gpu) or {}
    log(f"GPU {args.gpu}: {gpu.get('gpu_name')}  driver {gpu.get('driver_version')}")
    log(f"Model: {args.model}  concurrency={args.concurrency}  "
        f"n_requests={args.n_requests}  max_tokens={args.max_tokens}")

    results = {}
    for env_flag, label in [(False, "baseline"), (True, "flag_on")]:
        log(f"\n{'='*60}\nCondition: {label}  (CUDA_DISABLE_PERF_BOOST "
            f"{'ON' if env_flag else 'OFF'})\n{'='*60}")
        proc = start_vllm_server([args.gpu], args.model, PORT,
                                 env_flag=env_flag, tp_size=1)
        try:
            log("  Waiting for vLLM...")
            if not wait_for_vllm(PORT):
                log("  ERROR: vLLM failed to start; skipping condition")
                continue
            log("  Warmup (8 requests)...")
            for _ in range(8):
                try:
                    send_vllm_request(args.model, PORT, max_tokens=64)
                except Exception:
                    pass
            log(f"  Running load...")
            res = run_load(args.model, PORT, args.gpu, args.concurrency,
                           args.n_requests, args.max_tokens)
            results[label] = res
            log(f"  {label}: {res['throughput_tok_s']:.1f} tok/s "
                f"({res['completed']}/{res['n_requests']} ok, "
                f"{res['failed']} failed) | "
                f"load power {res['load_power_mean_w']:.0f}W "
                f"(max {res['load_power_max_w']:.0f}) | "
                f"SM {res['load_sm_clock_mean_mhz']:.0f}MHz "
                f"(max {res['load_sm_clock_max_mhz']:.0f}) | "
                f"lat p50 {res['latency_p50_s']*1000:.0f}ms "
                f"p99 {res['latency_p99_s']*1000:.0f}ms")
        finally:
            log("  Stopping vLLM...")
            stop_vllm(proc)
            time.sleep(10)

    if "baseline" in results and "flag_on" in results:
        b = results["baseline"]["throughput_tok_s"]
        f = results["flag_on"]["throughput_tok_s"]
        if b and f:
            log(f"\n{'='*60}\nRESULT: baseline {b:.1f} tok/s  vs  "
                f"flag_on {f:.1f} tok/s  ({100*(f-b)/b:+.1f}%)\n{'='*60}")
            log("  If within a few %, the flag imposes no throughput penalty "
                "under load (the target claim). A large drop means it caps "
                "active clocks and the steady-state claim must be scoped.")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"throughput_{ts}.json"
    with open(out, "w") as fh:
        json.dump({
            "experiment": "throughput_flag_comparison",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gpu": gpu,
            "model": args.model,
            "results": results,
        }, fh, indent=2)
    log(f"\nSaved {out}")


if __name__ == "__main__":
    main()
