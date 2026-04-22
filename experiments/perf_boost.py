#!/usr/bin/env python3
"""
CUDA_DISABLE_PERF_BOOST Experiment

Tests whether NVIDIA's CUDA_DISABLE_PERF_BOOST environment variable
(shipped in driver 580.105.08, Nov 2025) reduces idle-with-context power
on datacenter GPUs under inference workloads.

Conditions (within-subject, same GPU, each in its own subprocess):
  a) Bare idle:  No CUDA context, no flag (baseline reference)
  b) Baseline:   CUDA context + torch.empty 16 GB, default behavior
  c) Flag on:    Same as (b) but with CUDA_DISABLE_PERF_BOOST=1
  d) vLLM base:  vLLM serving Qwen2.5-7B, idle between requests, default
  e) vLLM flag:  Same as (d) with CUDA_DISABLE_PERF_BOOST=1

Measurements per condition:
  nvidia-smi every 30s for 20 min (or 5 min in --quick mode), n=40.
  Records: SM clock, memory clock, power, temperature, perf state.

Prerequisites:
  - NVIDIA driver >= 580.105.08
  - pip install torch vllm transformers
  - A100 and H100 are both supported (same code path; 16GB static alloc fits either card).
  - On hosts with a small root disk (e.g. RunPod docker overlay), either export HF
    cache env vars or pass --hf-cache; by default the script uses
    /workspace/.cache/huggingface if /workspace exists.

Usage:
  # Run all conditions on GPU 0
  python perf_boost.py --gpu 0

  # Static tests only (skip vLLM)
  python perf_boost.py --gpu 0 --skip-vllm

  # Quick mode (5-min phases instead of 20-min)
  python perf_boost.py --gpu 0 --quick

  # Custom vLLM model
  python perf_boost.py --gpu 0 --model Qwen/Qwen2.5-7B

  # Pin Hugging Face / model cache to a large volume (e.g. RunPod /workspace)
  python perf_boost.py --gpu 0 --hf-cache /workspace/.cache/huggingface
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
VLLM_PORT = 8192
MIN_DRIVER_VERSION = (580, 105, 8)
# If /workspace exists (common on RunPod), use it so HF and vLLM do not fill small OS disks.
_HF_DEFAULT_AUTO = "/workspace/.cache/huggingface"


def setup_hf_cache(cache_root=None):
    """Set HF cache env for this process and all child subprocesses.

    Resolution order:
      1) *cache_root* (from --hf-cache) if set
      2) existing ``HF_HOME`` in the environment (return without overriding)
      3) * _HF_DEFAULT_AUTO* when ``/workspace`` exists (typical RunPod)
      4) else leave default HF cache locations unchanged (returns None)

    Set ``PERF_BOOST_NO_HF_CACHE_AUTO=1`` to disable step 3.

    Returns the effective cache root (string), or None if step 2 or 4 applies.
    """
    root = (cache_root or "").strip() or None
    if root is None and os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"].rstrip("/")
    if root is None and Path("/workspace").is_dir() and not os.environ.get(
        "PERF_BOOST_NO_HF_CACHE_AUTO"
    ):
        root = _HF_DEFAULT_AUTO
    if not root:
        return None
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    hub = p / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    tfc = p / "transformers"
    tfc.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(p)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["TRANSFORMERS_CACHE"] = str(tfc)
    return str(p)


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def query_nvidia_smi(gpu_id):
    """Query nvidia-smi for a single GPU snapshot."""
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,memory.free,"
             "power.draw,power.limit,temperature.gpu,temperature.memory,"
             "clocks.current.sm,clocks.current.memory,uuid,"
             "utilization.gpu,utilization.memory,pstate,"
             "driver_version",
             f"--id={gpu_id}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            log(f"  nvidia-smi error: {r.stderr.strip()}")
            return None
        f = [x.strip() for x in r.stdout.strip().split(",")]
        return {
            "timestamp": now_utc(),
            "gpu_name": f[0],
            "mem_total_mb": _float(f[1]),
            "mem_used_mb": _float(f[2]),
            "mem_free_mb": _float(f[3]),
            "power_w": _float(f[4]),
            "power_limit_w": _float(f[5]),
            "gpu_temp_c": _float(f[6]),
            "mem_temp_c": _float(f[7]) if f[7] not in ("N/A", "[N/A]") else None,
            "sm_clock_mhz": _float(f[8]),
            "mem_clock_mhz": _float(f[9]),
            "uuid": f[10],
            "gpu_util_pct": _float(f[11]),
            "mem_util_pct": _float(f[12]),
            "pstate": f[13],
            "driver_version": f[14],
        }
    except Exception as e:
        log(f"  nvidia-smi exception: {e}")
        return None


def parse_driver_version(version_str):
    """Parse driver version string like '580.105.08' into tuple."""
    parts = version_str.strip().split(".")
    return tuple(int(p) for p in parts)


def check_driver_version(gpu_id):
    """Verify driver >= 580.105.08. Returns (ok, version_str)."""
    snap = query_nvidia_smi(gpu_id)
    if not snap or "driver_version" not in snap:
        return False, "unknown"
    version_str = snap["driver_version"]
    try:
        version = parse_driver_version(version_str)
        return version >= MIN_DRIVER_VERSION, version_str
    except (ValueError, TypeError):
        return False, version_str


def write_record(fh, record):
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def record_phase(output_path, gpu_id, phase_name, condition, duration, interval,
                 extra_fields=None):
    """Record nvidia-smi samples for one phase. Appends to output_path.
    Returns list of samples."""
    start = time.time()
    samples = []
    n = 0
    with open(output_path, "a") as fh:
        while time.time() - start < duration:
            s = query_nvidia_smi(gpu_id)
            if s:
                s["phase"] = phase_name
                s["condition"] = condition
                s["gpu_id"] = gpu_id
                if extra_fields:
                    s.update(extra_fields)
                write_record(fh, s)
                samples.append(s)
                n += 1
                if n == 1 or n % 10 == 0:
                    log(f"  #{n}: Power={s['power_w']}W  "
                        f"SMClk={s['sm_clock_mhz']}MHz  "
                        f"PState={s['pstate']}  "
                        f"Temp={s['gpu_temp_c']}C")
            time.sleep(interval)
    return samples


def summarize_samples(samples, label):
    """Print summary statistics for a phase."""
    powers = [s["power_w"] for s in samples if s.get("power_w") is not None]
    clocks = [s["sm_clock_mhz"] for s in samples if s.get("sm_clock_mhz") is not None]
    if not powers:
        log(f"  {label}: no valid samples")
        return
    mean_p = sum(powers) / len(powers)
    std_p = (sum((p - mean_p) ** 2 for p in powers) / len(powers)) ** 0.5
    mean_c = sum(clocks) / len(clocks) if clocks else 0
    pstates = [s.get("pstate", "?") for s in samples]
    pstate_mode = max(set(pstates), key=pstates.count)
    log(f"  {label}: {mean_p:.1f} +/- {std_p:.1f} W  "
        f"(n={len(powers)}, SM={mean_c:.0f} MHz, PState={pstate_mode})")


def run_single_static_condition(gpu_id, env_flag, phase_duration, interval,
                                output_path):
    """Run one static condition in its own subprocess so the env var is
    read cleanly by the CUDA driver at init time."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if env_flag:
        env["CUDA_DISABLE_PERF_BOOST"] = "1"
    else:
        env.pop("CUDA_DISABLE_PERF_BOOST", None)

    condition = "flag_on" if env_flag else "baseline"

    # Worker: allocates 16GB, holds until killed.
    worker = """
import torch, time, signal, sys
torch.cuda.init()
x = torch.empty(int(16 * 1024**3 / 4), dtype=torch.float32, device='cuda:0')
torch.cuda.synchronize()
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(1)
"""

    log(f"\n  Starting worker: condition={condition} "
        f"(flag={'on' if env_flag else 'off'})")
    proc = subprocess.Popen(
        [sys.executable, "-c", worker],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    log("  Stabilizing (60s)...")
    time.sleep(60)

    # Verify worker is alive and GPU is allocated
    snap = query_nvidia_smi(gpu_id)
    if snap:
        log(f"  Pre-record state: Power={snap['power_w']}W  "
            f"SMClk={snap['sm_clock_mhz']}MHz  "
            f"VRAM={snap['mem_used_mb']}MB")
    if proc.poll() is not None:
        log(f"  ERROR: worker exited prematurely (rc={proc.returncode})")
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        log(f"  stderr: {stderr[:500]}")
        return []

    log(f"\n  Phase: cuda_ctx_16gb ({condition})")
    samples = record_phase(
        output_path, gpu_id, "cuda_ctx_16gb", condition,
        phase_duration, interval,
        extra_fields={"target_vram_gb": 16, "env_flag": env_flag},
    )
    summarize_samples(samples, f"CUDA ctx + 16GB ({condition})")

    log("  Stopping worker...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    log("  Cool-down (30s)...")
    time.sleep(30)
    return samples


def run_static_conditions(gpu_id, phase_duration, interval, output_path):
    """Bare idle, then baseline, then flag_on. Each in its own subprocess."""
    results = {}

    log(f"\n{'=' * 60}")
    log("BARE IDLE (no CUDA context, no flag)")
    log(f"{'=' * 60}")
    log("  Phase: bare_idle")
    bare = record_phase(
        output_path, gpu_id, "bare_idle", "bare",
        phase_duration, interval,
        extra_fields={"target_vram_gb": 0, "env_flag": False},
    )
    summarize_samples(bare, "Bare idle")
    results["bare"] = {"samples": bare}
    time.sleep(30)

    log(f"\n{'=' * 60}")
    log("CONDITION: BASELINE (default behavior, no flag)")
    log(f"{'=' * 60}")
    results["baseline"] = {
        "cuda_ctx_16gb": run_single_static_condition(
            gpu_id, env_flag=False,
            phase_duration=phase_duration, interval=interval,
            output_path=output_path,
        )
    }

    log(f"\n{'=' * 60}")
    log("CONDITION: FLAG_ON (CUDA_DISABLE_PERF_BOOST=1)")
    log(f"{'=' * 60}")
    results["flag_on"] = {
        "cuda_ctx_16gb": run_single_static_condition(
            gpu_id, env_flag=True,
            phase_duration=phase_duration, interval=interval,
            output_path=output_path,
        )
    }

    return results


def start_vllm_server(gpu_id, model, env_flag=False):
    """Start vLLM server as a subprocess. Returns Popen."""
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
    log(f"  Starting vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return proc


def wait_for_vllm(timeout=600):
    """Wait for vLLM /health to return 200."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(
                f"http://localhost:{VLLM_PORT}/health", method="GET",
            )
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            time.sleep(5)
    return False


def stop_vllm(proc):
    """Gracefully stop vLLM server."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def send_vllm_request(model, prompt="Hello, world!", max_tokens=64):
    """Send a single completion request to vLLM. Returns response JSON."""
    import urllib.request
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
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


def measure_vllm_latency(model, n_requests=50):
    """Measure latency distribution for vLLM requests."""
    latencies = []
    prompt = "Explain the concept of GPU power management in one paragraph."
    for i in range(n_requests):
        start = time.time()
        try:
            send_vllm_request(model, prompt=prompt, max_tokens=128)
            elapsed = time.time() - start
            latencies.append(elapsed)
            if (i + 1) % 10 == 0:
                log(f"    Request {i+1}/{n_requests}: {elapsed:.2f}s")
        except Exception as e:
            log(f"    Request {i+1} failed: {e}")
    if not latencies:
        return {"n": 0, "error": "no successful requests"}
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def pct(p):
        idx = min(int(n * p / 100), n - 1)
        return latencies_sorted[idx]

    return {
        "n": n,
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "mean": sum(latencies_sorted) / n,
        "min": latencies_sorted[0],
        "max": latencies_sorted[-1],
        "all": latencies,
    }


def measure_cold_start(model, n_warm=5):
    """Measure the latency of one request after idle, then a few more
    to observe DVFS-driven recovery."""
    log("  Cold-start test: first request after idle...")
    t0 = time.time()
    try:
        send_vllm_request(model, prompt="Hello", max_tokens=32)
        cold = time.time() - t0
    except Exception as e:
        log(f"    Cold request failed: {e}")
        cold = None
    log(f"    Cold: {cold:.3f}s" if cold else "    Cold: FAILED")

    warm = []
    for i in range(n_warm):
        t0 = time.time()
        try:
            send_vllm_request(model, prompt="Hello", max_tokens=32)
            warm.append(time.time() - t0)
        except Exception as e:
            log(f"    Warm request {i+1} failed: {e}")
    log(f"    Subsequent: {[f'{l:.3f}s' for l in warm]}")

    return {"cold_s": cold, "subsequent_s": warm}


def run_vllm_conditions(gpu_id, model, phase_duration, interval, output_path):
    """Run vLLM conditions: baseline vs flag_on. Each in a separate process."""
    results = {}

    for condition, env_flag in [("vllm_baseline", False), ("vllm_flag_on", True)]:
        log(f"\n{'=' * 60}")
        log(f"CONDITION: {condition.upper()}"
            f"{' (CUDA_DISABLE_PERF_BOOST=1)' if env_flag else ' (default)'}")
        log(f"{'=' * 60}")

        proc = start_vllm_server(gpu_id, model, env_flag=env_flag)
        log("  Waiting for vLLM to be ready...")

        if not wait_for_vllm():
            log("  ERROR: vLLM did not start within timeout")
            stop_vllm(proc)
            continue
        log("  vLLM ready")

        # Warm up
        log("  Warming up (3 requests)...")
        for _ in range(3):
            try:
                send_vllm_request(model)
            except Exception as e:
                log(f"    Warmup request failed: {e}")

        # Steady-state latency distribution
        log("  Measuring steady-state latency (50 requests)...")
        latency_data = measure_vllm_latency(model, n_requests=50)
        if latency_data.get("n", 0) > 0:
            log(f"  Latency p50={latency_data['p50']:.2f}s  "
                f"p95={latency_data['p95']:.2f}s  "
                f"p99={latency_data['p99']:.2f}s  "
                f"max={latency_data['max']:.2f}s")

        with open(output_path, "a") as fh:
            write_record(fh, {
                "timestamp": now_utc(),
                "phase": "vllm_latency_warm",
                "condition": condition,
                "model": model,
                "env_flag": env_flag,
                "latency": latency_data,
            })

        # Let it idle, measure power
        log("  Stabilizing (60s idle)...")
        time.sleep(60)

        log(f"  Recording vLLM idle power for {phase_duration}s...")
        idle_samples = record_phase(
            output_path, gpu_id, "vllm_idle", condition,
            phase_duration, interval,
            extra_fields={
                "model": model,
                "env_flag": env_flag,
                "vllm_port": VLLM_PORT,
            },
        )
        summarize_samples(idle_samples, f"vLLM idle ({condition})")

        # Cold-start latency: first request after the long idle
        cold_data = measure_cold_start(model, n_warm=5)
        with open(output_path, "a") as fh:
            write_record(fh, {
                "timestamp": now_utc(),
                "phase": "vllm_cold_start",
                "condition": condition,
                "model": model,
                "env_flag": env_flag,
                **cold_data,
            })

        log("  Stopping vLLM...")
        stop_vllm(proc)

        log("  Cool-down (60s)...")
        time.sleep(60)

        results[condition] = {
            "idle_samples": idle_samples,
            "latency": latency_data,
            "cold_start": cold_data,
        }

    return results


def run_experiment(args):
    ok, version = check_driver_version(args.gpu)
    if not ok:
        log(f"ERROR: Driver version {version} < "
            f"{'.'.join(str(v) for v in MIN_DRIVER_VERSION)}")
        log("CUDA_DISABLE_PERF_BOOST requires driver >= 580.105.08")
        sys.exit(1)
    log(f"Driver version: {version} (>= 580.105.08, OK)")

    snap = query_nvidia_smi(args.gpu)
    if not snap:
        log("ERROR: Cannot query GPU")
        sys.exit(1)
    gpu_name = snap["gpu_name"]
    log(f"GPU: {gpu_name}")
    log(f"UUID: {snap['uuid']}")

    phase_duration = 300 if args.quick else 1200
    interval = args.interval

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"perf_boost_{ts}.jsonl"
    manifest_path = output_dir / f"perf_boost_{ts}_manifest.json"

    log(f"Output: {output_path}")
    log(f"Phase duration: {phase_duration}s")
    log(f"Sample interval: {interval}s")
    log(f"Quick mode: {args.quick}")
    log(f"Skip vLLM: {args.skip_vllm}")
    hfc = getattr(args, "hf_cache_resolved", None)
    if hfc:
        log(f"Hugging Face cache: {hfc}")

    manifest = {
        "experiment": "cuda_disable_perf_boost",
        "start_time": now_utc(),
        "gpu_id": args.gpu,
        "gpu_name": gpu_name,
        "gpu_uuid": snap["uuid"],
        "driver_version": version,
        "huggingface_cache_root": getattr(args, "hf_cache_resolved", None),
        "phase_duration_s": phase_duration,
        "sample_interval_s": interval,
        "quick_mode": args.quick,
        "conditions": [],
    }

    # Create the output file
    output_path.touch()

    log("\n" + "=" * 60)
    log("PART 1: Static conditions (torch.empty 16 GB)")
    log("=" * 60)
    static_results = run_static_conditions(
        args.gpu, phase_duration, interval, output_path,
    )
    manifest["conditions"].extend(["bare", "baseline", "flag_on"])

    vllm_results = {}
    if not args.skip_vllm:
        log("\n" + "=" * 60)
        log("PART 2: vLLM conditions")
        log("=" * 60)
        vllm_results = run_vllm_conditions(
            args.gpu, args.model, phase_duration, interval, output_path,
        )
        manifest["conditions"].extend(["vllm_baseline", "vllm_flag_on"])
        manifest["vllm_model"] = args.model

    manifest["end_time"] = now_utc()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"\n{'=' * 60}")
    log("EXPERIMENT COMPLETE")
    log(f"{'=' * 60}")
    log(f"Output: {output_path}")
    log(f"Manifest: {manifest_path}")

    # Summary deltas
    def _mean_power(samples):
        powers = [s["power_w"] for s in samples if s.get("power_w") is not None]
        return sum(powers) / len(powers) if powers else None

    bare_mean = _mean_power(static_results.get("bare", {}).get("samples", []))
    base_mean = _mean_power(static_results.get("baseline", {}).get("cuda_ctx_16gb", []))
    flag_mean = _mean_power(static_results.get("flag_on", {}).get("cuda_ctx_16gb", []))

    log("\n--- STATIC RESULTS ---")
    if bare_mean is not None:
        log(f"  Bare idle:        {bare_mean:.1f} W")
    if base_mean is not None:
        log(f"  CUDA ctx + 16GB:  {base_mean:.1f} W")
    if flag_mean is not None:
        log(f"  Flag on + 16GB:   {flag_mean:.1f} W")
    if base_mean is not None and flag_mean is not None:
        log(f"  Flag reduction:   {base_mean - flag_mean:+.1f} W "
            f"({100 * (base_mean - flag_mean) / base_mean:.1f}%)")
    if bare_mean is not None and flag_mean is not None:
        log(f"  Flag vs bare:     {flag_mean - bare_mean:+.1f} W")

    if not args.skip_vllm:
        vbase_mean = _mean_power(
            vllm_results.get("vllm_baseline", {}).get("idle_samples", []))
        vflag_mean = _mean_power(
            vllm_results.get("vllm_flag_on", {}).get("idle_samples", []))
        log("\n--- VLLM RESULTS ---")
        if vbase_mean is not None:
            log(f"  vLLM baseline idle: {vbase_mean:.1f} W")
        if vflag_mean is not None:
            log(f"  vLLM flag idle:     {vflag_mean:.1f} W")
        if vbase_mean is not None and vflag_mean is not None:
            log(f"  vLLM reduction:     {vbase_mean - vflag_mean:+.1f} W "
                f"({100 * (vbase_mean - vflag_mean) / vbase_mean:.1f}%)")

        for c in ("vllm_baseline", "vllm_flag_on"):
            lat = vllm_results.get(c, {}).get("latency", {})
            cold = vllm_results.get(c, {}).get("cold_start", {})
            if lat.get("n", 0) > 0:
                log(f"  {c} latency:  p50={lat['p50']:.2f}s  "
                    f"p95={lat['p95']:.2f}s  "
                    f"p99={lat['p99']:.2f}s")
            if cold.get("cold_s") is not None:
                log(f"  {c} cold-start: {cold['cold_s']:.2f}s  "
                    f"subsequent: {[f'{l:.2f}s' for l in cold.get('subsequent_s', [])]}")


def main():
    parser = argparse.ArgumentParser(
        description="CUDA_DISABLE_PERF_BOOST experiment for datacenter GPUs")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (default: 0)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 5-min phases instead of 20-min")
    parser.add_argument("--skip-vllm", action="store_true",
                        help="Skip vLLM conditions")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"vLLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--interval", type=int, default=30,
                        help="Sample interval in seconds (default: 30)")
    parser.add_argument("--output-dir", type=str,
                        default="data/experiments/perf_boost",
                        help="Output directory")
    parser.add_argument(
        "--hf-cache",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Hugging Face root cache (sets HF_HOME, HUGGINGFACE_HUB_CACHE, "
            "TRANSFORMERS_CACHE). "
            f"Default: use {_HF_DEFAULT_AUTO} if /workspace exists, else use "
            "existing HF_HOME or system default"
        ),
    )
    args = parser.parse_args()
    args.hf_cache_resolved = setup_hf_cache(args.hf_cache)
    run_experiment(args)


if __name__ == "__main__":
    main()