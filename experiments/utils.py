"""Shared utilities for GPU parking-tax experiments.

Extracted from experiments/perf_boost.py (which remains unchanged for Paper 1
reproducibility).  New experiment scripts should import from here rather than
duplicating these helpers.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-7B"
MIN_DRIVER_VERSION = (580, 105, 8)
_HF_DEFAULT_AUTO = "/workspace/.cache/huggingface"


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# HF cache
# ---------------------------------------------------------------------------

def setup_hf_cache(cache_root=None):
    """Set HF cache env for this process and all child subprocesses.

    Resolution order:
      1) *cache_root* (from --hf-cache) if set
      2) existing ``HF_HOME`` in the environment (return without overriding)
      3) ``_HF_DEFAULT_AUTO`` when ``/workspace`` exists (typical RunPod)
      4) else leave default HF cache locations unchanged (returns None)
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
    for sub in ("hub", "transformers"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(p)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(p / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(p / "transformers")
    return str(p)


# ---------------------------------------------------------------------------
# nvidia-smi
# ---------------------------------------------------------------------------

def query_nvidia_smi(gpu_id, timeout=30):
    """Query nvidia-smi for a single GPU snapshot.

    Default timeout is 30 s (raised from the 10 s used in the single-GPU
    script) because nvidia-smi can stall under active NCCL workloads.
    """
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,"
                "power.draw,power.limit,temperature.gpu,temperature.memory,"
                "clocks.current.sm,clocks.current.memory,uuid,"
                "utilization.gpu,utilization.memory,pstate,"
                "driver_version",
                f"--id={gpu_id}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            log(f"  nvidia-smi error (GPU {gpu_id}): {r.stderr.strip()}")
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
    except subprocess.TimeoutExpired:
        log(f"  nvidia-smi timeout (GPU {gpu_id}, {timeout}s)")
        return None
    except Exception as e:
        log(f"  nvidia-smi exception (GPU {gpu_id}): {e}")
        return None


def parse_driver_version(version_str):
    parts = version_str.strip().split(".")
    return tuple(int(p) for p in parts)


def check_driver_version(gpu_id=0):
    """Verify driver >= 580.105.08.  Returns (ok, version_str)."""
    snap = query_nvidia_smi(gpu_id)
    if not snap or "driver_version" not in snap:
        return False, "unknown"
    version_str = snap["driver_version"]
    try:
        version = parse_driver_version(version_str)
        return version >= MIN_DRIVER_VERSION, version_str
    except (ValueError, TypeError):
        return False, version_str


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def write_record(fh, record):
    fh.write(json.dumps(record) + "\n")
    fh.flush()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def summarize_samples(samples, label):
    """Print summary statistics for a list of nvidia-smi samples."""
    powers = [s["power_w"] for s in samples if s.get("power_w") is not None]
    clocks = [s["sm_clock_mhz"] for s in samples
              if s.get("sm_clock_mhz") is not None]
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


def mean_power(samples):
    powers = [s["power_w"] for s in samples if s.get("power_w") is not None]
    return sum(powers) / len(powers) if powers else None


def std_power(samples):
    powers = [s["power_w"] for s in samples if s.get("power_w") is not None]
    if len(powers) < 2:
        return 0.0
    m = sum(powers) / len(powers)
    return (sum((p - m) ** 2 for p in powers) / len(powers)) ** 0.5


def percentile(data, p):
    s = sorted(data)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


# ---------------------------------------------------------------------------
# vLLM lifecycle
# ---------------------------------------------------------------------------

def start_vllm_server(gpu_ids, model, port, env_flag=False, tp_size=1):
    """Start a vLLM OpenAI-compatible server as a subprocess.

    Physical-to-logical GPU mapping:
      CUDA_VISIBLE_DEVICES is set to the comma-joined *physical* GPU indices
      (e.g. "0,1").  Inside the vLLM process these appear as *logical*
      GPUs 0..N-1.  nvidia-smi always uses physical indices.

    Returns the Popen handle.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    if env_flag:
        env["CUDA_DISABLE_PERF_BOOST"] = "1"
    else:
        env.pop("CUDA_DISABLE_PERF_BOOST", None)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--dtype", "float16",
        "--port", str(port),
        "--max-model-len", "4096",
        "--no-enable-log-requests",
    ]
    if tp_size > 1:
        cmd.extend(["--tensor-parallel-size", str(tp_size)])

    log(f"  Starting vLLM: tp={tp_size}, GPUs={gpu_ids}, port={port}, "
        f"flag={'ON' if env_flag else 'OFF'}")
    # Do not use PIPE: vLLM logs can fill the buffer and block the child.
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc


def wait_for_vllm(port, timeout=600):
    """Wait for vLLM /health endpoint to return 200."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(
                f"http://localhost:{port}/health", method="GET",
            )
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            time.sleep(5)
    return False


def stop_vllm(proc, label="vLLM"):
    """Gracefully stop a vLLM process.

    Sends SIGTERM first (required for clean NCCL shared-memory cleanup under
    TP).  Only escalates to SIGKILL after 60 s.  SIGKILL leaks shared memory
    and can pollute subsequent conditions.
    """
    if proc.poll() is not None:
        return
    log(f"  Sending SIGTERM to {label} (pid={proc.pid})...")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
        log(f"  {label} exited cleanly.")
    except subprocess.TimeoutExpired:
        log(f"  {label} did not exit in 60s, sending SIGKILL...")
        proc.kill()
        proc.wait(timeout=10)


def send_vllm_request(model, port, prompt=None, max_tokens=128):
    """Send a single completion request to a vLLM server."""
    import urllib.request
    if prompt is None:
        prompt = "Explain the concept of GPU power management in one paragraph."
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def measure_warm_latency(model, port, n_requests=50, max_tokens=128):
    """Measure back-to-back inference latency.

    Sends *n_requests* sequentially with no idle gap.
    Returns a stats dict (mean, p50, p95, p99, stdev, etc.).
    """
    latencies = []
    prompt = "Explain the concept of GPU power management in one paragraph."
    for i in range(n_requests):
        t0 = time.time()
        try:
            send_vllm_request(model, port, prompt=prompt, max_tokens=max_tokens)
            elapsed = time.time() - t0
            latencies.append(elapsed)
            if (i + 1) % 10 == 0:
                log(f"    Request {i+1}/{n_requests}: {elapsed*1000:.1f}ms")
        except Exception as e:
            log(f"    Request {i+1} FAILED: {e}")
    if not latencies:
        return {"n": 0, "error": "no successful requests"}
    n = len(latencies)
    mean = sum(latencies) / n
    stdev = (sum((l - mean) ** 2 for l in latencies) / n) ** 0.5 if n > 1 else 0
    return {
        "n": n,
        "mean": mean,
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "stdev": stdev,
        "min": min(latencies),
        "max": max(latencies),
        "all_ms": [l * 1000 for l in latencies],
    }
