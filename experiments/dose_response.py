#!/usr/bin/env python3
"""
VRAM Dose-Response Experiment

Measures idle GPU power as a function of VRAM allocation to quantify the
"model parking tax." Creates a persistent CUDA context, then sequentially
allocates increasing VRAM levels using torch.empty tensors. At each level:
stabilize 60s, record nvidia-smi every 30s for the phase duration, release,
cool down 30s.

Auto-detects GPU architecture and selects appropriate VRAM levels, or
accepts custom levels via --levels.

Supported GPUs:
  H100 80GB (HBM3)  — levels up to 64 GB
  A100 80GB (HBM2e) — levels up to 72 GB
  L40S 48GB (GDDR6) — levels up to 40 GB

Usage:
  # Auto-detect GPU, full experiment (~3.5 hours)
  python dose_response.py --gpu 0

  # Quick test (~45 min, fewer VRAM levels)
  python dose_response.py --gpu 0 --quick

  # Custom levels and duration
  python dose_response.py --gpu 0 --levels 0,8,32,64 --duration 600
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# Per-architecture VRAM levels (GB)
GPU_CONFIGS = {
    "H100": {
        "full": [0, 1, 4, 8, 16, 32, 48, 64],
        "quick": [0, 4, 16, 48, 64],
    },
    "A100": {
        "full": [0, 1, 4, 8, 16, 32, 48, 64, 72],
        "quick": [0, 4, 16, 48, 64],
    },
    "L40S": {
        "full": [0, 1, 4, 8, 16, 24, 32, 40],
        "quick": [0, 4, 16, 32, 40],
    },
}

# Fallback for unrecognized GPUs: conservative levels
DEFAULT_CONFIG = {
    "full": [0, 1, 4, 8, 16, 32, 48],
    "quick": [0, 4, 16, 32],
}


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
             "utilization.gpu,utilization.memory",
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
            "mem_temp_c": _float(f[7]),
            "sm_clock_mhz": _float(f[8]),
            "mem_clock_mhz": _float(f[9]),
            "uuid": f[10],
            "gpu_util_pct": _float(f[11]),
            "mem_util_pct": _float(f[12]),
        }
    except Exception as e:
        log(f"  nvidia-smi exception: {e}")
        return None


def detect_gpu_type(gpu_id):
    """Auto-detect GPU architecture from nvidia-smi."""
    snap = query_nvidia_smi(gpu_id)
    if not snap:
        return None, None
    name = snap["gpu_name"]
    for key in GPU_CONFIGS:
        if key in name.upper():
            return key, name
    return None, name


def write_record(fh, record):
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def record_phase(fh, gpu_id, phase_name, target_vram_gb, duration, interval):
    """Record nvidia-smi samples for one phase. Returns sample count."""
    start = time.time()
    n = 0
    while time.time() - start < duration:
        s = query_nvidia_smi(gpu_id)
        if s:
            s["phase"] = phase_name
            s["target_vram_gb"] = target_vram_gb
            s["gpu_id"] = gpu_id
            write_record(fh, s)
            n += 1
            if n == 1 or n % 10 == 0:
                log(f"  #{n}: Power={s['power_w']}W  "
                    f"SMClk={s['sm_clock_mhz']}MHz  "
                    f"VRAM={s['mem_used_mb']}MB  "
                    f"Temp={s['gpu_temp_c']}C")
        time.sleep(interval)
    return n


def run_experiment(gpu_id, vram_levels, phase_duration, sample_interval, output_path):
    import torch

    device = torch.device(f"cuda:{gpu_id}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    phase_summary = []

    with open(output_path, "w") as fh:

        # ── Bare idle (no CUDA context) ──
        log(f"\n{'='*60}")
        log("BARE IDLE (no CUDA context)")
        log(f"{'='*60}")
        n = record_phase(fh, gpu_id, "bare_idle", 0, phase_duration, sample_interval)
        phase_summary.append(("bare_idle", 0, n))
        log(f"  Done: {n} samples")

        # ── Create persistent CUDA context ──
        log("\nCreating persistent CUDA context...")
        persistent_ctx = torch.zeros(1024, device=device)
        torch.cuda.synchronize()
        snap = query_nvidia_smi(gpu_id)
        if snap:
            log(f"  Context active: Power={snap['power_w']}W  "
                f"SMClk={snap['sm_clock_mhz']}MHz  "
                f"VRAM={snap['mem_used_mb']}MB")

        # ── VRAM dose-response phases ──
        for i, vram_gb in enumerate(vram_levels):
            phase_name = f"vram_{vram_gb:.0f}gb"

            log(f"\n{'='*60}")
            log(f"PHASE {i+1}/{len(vram_levels)}: {vram_gb:.0f} GB VRAM")
            log(f"{'='*60}")

            tensor = None
            if vram_gb > 0:
                try:
                    n_elements = int(vram_gb * 1024**3 / 4)
                    log(f"  Allocating {vram_gb:.0f} GB...")
                    tensor = torch.empty(n_elements, dtype=torch.float32, device=device)
                    actual_gb = tensor.nelement() * tensor.element_size() / 1024**3
                    log(f"  Allocated {actual_gb:.2f} GB")
                except torch.cuda.OutOfMemoryError:
                    log(f"  OOM at {vram_gb}GB — skipping")
                    phase_summary.append((phase_name, vram_gb, 0))
                    continue
            else:
                log("  CUDA context only (0 GB extra)")

            log("  Stabilizing (60s)...")
            time.sleep(60)

            log(f"  Recording for {phase_duration}s...")
            n = record_phase(fh, gpu_id, phase_name, vram_gb, phase_duration, sample_interval)
            phase_summary.append((phase_name, vram_gb, n))
            log(f"  Done: {n} samples")

            if tensor is not None:
                del tensor
                torch.cuda.empty_cache()

            log("  Cool-down (30s)...")
            time.sleep(30)

        # Keep persistent context reference alive until here
        del persistent_ctx

    # ── Summary ──
    log(f"\n{'='*60}")
    log("EXPERIMENT COMPLETE")
    log(f"{'='*60}")
    log(f"  Output: {output_path}")
    total = sum(s for _, _, s in phase_summary)
    log(f"  Total samples: {total}")
    log(f"\n  {'Phase':<25} {'VRAM':>8} {'Samples':>8}")
    log(f"  {'─'*45}")
    for name, vram, samples in phase_summary:
        log(f"  {name:<25} {vram:>7.0f}G {samples:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="VRAM dose-response experiment for measuring the model parking tax")
    parser.add_argument("--gpu", type=int, required=True,
                        help="GPU index (e.g., 0)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with fewer VRAM levels and 5-min phases")
    parser.add_argument("--duration", type=int, default=None,
                        help="Phase duration in seconds (default: 1200 full, 300 quick)")
    parser.add_argument("--levels", type=str, default=None,
                        help="Comma-separated VRAM levels in GB (e.g., 0,8,32,64)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Sample interval in seconds (default: 30)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: auto-generated)")
    args = parser.parse_args()

    # Detect GPU type
    gpu_type, gpu_name = detect_gpu_type(args.gpu)
    if gpu_type:
        log(f"Detected {gpu_name} → using {gpu_type} config")
        config = GPU_CONFIGS[gpu_type]
    else:
        log(f"GPU: {gpu_name or 'unknown'} — using default VRAM levels")
        config = DEFAULT_CONFIG
        gpu_type = "gpu"

    # Resolve parameters
    if args.levels:
        vram_levels = [float(v) for v in args.levels.split(",")]
    elif args.quick:
        vram_levels = config["quick"]
    else:
        vram_levels = config["full"]

    if args.duration:
        phase_duration = args.duration
    elif args.quick:
        phase_duration = 300
    else:
        phase_duration = 1200

    if args.output:
        output_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/experiments/{gpu_type.lower()}_dose_response_{ts}.jsonl"

    log(f"VRAM levels: {vram_levels}")
    log(f"Phase duration: {phase_duration}s")
    log(f"Output: {output_path}")

    run_experiment(args.gpu, vram_levels, phase_duration, args.interval, output_path)


if __name__ == "__main__":
    main()
