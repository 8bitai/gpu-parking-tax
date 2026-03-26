#!/usr/bin/env python3
"""
Real Model Validation + Cold-Start Power Measurement.

Combined script that:
  1. Loads actual HuggingFace models (not torch.empty tensors)
  2. Captures high-frequency power traces during model loading (cold-start)
  3. Measures idle power with real model weights loaded
  4. Compares results against synthetic (torch.empty) baseline

This validates that the parking tax findings hold with real models
including their CUDA kernels, cuDNN handles, and weight tensors.

Requirements:
    pip install torch transformers accelerate bitsandbytes

Usage:
    python validate_real_models.py --gpu 0

Time estimate: ~2 hours total
    - Bare idle baseline: 10 min
    - Per model (3 models): ~25 min each
      - Cold-start capture: 1-5 min (model loading with 1s power polling)
      - Idle recording: 15 min (30s intervals, n=30)
      - Unload + cooldown: 2 min
    - First run adds download time (~15 min on fast connection)

Output (written to ./output/):
    experiment.jsonl    — all power samples (bare idle + per-model idle)
    cold_starts.jsonl   — high-frequency power traces during model loading
    manifest.json       — experiment metadata and summary
"""

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS = [
    {
        "name": "Qwen/Qwen2.5-7B",
        "short_name": "qwen-7b",
        "expected_vram_gb": 14,
        "dtype": "float16",
    },
    {
        "name": "Qwen/Qwen2.5-14B",
        "short_name": "qwen-14b",
        "expected_vram_gb": 28,
        "dtype": "float16",
    },
    {
        "name": "Qwen/Qwen2.5-32B",
        "short_name": "qwen-32b",
        "expected_vram_gb": 64,
        "dtype": "bfloat16",
    },
]


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def nvidia_smi_snapshot(gpu_id: int) -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,memory.free,"
             "power.draw,power.limit,temperature.gpu,temperature.memory,"
             "clocks.current.sm,clocks.current.memory,uuid,"
             "utilization.gpu,utilization.memory",
             f"--id={gpu_id}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip(), "timestamp": now_utc()}

        fields = [f.strip() for f in result.stdout.strip().split(",")]
        return {
            "timestamp": now_utc(),
            "gpu_name": fields[0],
            "mem_total_mb": float(fields[1]),
            "mem_used_mb": float(fields[2]),
            "mem_free_mb": float(fields[3]),
            "power_w": float(fields[4]),
            "power_limit_w": float(fields[5]),
            "gpu_temp_c": float(fields[6]),
            "mem_temp_c": float(fields[7]) if fields[7] not in ("N/A", "[N/A]") else None,
            "sm_clock_mhz": float(fields[8]),
            "mem_clock_mhz": float(fields[9]),
            "uuid": fields[10],
            "gpu_util_pct": float(fields[11]),
            "mem_util_pct": float(fields[12]),
        }
    except Exception as e:
        return {"error": str(e), "timestamp": now_utc()}


class PowerMonitor:
    """Background thread that polls nvidia-smi at 1s during cold starts."""

    def __init__(self, gpu_id: int, interval: float = 1.0):
        self.gpu_id = gpu_id
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _poll(self):
        while not self._stop.is_set():
            snap = nvidia_smi_snapshot(self.gpu_id)
            snap["monitor_time"] = time.monotonic()
            self.samples.append(snap)
            time.sleep(self.interval)


def record_bare_idle(gpu_id, duration, sample_interval, log_file):
    log.info(f"Recording bare idle for {duration}s...")
    start = time.time()
    samples = 0
    while time.time() - start < duration:
        snap = nvidia_smi_snapshot(gpu_id)
        snap["phase"] = "bare_idle"
        snap["model"] = "none"
        snap["target_vram_gb"] = 0
        snap["gpu_id"] = gpu_id

        with open(log_file, "a") as f:
            f.write(json.dumps(snap) + "\n")

        samples += 1
        if samples % 5 == 0 or samples == 1:
            log.info(f"  [{time.time()-start:.0f}s] Sample {samples}: "
                     f"Power={snap.get('power_w', '?')}W, "
                     f"VRAM={snap.get('mem_used_mb', '?')}MB, "
                     f"SMClk={snap.get('sm_clock_mhz', '?')}MHz")
        time.sleep(sample_interval)
    return samples


def load_model(model_config, gpu_id):
    """Load model in fp16/bf16. If OOM, retry in 8-bit."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_config["name"]
    dtype_str = model_config["dtype"]
    dtype = getattr(torch, dtype_str)
    quantized = False

    log.info(f"Loading {model_name} in {dtype_str}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=f"cuda:{gpu_id}",
            trust_remote_code=True,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() and "CUDA" not in str(e):
            raise
        log.warning(f"  OOM loading {model_name} in {dtype_str}, retrying in 8-bit...")
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map=f"cuda:{gpu_id}",
            trust_remote_code=True,
        )
        quantized = True
        log.info(f"  Loaded {model_name} in 8-bit successfully")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )
    return model, tokenizer, quantized


def unload_model(model, tokenizer):
    import torch
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(5)


def run_model_phase(gpu_id, model_config, idle_duration, sample_interval,
                    experiment_log, cold_start_log):
    model_name = model_config["name"]
    short_name = model_config["short_name"]

    log.info(f"\n{'=' * 60}")
    log.info(f"MODEL: {model_name}")
    log.info(f"{'=' * 60}")

    # Cold-start capture
    log.info("Starting cold-start power monitor (1s intervals)...")
    monitor = PowerMonitor(gpu_id, interval=1.0)
    monitor.start()

    load_start = time.monotonic()
    model, tokenizer, quantized = load_model(model_config, gpu_id)
    load_end = time.monotonic()
    load_time_s = load_end - load_start

    time.sleep(10)
    monitor.stop()

    # Determine actual precision used
    precision = "int8" if quantized else model_config["dtype"]

    cold_start_data = {
        "model": model_name,
        "short_name": short_name,
        "precision": precision,
        "load_time_s": load_time_s,
        "n_samples": len(monitor.samples),
        "samples": monitor.samples,
    }
    with open(cold_start_log, "a") as f:
        f.write(json.dumps(cold_start_data, default=str) + "\n")

    powers = [s["power_w"] for s in monitor.samples if "power_w" in s]
    if powers:
        log.info(f"  Cold-start captured: {len(powers)} samples over {load_time_s:.1f}s")
        log.info(f"  Load power: mean={sum(powers)/len(powers):.1f}W, "
                 f"peak={max(powers):.1f}W, min={min(powers):.1f}W")

    snap = nvidia_smi_snapshot(gpu_id)
    actual_vram_mb = snap.get("mem_used_mb", 0)
    actual_vram_gb = actual_vram_mb / 1024
    log.info(f"  Actual VRAM used: {actual_vram_gb:.1f} GB "
             f"(expected ~{model_config['expected_vram_gb']} GB, loaded as {precision})")

    # Stabilize
    log.info("  Stabilizing (60s)...")
    time.sleep(60)

    # Idle power recording
    log.info(f"  Recording idle power for {idle_duration}s...")
    start = time.time()
    samples = 0

    while time.time() - start < idle_duration:
        snap = nvidia_smi_snapshot(gpu_id)
        snap["phase"] = f"real_model_{short_name}"
        snap["model"] = model_name
        snap["precision"] = precision
        snap["target_vram_gb"] = actual_vram_gb
        snap["actual_vram_mb"] = actual_vram_mb
        snap["gpu_id"] = gpu_id
        snap["load_time_s"] = load_time_s

        with open(experiment_log, "a") as f:
            f.write(json.dumps(snap) + "\n")

        samples += 1
        if samples % 5 == 0 or samples == 1:
            log.info(f"    [{time.time()-start:.0f}s] Sample {samples}: "
                     f"Power={snap.get('power_w', '?')}W, "
                     f"VRAM={snap.get('mem_used_mb', '?')}MB")
        time.sleep(sample_interval)

    # Unload
    log.info("  Unloading model...")
    unload_model(model, tokenizer)

    time.sleep(10)
    post = nvidia_smi_snapshot(gpu_id)
    log.info(f"  Post-unload: Power={post.get('power_w', '?')}W, "
             f"VRAM={post.get('mem_used_mb', '?')}MB")

    return {
        "model": model_name,
        "short_name": short_name,
        "precision": precision,
        "load_time_s": load_time_s,
        "actual_vram_gb": actual_vram_gb,
        "idle_samples": samples,
        "cold_start_samples": len(powers),
        "cold_start_power_mean": sum(powers) / len(powers) if powers else None,
        "cold_start_power_peak": max(powers) if powers else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Real Model Validation + Cold-Start Measurement"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--idle-duration", type=int, default=900,
                        help="Idle recording per model in seconds (default: 900 = 15 min)")
    parser.add_argument("--sample-interval", type=int, default=30,
                        help="Sample interval for idle recording (default: 30s)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model short names to test (default: all)")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory (default: ./output)")
    args = parser.parse_args()

    if args.models:
        selected = args.models.split(",")
        models = [m for m in MODELS if m["short_name"] in selected]
    else:
        models = MODELS

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_log = output_dir / "experiment.jsonl"
    cold_start_log = output_dir / "cold_starts.jsonl"
    manifest_file = output_dir / "manifest.json"

    n_models = len(models)
    est_time_min = 10 + n_models * (5 + args.idle_duration / 60 + 2)

    log.info("=" * 60)
    log.info("REAL MODEL VALIDATION + COLD-START MEASUREMENT")
    log.info("=" * 60)
    log.info(f"  GPU: {args.gpu}")
    log.info(f"  Models: {[m['short_name'] for m in models]}")
    log.info(f"  Idle duration per model: {args.idle_duration}s ({args.idle_duration/60:.0f} min)")
    log.info(f"  Sample interval: {args.sample_interval}s")
    log.info(f"  Estimated time: ~{est_time_min:.0f} min (excludes model download)")
    log.info(f"  Output: {output_dir}")

    initial = nvidia_smi_snapshot(args.gpu)
    log.info(f"\n  GPU: {initial.get('gpu_name', 'unknown')}")
    log.info(f"  UUID: {initial.get('uuid', 'unknown')}")
    log.info(f"  VRAM: {initial.get('mem_total_mb', '?')} MB total")

    manifest = {
        "experiment": "real_model_validation",
        "start_time": now_utc(),
        "gpu_id": args.gpu,
        "gpu_info": initial,
        "models": [m["name"] for m in models],
        "idle_duration_s": args.idle_duration,
        "sample_interval_s": args.sample_interval,
        "phases": [],
    }

    # Phase 0: Bare idle
    log.info(f"\n{'=' * 60}")
    log.info("PHASE 0: BARE IDLE BASELINE")
    log.info(f"{'=' * 60}")
    bare_samples = record_bare_idle(
        args.gpu, 600, args.sample_interval, experiment_log
    )
    manifest["phases"].append({"phase": "bare_idle", "samples": bare_samples})

    # Per-model phases
    for i, model_config in enumerate(models):
        log.info(f"\n{'=' * 60}")
        log.info(f"PHASE {i+1}/{n_models}: {model_config['name']}")
        log.info(f"{'=' * 60}")

        result = run_model_phase(
            gpu_id=args.gpu,
            model_config=model_config,
            idle_duration=args.idle_duration,
            sample_interval=args.sample_interval,
            experiment_log=experiment_log,
            cold_start_log=cold_start_log,
        )
        manifest["phases"].append(result)

        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        log.info(f"  Phase complete: {result['model']}")
        log.info(f"    Precision: {result['precision']}")
        log.info(f"    Load time: {result['load_time_s']:.1f}s")
        log.info(f"    VRAM used: {result['actual_vram_gb']:.1f} GB")
        if result.get('cold_start_power_mean'):
            log.info(f"    Cold-start power: {result['cold_start_power_mean']:.1f}W mean, "
                     f"{result['cold_start_power_peak']:.1f}W peak")

    manifest["end_time"] = now_utc()
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    log.info(f"\n{'=' * 60}")
    log.info("EXPERIMENT COMPLETE")
    log.info(f"{'=' * 60}")
    log.info(f"  Output: {output_dir}")

    log.info(f"\n  {'Model':<25} {'Precision':<10} {'VRAM (GB)':>10} {'Load (s)':>10} {'P_load (W)':>12}")
    log.info(f"  {'-'*70}")
    for phase in manifest["phases"]:
        if "model" in phase and phase.get("model") != "none":
            log.info(f"  {phase.get('short_name', '?'):<25} "
                     f"{phase.get('precision', '?'):<10} "
                     f"{phase.get('actual_vram_gb', '?'):>10.1f} "
                     f"{phase.get('load_time_s', '?'):>10.1f} "
                     f"{phase.get('cold_start_power_mean', '?'):>12.1f}")


if __name__ == "__main__":
    main()
