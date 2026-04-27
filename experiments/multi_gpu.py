#!/usr/bin/env python3
"""
Multi-GPU CUDA_DISABLE_PERF_BOOST Experiment

Tests whether the CUDA_DISABLE_PERF_BOOST parking tax multiplies under
tensor parallelism (TP) and data parallelism (DP) configurations on a
4xH100 SXM node.

Conditions (within-subject, same node, each in its own subprocess):
  1) Bare idle:      All 4 GPUs, no CUDA context (baseline reference)
  2) TP=2 baseline:  vLLM TP=2 on GPUs 0,1 — flag off
  3) TP=2 flag on:   vLLM TP=2 on GPUs 0,1 — CUDA_DISABLE_PERF_BOOST=1
  4) TP=4 baseline:  vLLM TP=4 on GPUs 0-3 — flag off
  5) TP=4 flag on:   vLLM TP=4 on GPUs 0-3 — CUDA_DISABLE_PERF_BOOST=1
  6) DP=2 baseline:  2x independent vLLM on GPU 0 and GPU 1 — flag off
  7) DP=2 flag on:   2x independent vLLM on GPU 0 and GPU 1 — flag on
  8) TP2xDP2 base:   2x TP=2 groups (GPUs 0,1 + GPUs 2,3) — flag off
  9) TP2xDP2 flag:   2x TP=2 groups (GPUs 0,1 + GPUs 2,3) — flag on

Measurements per vLLM condition:
  - Steady-state idle power per GPU (n=20 quick / n=40 full, 30s interval)
  - Warm inference latency (n=50 back-to-back, 128-token completion)
  - DVFS decay curve (5 min at 1s interval after warm burst — captures
    transition from active to idle, essential for dynamic flag-toggle timing)
  - Cold-start ramp (60s idle soak -> 1 cold + 4 recovery requests)

Prerequisites:
  - NVIDIA driver >= 580.105.08
  - 4x H100 SXM with NVLink + NCCL
  - pip install vllm transformers
  - Qwen/Qwen2.5-7B available (downloaded on first run)

Usage:
  # Smoke test (~15 min, ~$3)
  python multi_gpu.py --smoke-test

  # Quick mode (10-min idle phases, ~3.5 hours, ~$35)
  python multi_gpu.py --quick

  # Full mode (20-min idle phases, ~6.5 hours, ~$75)
  python multi_gpu.py

  # Custom model / cache location
  python multi_gpu.py --model Qwen/Qwen2.5-7B --hf-cache /workspace/.cache/huggingface
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from utils import (
    now_utc, log, setup_hf_cache,
    query_nvidia_smi, check_driver_version,
    write_record, summarize_samples, mean_power, std_power, percentile,
    start_vllm_server, wait_for_vllm, stop_vllm, send_vllm_request,
    measure_warm_latency,
    DEFAULT_MODEL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_GPU_IDS = [0, 1, 2, 3]
TP2_GPU_IDS = [0, 1]
TP4_GPU_IDS = [0, 1, 2, 3]

# DP=2 layout: two independent replicas, one GPU each.
#   Replica 0  ->  physical GPU 0  ->  CUDA_VISIBLE_DEVICES=0  ->  port 8192
#   Replica 1  ->  physical GPU 1  ->  CUDA_VISIBLE_DEVICES=1  ->  port 8193
# nvidia-smi always uses physical indices; inside each vLLM process the
# single GPU appears as logical GPU 0.
DP_REPLICA_GPUS = {0: [0], 1: [1]}
DP_PORTS = {0: 8192, 1: 8193}
# Mixed TP+DP layout: two TP=2 groups, each acting as a DP replica.
#   Replica 0: physical GPUs 0,1  ->  CUDA_VISIBLE_DEVICES=0,1  ->  port 8192
#   Replica 1: physical GPUs 2,3  ->  CUDA_VISIBLE_DEVICES=2,3  ->  port 8193
# Inside each vLLM process the 2 GPUs appear as logical 0,1.
# nvidia-smi always uses physical indices.
TPDP_REPLICA_GPUS = {0: [0, 1], 1: [2, 3]}
TPDP_PORTS = {0: 8192, 1: 8193}

DEFAULT_PORT = 8192

# NCCL stabilization wait (seconds) after vLLM reports ready.
# NCCL init spikes all GPUs to ~400W+; we wait for clocks to settle.
NCCL_STABILIZE_S = {2: 120, 4: 180}

# vLLM startup timeout (seconds).
# TP=4 needs up to 10 min for sharding + NCCL init + CUDA graph capture.
VLLM_TIMEOUT_S = {1: 600, 2: 600, 4: 900}

COOLDOWN_S = 120          # inter-condition cooldown
MAX_TEMP_C = 60           # abort threshold
COLD_SOAK_S = 60          # idle soak before cold burst
N_COLD = 1                # cold requests
N_RECOVERY = 4            # recovery requests after cold
N_WARM = 50               # warm latency requests
SAMPLE_INTERVAL = 30      # seconds between nvidia-smi samples

# Decay curve: captures DVFS transition from active to idle.
# Essential for determining when to engage the flag in a dynamic toggle scheme.
DECAY_DURATION_S = 300    # 5 minutes
DECAY_INTERVAL_S = 1      # 1-second sampling

# Single-GPU reference values from Paper 1 (H100 SXM)
SINGLE_H100_PARKING_TAX_W = 49.9
SINGLE_H100_COLD_PENALTY_MS = 150


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def count_gpus():
    """Count GPUs visible to nvidia-smi."""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return 0
        return len([l for l in r.stdout.strip().split("\n") if l.strip()])
    except Exception:
        return 0


def check_gpu_temperatures(gpu_ids, max_temp=MAX_TEMP_C):
    """Check all GPU temperatures. Returns (all_ok, {gpu_id: temp_c})."""
    temps = {}
    all_ok = True
    for gid in gpu_ids:
        snap = query_nvidia_smi(gid)
        if snap and snap.get("gpu_temp_c") is not None:
            temps[gid] = snap["gpu_temp_c"]
            if snap["gpu_temp_c"] > max_temp:
                all_ok = False
        else:
            temps[gid] = None
    return all_ok, temps


# ---------------------------------------------------------------------------
# Multi-GPU recording
# ---------------------------------------------------------------------------

def record_phase_multi(output_path, gpu_ids, phase_name, condition, duration,
                       interval, extra_fields=None, gpu_extra_fields=None):
    """Record nvidia-smi samples for multiple physical GPUs over a phase.

    Writes one JSONL record per GPU per sample interval.
    *gpu_extra_fields* maps gpu_id -> dict of additional fields for that GPU
    (e.g. {0: {"replica_id": 0}, 1: {"replica_id": 1}} for DP).

    Returns {gpu_id: [samples]} dict.
    """
    start = time.time()
    all_samples = {gid: [] for gid in gpu_ids}
    n = 0
    with open(output_path, "a") as fh:
        while time.time() - start < duration:
            n += 1
            for gid in gpu_ids:
                s = query_nvidia_smi(gid)
                if s is None:
                    log(f"  WARNING: nvidia-smi failed for GPU {gid}, "
                        f"skipping sample #{n}")
                    continue
                s["phase"] = phase_name
                s["condition"] = condition
                s["gpu_id"] = gid  # always physical index
                if extra_fields:
                    s.update(extra_fields)
                if gpu_extra_fields and gid in gpu_extra_fields:
                    s.update(gpu_extra_fields[gid])
                write_record(fh, s)
                all_samples[gid].append(s)
            # Log first sample and every 10th
            if n == 1 or n % 10 == 0:
                parts = []
                for gid in gpu_ids:
                    if all_samples[gid]:
                        last = all_samples[gid][-1]
                        parts.append(
                            f"GPU{gid}={last.get('power_w', '?')}W/"
                            f"{last.get('sm_clock_mhz', '?')}MHz/"
                            f"{last.get('pstate', '?')}")
                log(f"  #{n}: {', '.join(parts)}")
            time.sleep(interval)
    return all_samples


# ---------------------------------------------------------------------------
# Inter-condition cooldown
# ---------------------------------------------------------------------------

def inter_condition_cooldown(gpu_ids, duration=COOLDOWN_S, max_temp=MAX_TEMP_C):
    """Wait between conditions, then verify temperatures are safe."""
    log(f"\n  Inter-condition cooldown ({duration}s)...")
    time.sleep(duration)
    ok, temps = check_gpu_temperatures(gpu_ids, max_temp)
    temp_strs = [f"GPU{gid}={t:.0f}C" if t is not None else f"GPU{gid}=N/A"
                 for gid, t in sorted(temps.items())]
    log(f"  Temperature check: {', '.join(temp_strs)}")
    if not ok:
        log(f"  ABORT: GPU temperature exceeds {max_temp}C "
            f"-- incomplete cleanup suspected")
        sys.exit(1)
    log("  Temperatures OK, proceeding.")


# ---------------------------------------------------------------------------
# Cold-start measurement
# ---------------------------------------------------------------------------

def measure_cold_start(model, port, idle_soak_s=COLD_SOAK_S):
    """Cold-start measurement with explicit idle soak.

    After *idle_soak_s* of silence, sends 1 cold request + 4 recovery
    requests back-to-back.  Records wall-clock timestamps of soak boundaries
    so the idle gap is unambiguous in the output.

    Returns dict with cold_ms, recovery_ms list, cold_penalty_ms.
    """
    soak_start = now_utc()
    log(f"  Cold-start: {idle_soak_s}s explicit idle soak "
        f"starting at {soak_start}...")
    time.sleep(idle_soak_s)
    soak_end = now_utc()
    log(f"  Idle soak ended at {soak_end}. Sending cold burst...")

    # Request 1: cold (DVFS ramp)
    t0 = time.time()
    try:
        send_vllm_request(model, port, prompt="Hello", max_tokens=32)
        cold_s = time.time() - t0
    except Exception as e:
        log(f"    Cold request FAILED: {e}")
        cold_s = None
    if cold_s is not None:
        log(f"    Request 1 (cold): {cold_s*1000:.1f}ms")
    else:
        log("    Request 1 (cold): FAILED")

    # Requests 2-5: recovery (back-to-back, no idle gap)
    recovery = []
    for i in range(N_RECOVERY):
        t0 = time.time()
        try:
            send_vllm_request(model, port, prompt="Hello", max_tokens=32)
            elapsed = time.time() - t0
            recovery.append(elapsed)
            log(f"    Request {i+2} (recovery): {elapsed*1000:.1f}ms")
        except Exception as e:
            log(f"    Request {i+2} FAILED: {e}")

    recovery_mean = sum(recovery) / len(recovery) if recovery else None
    cold_penalty = None
    if cold_s is not None and recovery_mean is not None:
        cold_penalty = cold_s - recovery_mean

    result = {
        "soak_start": soak_start,
        "soak_end": soak_end,
        "soak_duration_s": idle_soak_s,
        "cold_s": cold_s,
        "cold_ms": cold_s * 1000 if cold_s is not None else None,
        "recovery_s": recovery,
        "recovery_ms": [r * 1000 for r in recovery],
        "recovery_mean_ms": recovery_mean * 1000 if recovery_mean is not None else None,
        "cold_penalty_ms": cold_penalty * 1000 if cold_penalty is not None else None,
    }
    if cold_penalty is not None:
        log(f"    Cold penalty: {cold_penalty*1000:+.1f}ms "
            f"(cold={cold_s*1000:.1f}ms, "
            f"recovery_mean={recovery_mean*1000:.1f}ms)")
    return result


# ---------------------------------------------------------------------------
# Condition 1: bare idle
# ---------------------------------------------------------------------------

def run_bare_idle(output_path, gpu_ids, phase_duration, interval):
    """All GPUs, no CUDA context, no vLLM -- baseline reference."""
    log(f"\n{'='*60}")
    log("CONDITION 1: BARE IDLE (no CUDA context, all 4 GPUs)")
    log(f"{'='*60}")

    gpu_samples = record_phase_multi(
        output_path, gpu_ids, "bare_idle", "bare_idle",
        phase_duration, interval,
        extra_fields={
            "tp_size": 0, "dp_size": 0,
            "physical_gpus": gpu_ids, "replica_id": None,
        },
    )
    for gid in gpu_ids:
        summarize_samples(gpu_samples[gid], f"Bare idle GPU {gid}")

    return {"condition": "bare_idle", "gpu_samples": gpu_samples}


# ---------------------------------------------------------------------------
# Conditions 2-5: tensor parallelism
# ---------------------------------------------------------------------------

def run_tp_condition(output_path, gpu_ids, model, tp_size, env_flag,
                     phase_duration, interval, port=DEFAULT_PORT):
    """Run one TP condition end-to-end.

    Physical-to-logical GPU mapping:
      CUDA_VISIBLE_DEVICES = comma-joined physical IDs (e.g. "0,1")
      vLLM sees them as logical 0..N-1.
      All nvidia-smi queries use PHYSICAL indices via --id=<physical>.
    """
    flag_str = "flag_on" if env_flag else "baseline"
    condition = f"tp{tp_size}_{flag_str}"

    log(f"\n{'='*60}")
    log(f"CONDITION: TP={tp_size} {flag_str.upper()} "
        f"(GPUs {gpu_ids}, "
        f"CUDA_DISABLE_PERF_BOOST={'1' if env_flag else 'off'})")
    log(f"{'='*60}")

    # --- Start vLLM with tensor parallelism ---
    proc = start_vllm_server(
        gpu_ids, model, port, env_flag=env_flag, tp_size=tp_size)

    timeout = VLLM_TIMEOUT_S.get(tp_size, 600)
    log(f"  Waiting for vLLM ready (timeout={timeout}s)...")
    if not wait_for_vllm(port, timeout=timeout):
        log("  ERROR: vLLM did not start within timeout")
        stop_vllm(proc, f"vLLM-TP{tp_size}")
        return None
    log("  vLLM ready.")

    # --- NCCL stabilization ---
    # NCCL init spikes all GPUs to ~400W+.  Wait for clocks/power to settle
    # before any measurement.
    stabilize_s = NCCL_STABILIZE_S.get(tp_size, 120)
    log(f"  NCCL stabilization wait ({stabilize_s}s)...")
    time.sleep(stabilize_s)

    # --- Warmup (3 throwaway requests) ---
    log("  Warmup (3 requests)...")
    for _ in range(3):
        try:
            send_vllm_request(model, port)
        except Exception as e:
            log(f"    Warmup failed: {e}")

    # --- Warm latency (n=50 back-to-back, 128 tokens) ---
    log(f"  Measuring warm latency ({N_WARM} requests)...")
    latency = measure_warm_latency(model, port, n_requests=N_WARM)
    if latency.get("n", 0) > 0:
        log(f"  Warm latency: mean={latency['mean']*1000:.1f}ms  "
            f"p50={latency['p50']*1000:.1f}ms  "
            f"p99={latency['p99']*1000:.1f}ms")

    with open(output_path, "a") as fh:
        write_record(fh, {
            "timestamp": now_utc(),
            "phase": "warm_latency",
            "condition": condition,
            "model": model, "env_flag": env_flag,
            "tp_size": tp_size, "dp_size": 1,
            "physical_gpus": gpu_ids, "replica_id": None,
            "latency": latency,
        })

    # --- Decay curve (DVFS transition from active to idle) ---
    # Records how quickly clocks/power drop after the warm burst ends.
    # Replaces the old 60s stabilize — 5 min of recording also lets
    # the GPU fully settle before the idle power phase.
    log(f"  Recording decay curve ({DECAY_DURATION_S}s, "
        f"{DECAY_INTERVAL_S}s interval)...")
    decay_samples = record_phase_multi(
        output_path, gpu_ids, "decay_curve", condition,
        DECAY_DURATION_S, DECAY_INTERVAL_S,
        extra_fields={
            "model": model, "env_flag": env_flag,
            "tp_size": tp_size, "dp_size": 1,
            "physical_gpus": gpu_ids, "replica_id": None,
        },
    )

    # --- Steady-state idle power (per GPU) ---
    log(f"  Recording idle power ({phase_duration}s, {interval}s interval)...")
    gpu_samples = record_phase_multi(
        output_path, gpu_ids, "idle_power", condition,
        phase_duration, interval,
        extra_fields={
            "model": model, "env_flag": env_flag,
            "tp_size": tp_size, "dp_size": 1,
            "physical_gpus": gpu_ids, "replica_id": None,
        },
    )
    for gid in gpu_ids:
        summarize_samples(gpu_samples[gid], f"Idle GPU {gid} ({condition})")

    # --- Cold-start ramp ---
    # The idle power phase already provided 10-20 min of silence.
    # measure_cold_start adds an explicit 60 s soak on top for clean isolation.
    log("  Cold-start measurement...")
    cold_data = measure_cold_start(model, port, idle_soak_s=COLD_SOAK_S)

    with open(output_path, "a") as fh:
        write_record(fh, {
            "timestamp": now_utc(),
            "phase": "cold_start",
            "condition": condition,
            "model": model, "env_flag": env_flag,
            "tp_size": tp_size, "dp_size": 1,
            "physical_gpus": gpu_ids, "replica_id": None,
            "cold_start": cold_data,
        })

    # --- Shutdown ---
    log(f"  Stopping vLLM TP={tp_size}...")
    stop_vllm(proc, f"vLLM-TP{tp_size}")

    return {
        "condition": condition,
        "gpu_samples": gpu_samples,
        "warm_latency": latency,
        "cold_start": cold_data,
    }


# ---------------------------------------------------------------------------
# Conditions 6-7: data parallelism
# ---------------------------------------------------------------------------

def run_dp_condition(output_path, model, env_flag, phase_duration, interval):
    """Run one DP=2 condition: two fully independent vLLM instances.

    Physical-to-logical GPU mapping:
      Replica 0: CUDA_VISIBLE_DEVICES=0, port 8192
        Physical GPU 0 -> logical GPU 0 inside vLLM replica 0
      Replica 1: CUDA_VISIBLE_DEVICES=1, port 8193
        Physical GPU 1 -> logical GPU 0 inside vLLM replica 1

    The replicas do NOT share a CUDA context.  Verify via nvidia-smi that two
    distinct processes exist, one per GPU.
    """
    flag_str = "flag_on" if env_flag else "baseline"
    condition = f"dp2_{flag_str}"
    gpu_ids = [0, 1]  # physical GPUs used

    log(f"\n{'='*60}")
    log(f"CONDITION: DP=2 {flag_str.upper()} "
        f"(2 independent replicas, "
        f"CUDA_DISABLE_PERF_BOOST={'1' if env_flag else 'off'})")
    log(f"{'='*60}")

    # --- Start two independent vLLM instances ---
    procs = {}
    for replica_id in sorted(DP_REPLICA_GPUS):
        gpu_list = DP_REPLICA_GPUS[replica_id]
        port = DP_PORTS[replica_id]
        log(f"  Replica {replica_id}: physical GPU {gpu_list}, port {port}")
        procs[replica_id] = start_vllm_server(
            gpu_list, model, port, env_flag=env_flag, tp_size=1)

    # Wait for both to become ready
    for replica_id in sorted(procs):
        port = DP_PORTS[replica_id]
        log(f"  Waiting for replica {replica_id} (port {port})...")
        if not wait_for_vllm(port, timeout=VLLM_TIMEOUT_S[1]):
            log(f"  ERROR: Replica {replica_id} did not start within timeout")
            for p in procs.values():
                stop_vllm(p, "DP-replica")
            return None
    log("  Both replicas ready.")

    # Log PIDs for CUDA context isolation verification
    for replica_id in sorted(procs):
        log(f"  Replica {replica_id} pid={procs[replica_id].pid}")

    # Stabilize (no NCCL involved, but let CUDA contexts settle)
    log("  Stabilizing (60s)...")
    time.sleep(60)

    # --- Warmup both replicas ---
    log("  Warmup (3 requests to each replica)...")
    for replica_id in sorted(procs):
        port = DP_PORTS[replica_id]
        for _ in range(3):
            try:
                send_vllm_request(model, port)
            except Exception as e:
                log(f"    Warmup failed (replica {replica_id}): {e}")

    # --- Warm latency per replica ---
    latency_per_replica = {}
    for replica_id in sorted(procs):
        port = DP_PORTS[replica_id]
        log(f"  Warm latency: replica {replica_id} ({N_WARM} requests)...")
        lat = measure_warm_latency(model, port, n_requests=N_WARM)
        latency_per_replica[replica_id] = lat
        if lat.get("n", 0) > 0:
            log(f"  Replica {replica_id}: mean={lat['mean']*1000:.1f}ms  "
                f"p50={lat['p50']*1000:.1f}ms  "
                f"p99={lat['p99']*1000:.1f}ms")

        with open(output_path, "a") as fh:
            write_record(fh, {
                "timestamp": now_utc(),
                "phase": "warm_latency",
                "condition": condition,
                "model": model, "env_flag": env_flag,
                "tp_size": 1, "dp_size": 2,
                "physical_gpus": DP_REPLICA_GPUS[replica_id],
                "replica_id": replica_id,
                "latency": lat,
            })

    # --- Decay curve (DVFS transition from active to idle) ---
    log(f"  Recording decay curve ({DECAY_DURATION_S}s, "
        f"{DECAY_INTERVAL_S}s interval)...")
    decay_samples = record_phase_multi(
        output_path, gpu_ids, "decay_curve", condition,
        DECAY_DURATION_S, DECAY_INTERVAL_S,
        extra_fields={
            "model": model, "env_flag": env_flag,
            "tp_size": 1, "dp_size": 2,
            "physical_gpus": gpu_ids,
        },
        gpu_extra_fields={
            0: {"replica_id": 0},
            1: {"replica_id": 1},
        },
    )

    # --- Steady-state idle power (both GPUs) ---
    log(f"  Recording idle power ({phase_duration}s, {interval}s interval)...")
    gpu_samples = record_phase_multi(
        output_path, gpu_ids, "idle_power", condition,
        phase_duration, interval,
        extra_fields={
            "model": model, "env_flag": env_flag,
            "tp_size": 1, "dp_size": 2,
            "physical_gpus": gpu_ids,
        },
        gpu_extra_fields={
            0: {"replica_id": 0},
            1: {"replica_id": 1},
        },
    )
    for gid in gpu_ids:
        summarize_samples(
            gpu_samples[gid],
            f"Idle GPU {gid} / replica {gid} ({condition})")

    # --- Cold-start per replica ---
    # After the idle power phase (10-20 min of silence), each replica is cold.
    # Test each independently: replica 0 first, then replica 1.
    # Replica 1's DVFS state is unaffected by requests to replica 0 since
    # they are on different physical GPUs with independent CUDA contexts.
    cold_per_replica = {}
    for replica_id in sorted(procs):
        port = DP_PORTS[replica_id]
        gpu = DP_REPLICA_GPUS[replica_id][0]
        log(f"  Cold-start test: replica {replica_id} "
            f"(GPU {gpu}, port {port})")
        cold = measure_cold_start(model, port, idle_soak_s=COLD_SOAK_S)
        cold_per_replica[replica_id] = cold

        with open(output_path, "a") as fh:
            write_record(fh, {
                "timestamp": now_utc(),
                "phase": "cold_start",
                "condition": condition,
                "model": model, "env_flag": env_flag,
                "tp_size": 1, "dp_size": 2,
                "physical_gpus": DP_REPLICA_GPUS[replica_id],
                "replica_id": replica_id,
                "cold_start": cold,
            })

    # --- Shutdown both replicas ---
    for replica_id in sorted(procs):
        stop_vllm(procs[replica_id], f"DP-replica-{replica_id}")

    return {
        "condition": condition,
        "gpu_samples": gpu_samples,
        "warm_latency": latency_per_replica,
        "cold_start": cold_per_replica,
    }


# ---------------------------------------------------------------------------
# Conditions 8-9: mixed tensor + data parallelism
# ---------------------------------------------------------------------------

def run_tpdp_condition(output_path, model, env_flag, phase_duration, interval):
    """Run a TP=2 x DP=2 condition: two TP=2 vLLM instances on 4 GPUs.

    Physical-to-logical GPU mapping:
      Replica 0: CUDA_VISIBLE_DEVICES=0,1, port 8192, --tensor-parallel-size 2
        Physical GPUs 0,1 -> logical GPUs 0,1 inside vLLM replica 0
      Replica 1: CUDA_VISIBLE_DEVICES=2,3, port 8193, --tensor-parallel-size 2
        Physical GPUs 2,3 -> logical GPUs 0,1 inside vLLM replica 1

    Each replica has its own NCCL group for TP collectives.  There is no
    cross-replica NCCL communication (pure DP between the two groups).
    """
    flag_str = "flag_on" if env_flag else "baseline"
    condition = f"tpdp_{flag_str}"
    gpu_ids = [0, 1, 2, 3]  # all 4 physical GPUs

    log(f"\n{'='*60}")
    log(f"CONDITION: TP=2xDP=2 {flag_str.upper()} "
        f"(2 TP=2 groups across 4 GPUs, "
        f"CUDA_DISABLE_PERF_BOOST={'1' if env_flag else 'off'})")
    log(f"{'='*60}")

    # --- Start two TP=2 vLLM instances ---
    procs = {}
    for replica_id in sorted(TPDP_REPLICA_GPUS):
        gpu_list = TPDP_REPLICA_GPUS[replica_id]
        port = TPDP_PORTS[replica_id]
        log(f"  Replica {replica_id}: physical GPUs {gpu_list}, "
            f"port {port}, TP=2")
        procs[replica_id] = start_vllm_server(
            gpu_list, model, port, env_flag=env_flag, tp_size=2)

    # Wait for both to become ready
    for replica_id in sorted(procs):
        port = TPDP_PORTS[replica_id]
        log(f"  Waiting for replica {replica_id} (port {port})...")
        if not wait_for_vllm(port, timeout=VLLM_TIMEOUT_S[2]):
            log(f"  ERROR: Replica {replica_id} did not start within timeout")
            for p in procs.values():
                stop_vllm(p, "TPDP-replica")
            return None
    log("  Both TP=2 replicas ready.")

    for replica_id in sorted(procs):
        log(f"  Replica {replica_id} pid={procs[replica_id].pid}")

    # NCCL stabilization (TP=2 collectives within each group)
    stabilize_s = NCCL_STABILIZE_S[2]
    log(f"  NCCL stabilization wait ({stabilize_s}s)...")
    time.sleep(stabilize_s)

    # --- Warmup both replicas ---
    log("  Warmup (3 requests to each replica)...")
    for replica_id in sorted(procs):
        port = TPDP_PORTS[replica_id]
        for _ in range(3):
            try:
                send_vllm_request(model, port)
            except Exception as e:
                log(f"    Warmup failed (replica {replica_id}): {e}")

    # --- Warm latency per replica ---
    latency_per_replica = {}
    for replica_id in sorted(procs):
        port = TPDP_PORTS[replica_id]
        log(f"  Warm latency: replica {replica_id} ({N_WARM} requests)...")
        lat = measure_warm_latency(model, port, n_requests=N_WARM)
        latency_per_replica[replica_id] = lat
        if lat.get("n", 0) > 0:
            log(f"  Replica {replica_id}: mean={lat['mean']*1000:.1f}ms  "
                f"p50={lat['p50']*1000:.1f}ms  "
                f"p99={lat['p99']*1000:.1f}ms")

        with open(output_path, "a") as fh:
            write_record(fh, {
                "timestamp": now_utc(),
                "phase": "warm_latency",
                "condition": condition,
                "model": model, "env_flag": env_flag,
                "tp_size": 2, "dp_size": 2,
                "physical_gpus": TPDP_REPLICA_GPUS[replica_id],
                "replica_id": replica_id,
                "latency": lat,
            })

    # --- Decay curve (all 4 GPUs) ---
    log(f"  Recording decay curve ({DECAY_DURATION_S}s, "
        f"{DECAY_INTERVAL_S}s interval)...")
    decay_samples = record_phase_multi(
        output_path, gpu_ids, "decay_curve", condition,
        DECAY_DURATION_S, DECAY_INTERVAL_S,
        extra_fields={
            "model": model, "env_flag": env_flag,
            "tp_size": 2, "dp_size": 2,
            "physical_gpus": gpu_ids,
        },
        gpu_extra_fields={
            0: {"replica_id": 0}, 1: {"replica_id": 0},
            2: {"replica_id": 1}, 3: {"replica_id": 1},
        },
    )

    # --- Steady-state idle power (all 4 GPUs) ---
    log(f"  Recording idle power ({phase_duration}s, {interval}s interval)...")
    gpu_samples = record_phase_multi(
        output_path, gpu_ids, "idle_power", condition,
        phase_duration, interval,
        extra_fields={
            "model": model, "env_flag": env_flag,
            "tp_size": 2, "dp_size": 2,
            "physical_gpus": gpu_ids,
        },
        gpu_extra_fields={
            0: {"replica_id": 0}, 1: {"replica_id": 0},
            2: {"replica_id": 1}, 3: {"replica_id": 1},
        },
    )
    for gid in gpu_ids:
        replica = 0 if gid in TPDP_REPLICA_GPUS[0] else 1
        summarize_samples(
            gpu_samples[gid],
            f"Idle GPU {gid} / replica {replica} ({condition})")

    # --- Cold-start per replica ---
    cold_per_replica = {}
    for replica_id in sorted(procs):
        port = TPDP_PORTS[replica_id]
        gpus = TPDP_REPLICA_GPUS[replica_id]
        log(f"  Cold-start test: replica {replica_id} "
            f"(GPUs {gpus}, port {port})")
        cold = measure_cold_start(model, port, idle_soak_s=COLD_SOAK_S)
        cold_per_replica[replica_id] = cold

        with open(output_path, "a") as fh:
            write_record(fh, {
                "timestamp": now_utc(),
                "phase": "cold_start",
                "condition": condition,
                "model": model, "env_flag": env_flag,
                "tp_size": 2, "dp_size": 2,
                "physical_gpus": TPDP_REPLICA_GPUS[replica_id],
                "replica_id": replica_id,
                "cold_start": cold,
            })

    # --- Shutdown both replicas ---
    for replica_id in sorted(procs):
        stop_vllm(procs[replica_id], f"TPDP-replica-{replica_id}")

    return {
        "condition": condition,
        "gpu_samples": gpu_samples,
        "warm_latency": latency_per_replica,
        "cold_start": cold_per_replica,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(model, hf_cache_resolved):
    """Quick verification that infrastructure works before a full run.

    1. Verify 4 GPUs present with correct driver
    2. 60 s bare idle on all 4 GPUs
    3. Start vLLM TP=2 on GPUs 0,1, send one request, shut down
    4. Start vLLM DP=2 (two instances), send one request to each, shut down
    """
    log("=" * 60)
    log("SMOKE TEST")
    log("=" * 60)

    # --- Step 1: verify GPUs and driver ---
    n_gpus = count_gpus()
    log(f"  GPUs detected: {n_gpus}")
    if n_gpus < 4:
        log(f"  FAIL: need 4 GPUs, found {n_gpus}")
        sys.exit(1)

    ok, version = check_driver_version(0)
    if not ok:
        log(f"  FAIL: driver {version} < 580.105.08")
        sys.exit(1)
    log(f"  Driver: {version} (OK)")

    for gid in ALL_GPU_IDS:
        snap = query_nvidia_smi(gid)
        if snap:
            log(f"  GPU {gid}: {snap['gpu_name']}  "
                f"UUID={snap['uuid']}  "
                f"Temp={snap['gpu_temp_c']}C")
        else:
            log(f"  FAIL: cannot query GPU {gid}")
            sys.exit(1)
    log("  All 4 GPUs OK.")

    if hf_cache_resolved:
        log(f"  HF cache: {hf_cache_resolved}")

    # --- Step 2: 60 s bare idle recording ---
    log(f"\n{'='*60}")
    log("SMOKE: Bare idle (60s, all 4 GPUs)")
    log(f"{'='*60}")
    for gid in ALL_GPU_IDS:
        snap = query_nvidia_smi(gid)
        if snap:
            log(f"  GPU {gid}: {snap['power_w']}W, "
                f"{snap['sm_clock_mhz']}MHz, {snap['pstate']}")
    log("  Recording 60s of bare idle...")
    time.sleep(60)
    for gid in ALL_GPU_IDS:
        snap = query_nvidia_smi(gid)
        if snap:
            log(f"  GPU {gid} after 60s: {snap['power_w']}W, "
                f"{snap['sm_clock_mhz']}MHz, {snap['pstate']}")
    log("  Bare idle OK.")

    # --- Step 3: vLLM TP=2 ---
    log(f"\n{'='*60}")
    log("SMOKE: vLLM TP=2 (GPUs 0,1)")
    log(f"{'='*60}")
    proc_tp = start_vllm_server(
        TP2_GPU_IDS, model, DEFAULT_PORT, env_flag=False, tp_size=2)
    log(f"  Waiting for vLLM TP=2 (timeout=600s)...")
    if not wait_for_vllm(DEFAULT_PORT, timeout=600):
        log("  FAIL: vLLM TP=2 did not start")
        stop_vllm(proc_tp, "vLLM-TP2-smoke")
        sys.exit(1)
    log("  vLLM TP=2 ready. Sending test request...")
    try:
        resp = send_vllm_request(model, DEFAULT_PORT,
                                 prompt="Hello", max_tokens=16)
        log(f"  Response received: {len(str(resp))} bytes")
    except Exception as e:
        log(f"  FAIL: request failed: {e}")
        stop_vllm(proc_tp, "vLLM-TP2-smoke")
        sys.exit(1)
    log("  Shutting down TP=2...")
    stop_vllm(proc_tp, "vLLM-TP2-smoke")
    log("  TP=2 smoke OK.")

    # Cooldown before DP test
    log("  Cooldown (30s)...")
    time.sleep(30)

    # --- Step 4: vLLM DP=2 ---
    log(f"\n{'='*60}")
    log("SMOKE: vLLM DP=2 (GPU 0 port 8192, GPU 1 port 8193)")
    log(f"{'='*60}")
    dp_procs = {}
    for replica_id in sorted(DP_REPLICA_GPUS):
        gpu_list = DP_REPLICA_GPUS[replica_id]
        port = DP_PORTS[replica_id]
        dp_procs[replica_id] = start_vllm_server(
            gpu_list, model, port, env_flag=False, tp_size=1)

    for replica_id in sorted(dp_procs):
        port = DP_PORTS[replica_id]
        log(f"  Waiting for replica {replica_id} (port {port})...")
        if not wait_for_vllm(port, timeout=600):
            log(f"  FAIL: replica {replica_id} did not start")
            for p in dp_procs.values():
                stop_vllm(p, "DP-smoke")
            sys.exit(1)
    log("  Both DP=2 replicas ready.")

    for replica_id in sorted(dp_procs):
        port = DP_PORTS[replica_id]
        log(f"  Sending request to replica {replica_id} (port {port})...")
        try:
            resp = send_vllm_request(model, port,
                                     prompt="Hello", max_tokens=16)
            log(f"  Replica {replica_id} responded: {len(str(resp))} bytes")
        except Exception as e:
            log(f"  FAIL: replica {replica_id} request failed: {e}")
            for p in dp_procs.values():
                stop_vllm(p, "DP-smoke")
            sys.exit(1)

    # Verify CUDA context isolation via PIDs
    for replica_id in sorted(dp_procs):
        log(f"  Replica {replica_id} pid={dp_procs[replica_id].pid}")

    log("  Shutting down DP=2...")
    for replica_id in sorted(dp_procs):
        stop_vllm(dp_procs[replica_id], f"DP-replica-{replica_id}-smoke")
    log("  DP=2 smoke OK.")

    log(f"\n{'='*60}")
    log("SMOKE TEST PASSED")
    log(f"{'='*60}")


# ---------------------------------------------------------------------------
# Summary / hypothesis check
# ---------------------------------------------------------------------------

def _gpu_stats(samples):
    """Per-GPU statistics from a list of nvidia-smi samples."""
    powers = [s["power_w"] for s in samples if s.get("power_w") is not None]
    clocks = [s["sm_clock_mhz"] for s in samples
              if s.get("sm_clock_mhz") is not None]
    pstates = [s.get("pstate", "?") for s in samples]
    if not powers:
        return None
    mean_p = sum(powers) / len(powers)
    std_p = (sum((p - mean_p) ** 2 for p in powers) / len(powers)) ** 0.5
    mean_c = sum(clocks) / len(clocks) if clocks else 0
    pstate_mode = max(set(pstates), key=pstates.count) if pstates else "?"
    return {
        "mean_w": mean_p, "std_w": std_p,
        "mean_clock_mhz": mean_c, "pstate": pstate_mode,
        "n": len(powers),
    }


def _fmt_latency(lat):
    """Format latency stats dict for printing (ms)."""
    if not lat or lat.get("n", 0) == 0:
        return "N/A"
    return (f"mean={lat['mean']*1000:.1f}ms  "
            f"p50={lat['p50']*1000:.1f}ms  "
            f"p99={lat['p99']*1000:.1f}ms")


def _fmt_cold(cold):
    """Format cold-start data for printing."""
    if not cold or cold.get("cold_ms") is None:
        return "N/A"
    parts = [f"cold req 1: {cold['cold_ms']:.1f}ms"]
    if cold.get("recovery_mean_ms") is not None:
        parts.append(f"req 2-5 mean: {cold['recovery_mean_ms']:.1f}ms")
    if cold.get("cold_penalty_ms") is not None:
        parts.append(f"penalty: {cold['cold_penalty_ms']:+.1f}ms")
    return "   ".join(parts)


def print_summary(results):
    """Print the formatted multi-GPU results table."""
    log(f"\n{'='*60}")
    log("--- MULTI-GPU RESULTS ---")
    log(f"{'='*60}")

    # Bare idle
    bare = results.get("bare_idle")
    if bare:
        all_bare_powers = []
        for gid in sorted(bare["gpu_samples"]):
            stats = _gpu_stats(bare["gpu_samples"][gid])
            if stats:
                all_bare_powers.append(stats["mean_w"])
        if all_bare_powers:
            bare_mean = sum(all_bare_powers) / len(all_bare_powers)
            bare_std = (sum((p - bare_mean) ** 2 for p in all_bare_powers)
                        / len(all_bare_powers)) ** 0.5
            n_samples = _gpu_stats(
                bare["gpu_samples"][sorted(bare["gpu_samples"])[0]])["n"]
            log(f"Bare idle (4 GPUs):            "
                f"{bare_mean:.1f} +/- {bare_std:.1f} W per GPU  "
                f"(n={n_samples})")
    else:
        bare_mean = None
        log("Bare idle: not run")

    # TP, DP, and mixed conditions
    for cond_name, label in [
        ("tp2_baseline",  "TP=2 vLLM baseline"),
        ("tp2_flag_on",   "TP=2 vLLM flag on"),
        ("tp4_baseline",  "TP=4 vLLM baseline"),
        ("tp4_flag_on",   "TP=4 vLLM flag on"),
        ("dp2_baseline",  "DP=2 vLLM baseline"),
        ("dp2_flag_on",   "DP=2 vLLM flag on"),
        ("tpdp_baseline", "TP=2xDP=2 vLLM baseline"),
        ("tpdp_flag_on",  "TP=2xDP=2 vLLM flag on"),
    ]:
        cond = results.get(cond_name)
        if not cond:
            log(f"\n{label}: not run")
            continue

        is_multi_replica = cond_name.startswith(("dp2", "tpdp"))

        log(f"\n{label}:")
        total_power = 0
        n_gpus = 0
        for gid in sorted(cond["gpu_samples"]):
            stats = _gpu_stats(cond["gpu_samples"][gid])
            if stats:
                total_power += stats["mean_w"]
                n_gpus += 1
                replica_info = ""
                if cond_name.startswith("tpdp"):
                    replica = 0 if gid in TPDP_REPLICA_GPUS[0] else 1
                    replica_info = f" (replica {replica})"
                elif cond_name.startswith("dp2"):
                    replica_info = f" (replica {gid})"
                log(f"  GPU {gid}{replica_info}: "
                    f"{stats['mean_w']:.1f} +/- {stats['std_w']:.1f} W,  "
                    f"{stats['mean_clock_mhz']:.0f} MHz,  "
                    f"{stats['pstate']}")

        if n_gpus > 0 and bare_mean is not None:
            tax = total_power - bare_mean * n_gpus
            log(f"  Total: {total_power:.1f} W  "
                f"(parking tax vs bare: {tax:+.1f} W)")
        elif n_gpus > 0:
            log(f"  Total: {total_power:.1f} W")

        # Warm latency
        if is_multi_replica:
            lat = cond.get("warm_latency", {})
            for rid in sorted(lat):
                log(f"  Replica {rid} warm latency: {_fmt_latency(lat[rid])}")
        else:
            lat = cond.get("warm_latency")
            log(f"  Warm latency: {_fmt_latency(lat)}")

        # Cold-start
        if is_multi_replica:
            cold = cond.get("cold_start", {})
            for rid in sorted(cold):
                log(f"  Replica {rid} {_fmt_cold(cold[rid])}")
        else:
            cold = cond.get("cold_start")
            log(f"  {_fmt_cold(cold)}")


def print_hypothesis_check(results):
    """Compare measured values against single-GPU Paper 1 reference."""
    log(f"\n{'='*60}")
    log("--- HYPOTHESIS CHECK ---")
    log(f"{'='*60}")

    # Compute bare idle per-GPU mean
    bare = results.get("bare_idle")
    bare_mean = None
    if bare:
        bare_powers = []
        for gid in bare["gpu_samples"]:
            stats = _gpu_stats(bare["gpu_samples"][gid])
            if stats:
                bare_powers.append(stats["mean_w"])
        if bare_powers:
            bare_mean = sum(bare_powers) / len(bare_powers)

    # Parking tax comparison
    log(f"\nSingle H100 parking tax (from Paper 1): "
        f"{SINGLE_H100_PARKING_TAX_W} W")

    per_gpu_taxes = {}
    for cond_name, label in [
        ("tp2_baseline", "TP=2"), ("tp4_baseline", "TP=4"),
        ("dp2_baseline", "DP=2"), ("tpdp_baseline", "TP2xDP2"),
    ]:
        cond = results.get(cond_name)
        if not cond or bare_mean is None:
            log(f"{label} measured per-GPU tax: N/A")
            continue
        gpu_powers = []
        for gid in cond["gpu_samples"]:
            stats = _gpu_stats(cond["gpu_samples"][gid])
            if stats:
                gpu_powers.append(stats["mean_w"])
        if gpu_powers:
            per_gpu_tax = sum(gpu_powers) / len(gpu_powers) - bare_mean
            per_gpu_taxes[cond_name] = per_gpu_tax
            log(f"{label} measured per-GPU tax: {per_gpu_tax:.1f} W  "
                f"(expected ~{SINGLE_H100_PARKING_TAX_W} W if hypothesis holds)")

    # Parking tax conclusion
    tp_taxes = [per_gpu_taxes.get(c) for c in ("tp2_baseline", "tp4_baseline")
                if c in per_gpu_taxes]
    if tp_taxes:
        spread = max(tp_taxes) - min(tp_taxes)
        ref_diff = max(abs(t - SINGLE_H100_PARKING_TAX_W) for t in tp_taxes)
        if ref_diff < 15:  # within 15W of single-GPU
            log(f"Conclusion: tax DOES multiply linearly with N under TP "
                f"(per-GPU tax within {ref_diff:.0f}W of single-GPU reference)")
        else:
            log(f"Conclusion: tax does NOT match single-GPU under TP "
                f"(per-GPU deviation: {ref_diff:.0f}W)")

    # Cold-start comparison
    log(f"\nSingle H100 cold penalty (from latency_retest): "
        f"+{SINGLE_H100_COLD_PENALTY_MS} ms")

    for cond_name, label in [
        ("tp2_flag_on", "TP=2"), ("tp4_flag_on", "TP=4"),
        ("dp2_flag_on", "DP=2"), ("tpdp_flag_on", "TP2xDP2"),
    ]:
        cond = results.get(cond_name)
        if not cond:
            log(f"{label} measured cold penalty: N/A")
            continue

        is_multi_replica = cond_name.startswith(("dp2", "tpdp"))
        if is_multi_replica:
            cold_data = cond.get("cold_start", {})
            penalties = [cold_data[rid].get("cold_penalty_ms")
                         for rid in cold_data
                         if cold_data[rid].get("cold_penalty_ms") is not None]
            if penalties:
                avg_p = sum(penalties) / len(penalties)
                note = ("TP=2 within groups" if cond_name.startswith("tpdp")
                        else "independent replicas")
                log(f"{label} measured cold penalty: +{avg_p:.0f} ms  "
                    f"(expected ~{SINGLE_H100_COLD_PENALTY_MS}ms, {note})")
        else:
            cold_data = cond.get("cold_start", {})
            penalty = cold_data.get("cold_penalty_ms")
            if penalty is not None:
                straggler_note = ""
                if penalty > SINGLE_H100_COLD_PENALTY_MS * 1.3:
                    straggler_note = " -- possible straggler effect"
                log(f"{label} measured cold penalty: +{penalty:.0f} ms  "
                    f"(expected ~{SINGLE_H100_COLD_PENALTY_MS}ms if "
                    f"synchronized, more if stragglers){straggler_note}")

    # Cold-start conclusion
    tp_penalties = []
    for c in ("tp2_flag_on", "tp4_flag_on"):
        cond = results.get(c)
        if cond and cond.get("cold_start", {}).get("cold_penalty_ms") is not None:
            tp_penalties.append(cond["cold_start"]["cold_penalty_ms"])
    if tp_penalties:
        max_penalty = max(tp_penalties)
        if max_penalty > SINGLE_H100_COLD_PENALTY_MS * 1.5:
            log(f"Conclusion: cold-start ramp DOES compound under NCCL "
                f"(max penalty {max_penalty:.0f}ms vs "
                f"{SINGLE_H100_COLD_PENALTY_MS}ms single-GPU)")
        else:
            log(f"Conclusion: cold-start ramp does NOT significantly compound "
                f"under NCCL (max penalty {max_penalty:.0f}ms, within "
                f"1.5x of single-GPU)")


# ---------------------------------------------------------------------------
# Full experiment orchestration
# ---------------------------------------------------------------------------

def run_experiment(args):
    """Run all 9 conditions in order with inter-condition cooldowns."""

    # --- Pre-checks ---
    n_gpus = count_gpus()
    log(f"GPUs detected: {n_gpus}")
    if n_gpus < 4:
        log(f"ERROR: need 4 GPUs, found {n_gpus}")
        sys.exit(1)

    ok, version = check_driver_version(0)
    if not ok:
        log(f"ERROR: driver {version} < 580.105.08")
        sys.exit(1)
    log(f"Driver: {version} (OK)")

    gpu_info = {}
    for gid in ALL_GPU_IDS:
        snap = query_nvidia_smi(gid)
        if not snap:
            log(f"ERROR: cannot query GPU {gid}")
            sys.exit(1)
        gpu_info[gid] = snap
        log(f"GPU {gid}: {snap['gpu_name']}  UUID={snap['uuid']}")

    ok, temps = check_gpu_temperatures(ALL_GPU_IDS)
    temp_strs = [f"GPU{gid}={t:.0f}C" if t is not None else f"GPU{gid}=N/A"
                 for gid, t in sorted(temps.items())]
    log(f"Initial temperatures: {', '.join(temp_strs)}")
    if not ok:
        log(f"ABORT: GPU temperature exceeds {MAX_TEMP_C}C at start")
        sys.exit(1)

    phase_duration = 600 if args.quick else 1200  # 10 min vs 20 min
    interval = SAMPLE_INTERVAL
    model = args.model
    hf_cache = getattr(args, "hf_cache_resolved", None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"multi_gpu_{ts}.jsonl"
    manifest_path = output_dir / f"multi_gpu_{ts}_manifest.json"

    log(f"\nOutput: {output_path}")
    log(f"Phase duration: {phase_duration}s")
    log(f"Sample interval: {interval}s")
    log(f"Quick mode: {args.quick}")
    log(f"Model: {model}")
    if hf_cache:
        log(f"HF cache: {hf_cache}")

    manifest = {
        "experiment": "multi_gpu_perf_boost",
        "start_time": now_utc(),
        "n_gpus": n_gpus,
        "gpu_info": {
            str(gid): {"name": gpu_info[gid]["gpu_name"],
                        "uuid": gpu_info[gid]["uuid"]}
            for gid in ALL_GPU_IDS
        },
        "driver_version": version,
        "model": model,
        "phase_duration_s": phase_duration,
        "sample_interval_s": interval,
        "quick_mode": args.quick,
        "hf_cache": hf_cache,
        "conditions_planned": [
            "bare_idle",
            "tp2_baseline", "tp2_flag_on",
            "tp4_baseline", "tp4_flag_on",
            "dp2_baseline", "dp2_flag_on",
            "tpdp_baseline", "tpdp_flag_on",
        ],
        "conditions_completed": [],
    }

    output_path.touch()
    results = {}

    # --- Condition 1: Bare idle ---
    results["bare_idle"] = run_bare_idle(
        output_path, ALL_GPU_IDS, phase_duration, interval)
    manifest["conditions_completed"].append("bare_idle")
    inter_condition_cooldown(ALL_GPU_IDS)

    # --- Condition 2: TP=2 baseline ---
    r = run_tp_condition(
        output_path, TP2_GPU_IDS, model, tp_size=2, env_flag=False,
        phase_duration=phase_duration, interval=interval)
    if r:
        results["tp2_baseline"] = r
        manifest["conditions_completed"].append("tp2_baseline")
    inter_condition_cooldown(ALL_GPU_IDS)

    # --- Condition 3: TP=2 flag on ---
    r = run_tp_condition(
        output_path, TP2_GPU_IDS, model, tp_size=2, env_flag=True,
        phase_duration=phase_duration, interval=interval)
    if r:
        results["tp2_flag_on"] = r
        manifest["conditions_completed"].append("tp2_flag_on")
    inter_condition_cooldown(ALL_GPU_IDS)

    # --- Condition 4: TP=4 baseline ---
    r = run_tp_condition(
        output_path, TP4_GPU_IDS, model, tp_size=4, env_flag=False,
        phase_duration=phase_duration, interval=interval)
    if r:
        results["tp4_baseline"] = r
        manifest["conditions_completed"].append("tp4_baseline")
    inter_condition_cooldown(ALL_GPU_IDS)

    # --- Condition 5: TP=4 flag on ---
    r = run_tp_condition(
        output_path, TP4_GPU_IDS, model, tp_size=4, env_flag=True,
        phase_duration=phase_duration, interval=interval)
    if r:
        results["tp4_flag_on"] = r
        manifest["conditions_completed"].append("tp4_flag_on")
    inter_condition_cooldown(ALL_GPU_IDS)

    # --- Condition 6: DP=2 baseline ---
    r = run_dp_condition(
        output_path, model, env_flag=False,
        phase_duration=phase_duration, interval=interval)
    if r:
        results["dp2_baseline"] = r
        manifest["conditions_completed"].append("dp2_baseline")
    inter_condition_cooldown(ALL_GPU_IDS)

    # --- Condition 7: DP=2 flag on ---
    r = run_dp_condition(
        output_path, model, env_flag=True,
        phase_duration=phase_duration, interval=interval)
    if r:
        results["dp2_flag_on"] = r
        manifest["conditions_completed"].append("dp2_flag_on")
    inter_condition_cooldown(ALL_GPU_IDS)

    # --- Condition 8: TP=2 x DP=2 baseline ---
    r = run_tpdp_condition(
        output_path, model, env_flag=False,
        phase_duration=phase_duration, interval=interval)
    if r:
        results["tpdp_baseline"] = r
        manifest["conditions_completed"].append("tpdp_baseline")
    inter_condition_cooldown(ALL_GPU_IDS)

    # --- Condition 9: TP=2 x DP=2 flag on ---
    r = run_tpdp_condition(
        output_path, model, env_flag=True,
        phase_duration=phase_duration, interval=interval)
    if r:
        results["tpdp_flag_on"] = r
        manifest["conditions_completed"].append("tpdp_flag_on")

    # --- Save manifest ---
    manifest["end_time"] = now_utc()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"\n{'='*60}")
    log("EXPERIMENT COMPLETE")
    log(f"{'='*60}")
    log(f"Output: {output_path}")
    log(f"Manifest: {manifest_path}")

    # --- Print summary and hypothesis check ---
    print_summary(results)
    print_hypothesis_check(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU CUDA_DISABLE_PERF_BOOST experiment")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick infrastructure verification (~15 min, ~$3)")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 10-min idle phases instead of 20-min")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"vLLM model (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--interval", type=int, default=SAMPLE_INTERVAL,
        help=f"nvidia-smi sample interval in seconds (default: {SAMPLE_INTERVAL})")
    parser.add_argument(
        "--output-dir", type=str,
        default="data/raw/multi_gpu",
        help="Output directory for JSONL and manifest")
    parser.add_argument(
        "--hf-cache", type=str, default=None, metavar="DIR",
        help="Hugging Face cache root (sets HF_HOME, etc.)")
    args = parser.parse_args()

    args.hf_cache_resolved = setup_hf_cache(args.hf_cache)

    if args.smoke_test:
        run_smoke_test(args.model, args.hf_cache_resolved)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
