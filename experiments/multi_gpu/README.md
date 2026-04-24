# Multi-GPU CUDA_DISABLE_PERF_BOOST Experiment

Tests whether the GPU parking tax (elevated idle power from a persistent CUDA
context) multiplies under tensor parallelism (TP) and data parallelism (DP)
on a 4×H100 SXM node.

## Prerequisites

- 4× NVIDIA H100 SXM with NVLink (RunPod Secure Cloud recommended)
- NVIDIA driver ≥ 580.105.08
- Python packages: `vllm`, `transformers`
- Model: `Qwen/Qwen2.5-7B` (downloaded automatically on first run)

## Running the smoke test

The smoke test verifies infrastructure before committing to a full run.
It checks GPU count, driver version, starts vLLM in both TP=2 and DP=2
configurations, sends a test request to each, and shuts down cleanly.

```bash
python multi_gpu.py --smoke-test
# Optional: point HF cache to a large volume
python multi_gpu.py --smoke-test --hf-cache /workspace/.cache/huggingface
```

**Expected time:** ~15 minutes
**Expected cost:** ~$3 (at ~$12/hr for 4×H100)

### What to look for

- `SMOKE TEST PASSED` at the end
- All 4 GPUs detected with correct driver
- vLLM TP=2 starts and responds on port 8192
- vLLM DP=2 starts two instances (ports 8192, 8193) and both respond
- Clean shutdown with no SIGKILL escalations

## Running the full experiment

### Quick mode (10-minute idle phases)

```bash
python multi_gpu.py --quick
```

**Expected time:** ~3.5 hours
**Expected cost:** ~$35
**Samples per condition:** n=20 (30s interval × 10 min)

### Full mode (20-minute idle phases)

```bash
python multi_gpu.py
```

**Expected time:** ~6.5 hours
**Expected cost:** ~$75
**Samples per condition:** n=40 (30s interval × 20 min)

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--quick` | off | 10-min idle phases instead of 20-min |
| `--model` | `Qwen/Qwen2.5-7B` | vLLM model to serve |
| `--interval` | 30 | Seconds between nvidia-smi samples |
| `--output-dir` | `data/experiments/multi_gpu` | Output directory |
| `--hf-cache` | auto | HF cache root (auto-detects `/workspace`) |

## Conditions

The experiment runs 9 conditions in fixed order:

| # | Config | GPUs | Flag | Purpose |
|---|--------|------|------|---------|
| 1 | Bare idle | 4 | — | Baseline (no CUDA context) |
| 2 | TP=2 baseline | 0,1 | off | Per-GPU parking tax under TP |
| 3 | TP=2 flag on | 0,1 | on | Does flag work with NCCL? |
| 4 | TP=4 baseline | 0–3 | off | Tax scaling with N |
| 5 | TP=4 flag on | 0–3 | on | Flag at TP=4, cold-start with 4-way sync |
| 6 | DP=2 baseline | 0,1 | off | Two independent replicas |
| 7 | DP=2 flag on | 0,1 | on | Independent contexts, independent cold-starts |
| 8 | TP2xDP2 baseline | 0–3 | off | Mixed parallelism (real deployment pattern) |
| 9 | TP2xDP2 flag on | 0–3 | on | Flag under mixed TP+DP |

Each vLLM condition measures:
1. **Warm latency** — 50 back-to-back requests (128 tokens, Qwen2.5-7B fp16)
2. **DVFS decay curve** — 5 min at 1s nvidia-smi sampling after warm burst
   (captures how quickly clocks/power drop; essential for dynamic flag-toggle
   timing)
3. **Steady-state idle power** — per-GPU nvidia-smi sampling at 30s intervals
4. **Cold-start ramp** — 60s explicit idle soak, then 1 cold + 4 recovery requests

## Output files

Each run produces two files in `--output-dir`:

- **`multi_gpu_YYYYMMDD_HHMMSS.jsonl`** — one JSON record per nvidia-smi
  sample per GPU, plus latency and cold-start records.  Same schema as the
  single-GPU experiment with additional fields: `tp_size`, `dp_size`,
  `physical_gpus`, `replica_id`.

- **`multi_gpu_YYYYMMDD_HHMMSS_manifest.json`** — experiment metadata
  (conditions run, timing, driver version, GPU list).

## Interpreting cross-GPU power numbers

### Per-GPU parking tax

```
Parking tax (per GPU) = condition idle power − bare idle power
```

If the hypothesis holds, per-GPU tax under TP should match the single-GPU
value (~50 W on H100).  The **total** tax scales linearly with N:

- TP=2: ~100 W total (2 × 50 W)
- TP=4: ~200 W total (4 × 50 W)

### What "flag works under NCCL" means

If `CUDA_DISABLE_PERF_BOOST=1` drops each GPU back to bare-idle power under
TP, the flag works with NCCL — each GPU's CUDA driver respects the flag
independently.

### Cold-start penalty under TP

Under tensor parallelism, all GPUs must DVFS-ramp before the first NCCL
collective completes.  The measured cold penalty could be:

- **~Same as single-GPU** — GPUs ramp simultaneously, no straggler
- **Worse than single-GPU** — the collective waits for the slowest GPU
  to ramp (straggler effect)

### DP as architectural baseline

DP=2 uses independent CUDA contexts on separate GPUs.  Both per-GPU tax
and cold-start penalty should match single-GPU values.  If they don't,
something is wrong with the methodology or there is an unexpected
cross-GPU interaction.

### TP=2 x DP=2 (mixed parallelism)

Real deployments commonly use mixed parallelism (e.g., TP=2 within a node,
DP=2 across replica groups).  This condition uses all 4 GPUs: two TP=2
groups of 2 GPUs each.  Expected behavior:

- Per-GPU tax matches TP=2 (each GPU has one CUDA context, same as pure TP)
- Cold-start penalty matches TP=2 (NCCL collectives are within TP groups)
- No cross-group NCCL, so the two groups should be independent

### DVFS decay curve

The decay curve records how quickly GPU clocks and power drop after the
last inference request.  This is critical for dynamic flag toggling:

- **When to engage the flag:** once power has settled to idle level
- **Cost of engaging too early:** negligible (flag doesn't affect active power)
- **Cost of disengaging on request arrival:** cold-start DVFS ramp penalty

Plot `power_w` and `sm_clock_mhz` vs `timestamp` from records with
`phase="decay_curve"` to visualize the transition.
