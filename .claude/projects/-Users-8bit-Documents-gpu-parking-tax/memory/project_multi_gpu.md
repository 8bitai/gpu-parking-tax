---
name: Multi-GPU experiment
description: Multi-GPU CUDA_DISABLE_PERF_BOOST experiment (TP/DP) at experiments/multi_gpu/multi_gpu.py, shared utils at experiments/_common/utils.py
type: project
---

Multi-GPU parking tax experiment added 2026-04-24.

**Why:** Extend Paper 1 single-GPU findings to multi-GPU configs. Hypothesis: parking tax multiplies by N under TP; DVFS cold-start ramp may compound under NCCL (straggler effect). DP=2 serves as architectural baseline (should match single-GPU).

**How to apply:** 7 conditions (bare idle, TP=2/4 baseline/flag, DP=2 baseline/flag). Shared utilities extracted to experiments/_common/utils.py (perf_boost.py unchanged for Paper 1 reproducibility). Target: 4xH100 SXM RunPod node, ~$25 quick / ~$60 full.
