---
name: Thesis direction - static flag enablement
description: Paper prescription reframed from "dynamic toggling" to "flag set at process start" after reviewer identified env var can't be toggled on running process
type: project
---

Original CEO directive (~2026-04-24): prescribe dynamic flag toggling. After reviewer feedback (2026-04-27), reframed to **static flag enablement at process start**. The data actually shows something better than toggling: with CUDA_DISABLE_PERF_BOOST=1 set before the inference server starts, the DVFS governor automatically drops clocks when idle and ramps on demand. No runtime toggling needed.

**Why:** The env var is read at CUDA context creation — it literally cannot be toggled on a running vLLM process without recreating the context. The decay curve data measures flag-at-start behavior, not mid-process toggling. Reframing as "set and forget" is both more accurate and more actionable (simpler deployment story).

**How to apply:** The paper now prescribes: "Set the flag in your container spec / launcher script. The GPU handles the rest." The scheduler evaluation (§7) is positioned as a fallback for older drivers / non-NVIDIA hardware.
