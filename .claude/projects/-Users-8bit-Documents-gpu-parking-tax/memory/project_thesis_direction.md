---
name: Thesis direction from CEO
description: CEO directive to orient thesis toward dynamic flag toggling prescription (flag on during hot idle, off on request arrival)
type: project
---

CEO directive (received ~2026-04-24): thesis should prescribe dynamically toggling CUDA_DISABLE_PERF_BOOST=1 when a model is stored hot in memory (idle), and automatically disabling it when an inference request arrives. This requires engineering (a runtime toggle mechanism). The paper should validate the flag's behavior under multi-GPU and characterize the cold-start penalty as the cost of this prescription.

**Why:** Shifts the paper from pure measurement to an actionable optimization. The decay curve (how fast power drops after last request) determines *when* to engage the flag. The cold-start ramp determines the latency cost of disengaging it.

**How to apply:** Decay curve data is now essential (not supplementary). Cold-start penalty characterization under TP/DP is the key trade-off analysis. Mixed TP+DP condition matters because real deployments use it.
