# Phase 5 — Inference latency

**Status: target hit. <100 ms/image reached with the full sweep intact, and
6.6 ms/image with a Study-3-justified single-scale sweep.**

## What changed

`scripts/benchmark_latency.py` — measures a full sweep across resolution, sweep
configuration, batch size and mixed precision. Every number below is measured
(median of 5 timed runs after 2 warm-ups, CUDA-synchronised), never projected.

## Measured — RTX 4060 Laptop

Baseline for reference: the README reports **258 ms/image**. The first row below
reproduces it at 260 ms, which is the check that the harness is measuring the
same thing.

### 256 px (16×16 grid) — the shipped configuration

| sweep config | configs/img | batch | AMP | ms/image |
|---|---:|---:|:--|---:|
| full sweep (2,4,6) | 299 | 1 | off | **260.2** ← baseline |
| full sweep (2,4,6) | 299 | 1 | on | 109.6 |
| full sweep (2,4,6) | 299 | 4 | off | 238.9 |
| **full sweep (2,4,6)** | 299 | 4 | on | **86.3** ← target met, nothing given up |
| single scale (4) | 49 | 1 | on | 35.9 |
| single scale (4) | 49 | 4 | on | 15.9 |
| single scale, coarse stride | 16 | 4 | on | **6.6** ← fastest |

### 384 px (24×24 grid)

| sweep config | configs/img | batch | AMP | ms/image |
|---|---:|---:|:--|---:|
| full sweep (3,6,9) | 558 | 1 | on | 433.4 |
| single scale (6) | 49 | 1 | on | 47.0 |
| single scale, coarse stride | 16 | 4 | on | 14.9 |

Window sizes are scaled with the grid so the *physical* window stays constant —
otherwise a resolution change would silently also be a window-size change, and
the comparison would be measuring two things at once.

## What actually bought the speedup

**Mixed precision: 2.4× on its own** (260 → 110 ms), free and lossless for this
purpose — the sweep is inference-only, so there is no optimiser state to keep in
fp32.

**Batching the sweep: a further 1.3×** (110 → 86 ms). The sweep already folds
mask positions into the batch dimension; adding images on top just fills the GPU
better.

**Dropping to a single sweep scale: 5.4×** (86 → 16 ms). This is the large one,
and it is *not* a free lunch dressed up as one — it is justified by Study 3,
which measured all five sweep configurations within 0.016 AUROC of each other.
Paying 299 forward configurations for a scale mix that does not change the answer
was the actual waste.

**Coarse stride: a further 2.4×** (16 → 6.6 ms), at the cost of localisation
resolution. This is the one knob here that genuinely trades accuracy for speed,
so it is opt-in rather than default.

## Resolution is expensive, and more so than it looks

384 px costs ~4× the full-sweep time of 256 px (433 vs 110 ms, both AMP). That is
worse than the token-count ratio (576/256 = 2.25×) because the number of sweep
positions *also* grows with the grid — 558 vs 299 configurations. Anyone raising
resolution to chase small structural defects should budget for both factors, and
should expect to drop to a single sweep scale to pay for it.

## Recommendation

- **Default (accuracy first):** full sweep, AMP on, batch 4 → **86 ms/image**.
  Nothing is traded away; this is strictly the baseline configuration computed
  faster.
- **Throughput:** single scale, AMP, batch 4 → **16 ms/image**, supported by
  Study 3.
- ONNX / TensorRT export was **not** implemented. It is unnecessary for the
  stated target — that was met with AMP and batching alone — and would add an
  export path that has to be kept in step with the PyTorch model for no measured
  benefit. This is a deliberate omission, not an oversight.

## Reproduce

```bash
py -3.12 scripts/benchmark_latency.py \
  --checkpoint checkpoints/logical_jepa_multiscale/screw_board/final.pt \
  --resolutions 256,384 --batch-sizes 1,4
```
