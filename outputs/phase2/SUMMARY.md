# Phase 2 — Fixing the diagnosed defects

## 2a. Per-block loss weighting

**Status: implemented and unit-tested; the measured arm runs as Study 5.**

`loss.reduction: {per_patch, per_block}`, defaulting to `per_patch` so every
number already in the README stays reproducible.

Under `per_patch` a block's influence scales with its area, so one 6×6 block (36
patches) outweighs three 2×2–3×3 blocks (~19 patches combined) roughly 2:1 —
meaning the `multiscale` arm was effectively training on its large-scale
objective alone. `per_block` averages within each block first, removing exactly
that asymmetry and nothing else.

The arithmetic is pinned by `tests/test_loss_reduction.py`, including the two
cases that matter for interpreting Study 5:

- the two modes **must** differ on unequal blocks (otherwise the study is a
  no-op that still produces a table);
- the two modes **must agree** when all blocks are the same size — which is why
  `random_patch` and `small` are included as controls. If those move, something
  other than the intended variable changed.

`jepa_loss` now also reports `loss_small` and `loss_large` separately, so a
scale whose term has stalled is visible during training rather than only in the
final metrics.

**Reading the result when Study 5 finishes:** if `multiscale` closes the gap to
`random_patch`, loss weighting was the cause. If it still loses, the second
hypothesis — anomaly scale versus block size — becomes the leading explanation
and the next experiment is block-size ranges, not loss weighting.

## 2b. Resolution

**Status: supported and measured.**

384 px (24×24 grid) and 512 px (32×32) work; `tests/test_model.py` covers the
patch-embed and transform paths at all three resolutions.

The cost is worse than token count alone suggests, and this is the point worth
recording: 384 px costs **~4×** the full-sweep time of 256 px (433 vs 110 ms,
both AMP), not the 2.25× the token ratio implies. The sweep grows too — 558
window positions versus 299. Anyone raising resolution to chase small structural
defects must budget for both factors. Full table in
`outputs/phase5_latency/SUMMARY.md`.

## 2c. Decoupled detection / localization normalization

**Status: implemented, measured, and a clear win.**

Study 4 established that no single normalization is best at both jobs. Rather
than compromise, the scorer now builds **two** maps:

- `anomaly.normalize_detection` (default `global`, per-position statistics
  fitted on normal images) drives the image-level verdict;
- `anomaly.normalize_localization` (default `zscore`, per-image) drives the
  heatmap and the pixel-level metrics.

`evaluate.py` reports each metric from the map built for its purpose, labels
both settings in the results, and additionally reports what localization *would*
have been under the detection normalization — so the trade-off stays visible in
one run instead of requiring two.

### Measured (synthetic `screw_board`, single seed)

| configuration | image AUROC | pixel AUROC | AU-PRO |
|---|---:|---:|---:|
| `global` for both (Study 4 winner on detection) | 0.842 | 0.615 | 0.554 |
| `zscore` for both (Study 4 winner on localization) | 0.461 | 0.876 | 0.655 |
| **decoupled (`global` + `zscore`)** | **0.842** | **0.873** | **0.661** |

**Pixel AUROC 0.615 → 0.873 (+0.258) and AU-PRO 0.554 → 0.661 (+0.107), with
image AUROC unchanged at 0.842.** The two objectives were never actually in
conflict — they were being forced through one map. Computing both costs one
extra fusion pass over an already-computed set of error grids; the sweep, which
dominates inference, runs once either way.

`tests/test_decoupled_norm.py` pins the safety property that makes this sound:
**changing the localization setting cannot move the image score.** A
presentation-layer choice must not be able to alter a reported detection number.

### One behaviour worth knowing

`global` falls back to `zscore` until the per-position statistics have been
fitted. That keeps a freshly built scorer usable instead of crashing, but it
means an *uncalibrated* scorer silently gives you per-image behaviour. The
fallback is explicit in `normalize_grid` and covered by
`test_uncalibrated_global_falls_back_to_zscore`.

## Reproduce

```bash
py -3.12 evaluate.py --checkpoint <ckpt> --config configs/loco.yaml
py -3.12 run_ablations.py --study loss_reduction --seeds 0,1,2
py -3.12 -m pytest tests/test_loss_reduction.py tests/test_decoupled_norm.py
```
