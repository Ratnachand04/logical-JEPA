# Phases 3b / 3c — Hierarchical slot loss and cardinality head

**Status: implemented, tested, measured. Both negative on synthetic `screw_board`.**
Neither the slot-set training term nor the cardinality scoring channel improves
detection, and neither recovers the missing-object anomalies they were built for.

## What changed

- `models/slot_bottleneck.py` — permutation-invariant helpers: Chamfer slot-set
  distance, sorted slot usage, background slot, region foreground mass.
- `models/cardinality.py` — `CardinalityHead`: (sorted context slot usage, region
  position, region size) → predicted foreground mass of the hidden region.
- `models/logical_jepa.py` — optional slot stage (`slots.enabled`):
  - slot module trained by reconstruction on **detached** teacher tokens;
  - **3b** slot term: predictions pasted into the teacher map vs the teacher map,
    compared as slot sets (Chamfer, cosine), slot parameters **frozen** for this
    term, weight λ ramped in from 30% of training;
  - **3c** cardinality loss: head only, inputs **stop-gradient**;
  - inference: per-window |predicted − actual| foreground mass, fixed slot
    initialisation so scores do not depend on batch position.
- `anomaly/scoring.py` — `anomaly.cardinality: off | add | only`, with its own
  per-position statistics fitted on normal images.
- `anomaly/metrics.py`, `datasets/synthetic_loco.py` — the generator writes a
  subtype manifest; evaluation reports AUROC per subtype and per direction
  (`removal`, `addition`, `rearrangement`). This is Phase 1's "next experiment 3".
- Studies 8 (`hierarchical`) and 9 (`cardinality`) in `configs/ablations.yaml`.
- 30 new tests (`test_hierarchical.py`, `test_cardinality.py`); 285 total.

All of it is off by default; every earlier number reproduces unchanged
(`test_cardinality_off_changes_nothing`).

## Study 8 — hierarchical slot loss

Synthetic `screw_board`, 80 epochs, **3 seeds**, `mean ± std`. Every arm trains
the slot stage; only λ differs.

| λ | image AUROC | logical | structural | removal | pixel AUROC | AU-PRO |
|---|---:|---:|---:|---:|---:|---:|
| **0 (control)** | **0.806 ± 0.022** | **0.812 ± 0.030** | **0.800 ± 0.049** | 0.573 ± 0.125 | **0.847** | **0.584** |
| 0.1 | 0.783 ± 0.035 | 0.796 ± 0.038 | 0.770 ± 0.032 | 0.537 ± 0.111 | 0.794 | 0.571 |
| 0.5 | 0.717 ± 0.081 | 0.739 ± 0.060 | 0.696 ± 0.102 | 0.533 ± 0.038 | 0.781 | 0.504 |
| 2 | 0.730 ± 0.062 | 0.762 ± 0.026 | 0.698 ± 0.127 | 0.473 ± 0.064 | 0.608 | 0.332 |

**The slot term only costs.** λ = 0.1 is inside seed noise; λ ≥ 0.5 is clearly
worse on every metric and far less stable across seeds. Phase 3a found slots
group by component *type*, not instance, so a set-level consistency term has
little compositional information to add over the patch term — and what it does
add competes with it.

## Study 9 — cardinality channel

Inference only, re-scoring the three λ = 0 checkpoints.

| scoring | image AUROC | logical | removal | addition | pixel AUROC | AU-PRO |
|---|---:|---:|---:|---:|---:|---:|
| JEPA, signed (baseline) | 0.806 ± 0.022 | 0.812 ± 0.030 | 0.573 ± 0.125 | 1.000 | **0.847** | **0.584** |
| JEPA, absolute (Study 7 fix) | **0.831 ± 0.046** | **0.829 ± 0.074** | 0.584 ± 0.194 | 1.000 | 0.676 | 0.487 |
| cardinality only | 0.541 ± 0.083 | 0.541 ± 0.117 | 0.509 ± 0.060 | 0.611 | 0.535 | 0.232 |
| JEPA + cardinality | 0.792 ± 0.012 | 0.792 ± 0.005 | 0.563 ± 0.139 | 1.000 | 0.693 | 0.430 |
| JEPA (absolute) + cardinality | 0.796 ± 0.004 | 0.794 ± 0.019 | 0.541 ± 0.129 | 1.000 | 0.640 | 0.384 |

**The cardinality channel does not carry signal.** Alone it is near chance
(0.541), including on removal (0.509) — the case it was designed for. Added to
the JEPA score it lowers image AUROC slightly and costs 0.15–0.21 pixel AUROC.

Likely cause: the target is the region's non-background attention share, and
the background slot is chosen per image by argmax usage. With type-level
grouping (one slot for all screws, one for the product body), a missing screw
moves little attention mass, and the argmax background choice can flip between
board and body across images — making the target noisy on normals and the
per-position statistics wide.

`dev_absolute` is the best detector (0.831) but sits inside seed noise of the
baseline (σ 0.046) and costs localisation (pixel 0.847 → 0.676). It does not fix
removal either.

## The missing-object finding survives every arm

| subtype | plain model, no slot stage (seed 42, single seed) |
|---|---:|
| `extra_screw`, `misplaced_screw` | 1.000 |
| `swapped_indicator` | 0.987 |
| `missing_screw` | 0.620 |
| `wrong_count` (two removed) | 0.433 |

Removal sits at 0.47–0.58 in all nine arms; addition is ~1.0 everywhere. So the
Phase 1 mechanism — *removed content is easier to predict, so it does not raise
latent error* — is now measured directly on synthetic data, by subtype, rather
than inferred from the logical aggregate. It is the open problem, and neither
3b nor 3c moves it.

## Two side findings

**Low `target_std` here is not collapse — and it is not precision either
(B5, settled).** `target_std` settles at ~0.08–0.11 rather than the ~0.5 the
README describes, with `cos_sim` ~0.999. A plain model with no slot stage does
the same and still scores image AUROC 0.855 / pixel 0.890 (seed 42) — above the
README's 0.762 for this configuration — so the training-time warning threshold
(0.05) is the right one and 0.1 is not a failure signal.

This was first seen on a GTX 1650 and attributed to **emulated bf16**. That
hypothesis has now been tested and **falsified**, on an RTX 4060 Laptop GPU
which reports `torch.cuda.is_bf16_supported() == True` (native bf16):

| epoch | `tstd_amp` (bf16) | `tstd_fp32` (`train.amp=false`) |
|---:|---:|---:|
| 1 | 0.4576 | 0.4574 |
| 40 | 0.0759 | 0.0758 |
| 80 | **0.0933** | **0.1144** |

The two trajectories agree to three or four decimal places at every logged
epoch, so disabling mixed precision changes nothing. Two independent reasons
rule out bf16: this GPU has native bf16 and still shows ~0.1 (a third run,
`collapse_check`, gave 0.1095), and pure fp32 shows it too. By the completion
guide's own decision table this is the middle row — *not precision; a property
of this torch version or run setup*.

The trajectory is also **not monotone toward zero**: it falls to ~0.076 by epoch
40 and recovers to 0.09–0.11 by epoch 80. A representation collapsing to a
constant does not come back. On a 200-image set of near-identical synthetic
boards, a low-variance target is the correct answer rather than a degenerate one.

*What would discriminate further:* re-run the same two commands with
`data.root=data/mvtec_loco_256`. Real LOCO normals vary far more than synthetic
ones, so that would show whether ~0.1 is a property of the data or of the setup.

**Checkpoint loading bug, fixed.** `load_model_from_checkpoint` merged the
caller's config over the checkpoint's, so an inference-only study whose config
merely *inherited* `slots.enabled: false` rebuilt the wrong architecture and
crashed. Architecture keys now always come from the checkpoint
(`test_checkpoint_architecture_survives_an_inherited_override`). No earlier
reported number was affected: all prior inference studies used the same
architecture as their checkpoints.

## Caveats

- Synthetic data only, 3 seeds, 80 epochs. The synthetic test split has 25
  normals and 6 images per logical subtype, so subtype AUROCs are coarse.
- Real MVTec LOCO was not run (license-gated download). Real `breakfast_box`
  and `splicing_connectors` have *distinguishable* components and may group by
  instance where `screw_board` does not — the one condition under which 3c
  could behave differently.

## Next experiments

1. **Removal-specific scoring.** Score the *expected* content of a region
   against what is there in the region's own direction: e.g. a per-position
   "should be occupied" prior from normals, compared with occupancy actually
   observed, instead of a context-conditioned mass.
2. **Re-run Study 9 on real LOCO**, starting with `splicing_connectors`.
3. **Re-check `target_std`** on a native-bf16 or fp32 run to confirm the
   ~0.1 level is precision-related.

## Reproduce

```bash
py -3.12 scripts/download_data.py --synthetic
py -3.12 run_ablations.py --config configs/ablations.yaml --study hierarchical \
  --seeds 0,1,2 --set experiment.name=phase3bc data.num_workers=0
py -3.12 run_ablations.py --config configs/ablations.yaml --study cardinality \
  --seeds 0,1,2 --no-retrain --set experiment.name=phase3bc data.num_workers=0
```

~6.5 h on a GTX 1650 (≈12 min training + 9 min evaluation per arm).
