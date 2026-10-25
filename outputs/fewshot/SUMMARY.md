# Few-shot Logical-JEPA with a pretrained encoder

**Status: built, tested and measured on synthetic data and two real MVTec LOCO
categories. From 16 normal images, the shipped few-shot model (JEPA + memory
bank on a frozen DINO encoder) scores 0.726 image AUROC on `juice_bottle`
against 0.668 for the from-scratch model trained on all 335 — and does not
match the full-data model on `pushpins`. Diagnostics locate the `pushpins` gap
in the frozen backbone, not in the JEPA or the calibration. On both categories
a memory bank alone scores higher than any combination with the JEPA error.**

## What changed

- **The encoder can start from self-supervised pretrained weights.**
  `encoder.pretrained: dino_vits16` loads DINO ViT-S/16 (86.7 MB, Apache-2.0,
  trained on ImageNet without labels) into the existing context encoder; the
  EMA teacher starts as its copy. Masking, the predictor, the sweep, scoring
  and calibration are unchanged — it is the same JEPA with different starting
  weights.
- **Why DINO and not I-JEPA.** Meta released I-JEPA only as ViT-H/14: a
  10.4 GB checkpoint and ~632M parameters. The anomaly sweep runs ~299 forward
  passes per image, which that encoder cannot do on an 8 GB laptop GPU. DINO
  is the closest practical substitute: same family (self-supervised, EMA
  teacher distilling into a student), same 16 px patch, ~22M parameters.
- **The transfer is verified, not assumed.** `test_pretrained.py` compares our
  encoder against a ViT forward written directly from the raw checkpoint
  tensors: cosine 0.9999998, max |diff| 4.5e-5. A partial load is refused.
- **Few-shot training.** `data.num_train_images: K` trains on a reproducible
  subset of K normal images. The same K images are the only ones the
  per-position statistics are fitted on; validation normals place the
  threshold, as in every configuration; the test set is never thinned. By
  default a few-shot epoch is as long as a full-data epoch, so arms differ in
  how many *distinct* images they see, never in optimiser steps.
- **`encoder.finetune: frozen`** (default for pretrained) trains only the
  predictor — 1.48M parameters.
- **Rebuilding a checkpoint never downloads anything**; evaluation and the
  demo work offline.
- `configs/fewshot.yaml` (the shipped few-shot model), Study 10 and Study 11
  in `configs/ablations.yaml`, `scripts/fetch_pretrained.py`,
  `scripts/diagnose_fewshot.py`. The default `configs/loco.yaml` is unchanged
  and still builds everything from scratch.

## Result 1 — synthetic `screw_board`: saturated

3 seeds each, 80 epochs, identical step counts. Each seed draws its own
16-image subset.

| arm | image AUROC | logical | structural | pixel | removal |
|---|---:|---:|---:|---:|---:|
| from scratch, all 200 images | 0.796 ± 0.128 | 0.768 | 0.823 | 0.794 | 0.494 |
| from scratch, 16 images | 0.735 ± 0.124 | 0.715 | 0.756 | 0.806 | 0.387 |
| **DINO frozen, 16 images** | **1.000 ± 0.000** | 1.000 | 1.000 | 0.980 | 1.000 |

Checked rather than celebrated: the full 85-image test set was scored, the
statistics came from the 16 training images, the threshold from 20 validation
normals. The result is real, but it says the synthetic set is too easy for
pretrained features. **It is not evidence about real data**, and the next
result shows why that distinction matters.

Two side observations from the from-scratch control:

- Its seed spread is very large (0.705 / 0.740 / 0.942).
- Its teacher's `target_std` falls to 0.05–0.28 while the frozen DINO teacher
  stays at ~0.80. But the from-scratch seed AUROC does **not** track
  `target_std` (the best seed ended at 0.125, the worst 16-image seed at
  0.418), so the drift does not explain the variance.

## Result 2 — real `pushpins`: the few-shot model does not win

3 seeds each, same schedule.

| arm | image AUROC | logical | structural | pixel | AU-PRO |
|---|---:|---:|---:|---:|---:|
| from scratch, all 372 images | **0.643 ± 0.006** | 0.453 | **0.856** | 0.417 | 0.377 |
| from scratch, 16 images | 0.574 ± 0.050 | 0.494 | 0.663 | 0.418 | 0.377 |
| DINO frozen, 16 images | 0.544 ± 0.006 | 0.501 | 0.592 | **0.590** | 0.376 |

The from-scratch control reproduces Phase 1 (0.632–0.645). At 16 images the
pretrained init is no better than from scratch on detection, though it is far
more stable across seeds (±0.006 vs ±0.050) and localises better (pixel 0.59
vs 0.42). Neither 16-image model reaches the full-data one.

## Result 3 — real `juice_bottle`: 16 images match 335

| arm | image AUROC | logical | structural | pixel | AU-PRO |
|---|---:|---:|---:|---:|---:|
| from scratch, all 335 images | 0.668 ± 0.052 | 0.683 | 0.645 | 0.784 | 0.488 |
| **DINO frozen, 16 images** | **0.678 ± 0.010** | **0.707** | 0.634 | **0.849** | **0.633** |

Indistinguishable on image AUROC (the runner's check: 0.3σ of seed noise)
with **5% of the training images**, better on localisation (AU-PRO +0.145),
and five times more stable across seeds.

## Why `pushpins` fails — three diagnostics

All three run on existing checkpoints or none
(`scripts/diagnose_fewshot.py`), so each took minutes, not a training run.

### 1. The calibration is biased — but that is not the cause

The predictor trains ~60 passes per image over its 16 images, and the
statistics are fitted on those same images, so they understate the error on
any image it has not seen. Held-out normals scored a mean of 28.6, against
3.4 for the from-scratch model. Refitting the statistics on images the model
never saw removes the inflation, **and barely moves the AUROC**:

| statistics fitted on | image | logical | structural | held-out normal score |
|---|---:|---:|---:|---:|
| the 16 training images (as evaluated) | 0.543 | 0.524 | 0.563 | 28.45 |
| 16 images it never saw | 0.589 | 0.556 | 0.627 | 4.88 |
| 64 images it never saw | 0.532 | 0.484 | 0.586 | 2.49 |
| all 356 images it never saw | 0.548 | 0.504 | 0.597 | — |

(DINO 16-shot, seed 0.)

### 2. The backbone is the limit on this category

A training-free probe: patch features of the same 16 images, from every DINO
layer, as a nearest-neighbour memory bank; each test image scored by its
patch farthest from the bank. The layer with the best image AUROC is shown
(`scripts/diagnose_fewshot.py layers`):

| category | best layer | image | logical | structural |
|---|---:|---:|---:|---:|
| juice_bottle | 3 | 0.821 | 0.816 | 0.830 |
| breakfast_box | 6 | 0.764 | 0.781 | 0.748 |
| splicing_connectors | 9 | 0.725 | 0.676 | 0.786 |
| screw_bag | 6 | 0.661 | 0.533 | 0.875 |
| **pushpins** | 6 | **0.650** | 0.617 | 0.687 |

Even without the JEPA, nothing built on these features exceeds 0.65 on
`pushpins`: a bent pin tip is small against a 16 px patch of a 256 px image.
The from-scratch model gets structural 0.856 there only because it learned
pushpin-specific features from 372 images. Run this probe before training a
few-shot model on a new category — it takes a minute.

### 3. The JEPA error is the weaker signal on frozen features

The same probe beats the JEPA prediction error built on those features, on
both categories. That motivated Study 11.

## Result 4 — Study 11: a memory-bank channel

Inference only: the Study 10 `fs_dino_k16` checkpoints re-scored three ways,
3 seeds each. The bank holds the teacher features of exactly the 16 training
images (4,096 patches); its statistics use leave-one-image-out distances. The
weight is fixed at 1 in advance — both channels are in z-units — not chosen on
the test set.

| `pushpins` | image AUROC | logical | structural | pixel | AU-PRO |
|---|---:|---:|---:|---:|---:|
| `mem_off` — JEPA error only | 0.544 ± 0.006 | 0.501 | 0.592 | 0.590 | 0.376 |
| `mem_add` — JEPA + memory (shipped) | 0.559 ± 0.002 | 0.508 | 0.616 | 0.589 | 0.410 |
| `mem_only` — memory bank alone | 0.585 ± 0.041 | 0.534 | 0.641 | 0.577 | 0.377 |

| `juice_bottle` | image AUROC | logical | structural | pixel | AU-PRO |
|---|---:|---:|---:|---:|---:|
| `mem_off` — JEPA error only | 0.678 ± 0.010 | 0.707 | 0.634 | 0.849 | 0.633 |
| `mem_add` — JEPA + memory (shipped) | **0.726 ± 0.015** | 0.754 | 0.685 | 0.892 | 0.760 |
| `mem_only` — memory bank alone | 0.796 ± 0.009 | 0.810 | 0.775 | 0.904 | 0.787 |

`mem_off` reproduces Study 10 exactly, so the new code path changes nothing
when the channel is off.

- **Adding the memory channel helps on every seed of both categories**
  (+0.015 and +0.048 image AUROC; AU-PRO +0.034 and +0.127). It is what
  `configs/fewshot.yaml` ships.
- **The memory bank alone is better still** — decisively on `juice_bottle`
  (0.796 vs 0.726, 5.7σ of seed noise), within noise on `pushpins`. On frozen
  pretrained features, the JEPA prediction error is currently the weaker
  signal, and adding it to the memory bank dilutes it.
- The shipped model keeps the JEPA term because the brief was a JEPA-based
  model. `anomaly.memory: only` is a one-line change for whoever prefers the
  higher number; it then no longer uses the predictor at all.

## The operating point is too conservative

Every few-shot arm flags few anomalies at the default 3σ threshold:

| model | TPR at threshold | FPR at threshold |
|---|---:|---:|
| from scratch, all images (`pushpins` / `juice_bottle`) | 0.22–0.30 / 0.23–0.39 | 0.11–0.12 / 0.02–0.03 |
| DINO 16-shot, JEPA + memory | 0.02–0.08 / 0.02–0.09 | 0.01–0.02 / 0.01–0.02 |

The scores of unseen normal images have a heavy tail — a few normals are far
from all 16 training images — so mean + 3σ lands above most anomalies. The
ranking is unaffected (AUROC above), but in the demo a correctly ranked
anomaly is often still called normal: on `juice_bottle` seed 0, a normal image
scores 10.7, a logical and a structural anomaly 40.3 and 28.9, and the
threshold is 57.0. `eval.sigma_threshold` moves the trade-off; the calibration
rule itself is shared with every other model and was not changed.

## What this means

- **Use the few-shot model when normal images are scarce and the layer probe
  is strong.** On `juice_bottle`, from 16 images, it scores above the
  full-data model (0.726 vs 0.668 ± 0.052 — a gap inside that model's seed
  spread) and localises much better (AU-PRO 0.760 vs 0.488).
- **On `pushpins`, keep the from-scratch model trained on every image.**
- **The synthetic set cannot rank pretrained models.** It saturates.
- **The open research question is the JEPA term.** On frozen features it adds
  less than an appearance memory. Adapting the encoder (`fs_dino_last2_k16`,
  defined but not run) or predicting an intermediate layer are the next
  things to try.

## Not done

- The k-curve (4 / 8 / 32 images), `fs_dino_all` and `fs_dino_last2_k16`
  arms are defined in Study 10 but were not run on real data; the three other
  real categories were only probed, not trained.
- PatchCore was not re-run on the pretrained features.
- Trained at 256 px; the probe suggests resolution is what limits `pushpins`.

## Reproduce

```bash
py -3.12 scripts/fetch_pretrained.py
# Study 10 (training), per category; chunk with --arms to stay under 2 h
py -3.12 run_ablations.py --study fewshot --seeds 0,1,2 \
  --arms fs_scratch_all,fs_scratch_k16,fs_dino_k16 --category pushpins \
  --set experiment.name=fewshot data.root=data/mvtec_loco_256 data.num_workers=0
# Study 11 (inference only, re-scores the fs_dino_k16 checkpoints)
py -3.12 run_ablations.py --study memory --seeds 0,1,2 --category pushpins \
  --set experiment.name=fewshot data.root=data/mvtec_loco_256 data.num_workers=0
# Diagnostics
py -3.12 scripts/diagnose_fewshot.py layers --category pushpins --k 16
py -3.12 scripts/diagnose_fewshot.py stats  --checkpoint checkpoints/fewshot/fs_dino_k16/seed_0/pushpins/final.pt
py -3.12 scripts/diagnose_fewshot.py memory --checkpoint checkpoints/fewshot/fs_dino_k16/seed_0/pushpins/final.pt
```

Use `data.num_workers=0` on Windows with limited RAM: each worker is a full
process, and a worker failing to start ("paging file is too small") killed
the first attempt at this study. Synthetic: `--category screw_board` without
`data.root`. ~10 min per from-scratch run and ~11 min per DINO run on real
data (RTX 4060 Laptop).
