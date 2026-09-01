# Logical-JEPA

**Multi-Scale Contextual Latent Prediction for Unsupervised Logical Anomaly Detection**

A Joint-Embedding Predictive Architecture (JEPA) built entirely from scratch in PyTorch,
trained only on normal industrial images, that detects and localizes not just scratches and
dents but **logical** anomalies — a missing component, an extra one, a component in the wrong
place, the wrong number of objects, an arrangement that violates the expected structure.

---

## The idea in one paragraph

Conventional anomaly detectors ask *"does this patch look unusual?"*. Logical-JEPA asks a
different question:

> **Given everything else in this image, is this region what I expected to exist here?**

The model hides a region of the image, encodes only the remaining context, and predicts the
*latent representation* of the hidden region. A second encoder — an EMA copy that sees the
real image — reports what is actually there. When the two disagree, the region violates the
structure the model learned from normal images.

That framing is what makes logical anomalies visible. A screw photographed in the wrong
position still looks exactly like a screw, so a texture-based detector sees nothing wrong.
But a predictor conditioned on the rest of the assembly expects *empty board* at that
location, and expects *a screw* at the mounting point that is now bare. Both produce a large
latent discrepancy.

```
                          NORMAL IMAGE
                               │
                        Patch Embedding
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
          Context patches              Full image
                 │                           │
                 ▼                           ▼
         Context Encoder              Target Encoder
          Tiny ViT (6 blk)             EMA copy, frozen
                 │                           │
                Zc                          Zt
                 │                           │
                 ▼                           │
             Predictor  ◄── target positions │
                 │                           │
                 ▼                           │
                Ẑt ────────  compare  ────── Zt
                               │
                               ▼
                      1 − cos(Ẑt, Zt)
```

At training time that comparison is the **loss**. At inference time the *same* comparison is
the **anomaly score**. Nothing else changes.

---

## Results

Two result sets, kept separate because they are not interchangeable:

- **Synthetic `screw_board`** (200 normal train, 85 test), 80 epochs, **single seed** — the
  sections below. These validated the pipeline and produced the design decisions.
- **Real MVTec LOCO AD** — see [`outputs/phase1/SUMMARY.md`](outputs/phase1/SUMMARY.md) for
  the multi-seed re-run and whether the synthetic ordering reproduced.

> **The synthetic numbers below are single-seed and are labelled as such throughout.** A
> single-seed AUROC on a small test split carries enough spread to reorder ablation arms by
> itself, which is why multi-seed reporting (`--seeds 0,1,2`, `mean ± std`) was added and why
> the real-data study uses it. The synthetic anomalies are also *small* relative to the
> largest mask scale, which matters for one of the findings below.

### Logical-JEPA vs baselines

Best Logical-JEPA configuration against both baselines, all trained from scratch on the same
normal images:

| Method | Image AUROC | **Logical** | **Structural** | Pixel AUROC | AU-PRO |
|---|---|---|---|---|---|
| **Logical-JEPA** (best arm) | **0.955** | **0.972** | **0.937** | 0.884 | **0.772** |
| Autoencoder | 0.775 | 0.672 | 0.879 | **0.894** | 0.643 |
| PatchCore (from-scratch features) | 0.587 | 0.665 | 0.508 | 0.682 | 0.499 |

At its 3σ operating point Logical-JEPA reaches **accuracy 0.81, TPR 0.75, FPR 0.04**.

**The profiles are inverted, and that is the actual finding.** The autoencoder is far better
at structural anomalies than logical ones (0.879 vs 0.672 — a 0.21 gap *favouring*
structural), which is the classic signature of a texture-based detector: it is trained to
copy its input, and a correctly-shaped screw in the wrong place copies perfectly well.
Logical-JEPA reverses the ordering (0.972 logical vs 0.937 structural) and beats the
autoencoder on logical anomalies by **+0.30 AUROC**. That is the contextual-prediction
objective doing exactly what it was chosen to do.

PatchCore is weak here for the reason its module documents: a memory bank of patch features
has no notion of *where* a patch was or *how many* like it there were. Note its absolute
numbers are not comparable to published PatchCore results — see [Baselines](#baselines).

### Study 1 · Masking strategy — RETRACTED on real data

> ⚠️ **This synthetic result did not reproduce.** It is kept here because the
> retraction is the more useful finding. See
> [`outputs/phase1/SUMMARY.md`](outputs/phase1/SUMMARY.md).

Synthetic `screw_board`, **single seed**:

| Arm | Image AUROC | Logical | Structural | Pixel AUROC | AU-PRO |
|---|---|---|---|---|---|
| `random_patch` (scattered, control) | **0.955** | **0.972** | **0.937** | 0.884 | **0.772** |
| `small` (2×2–3×3) | 0.927 | 0.943 | 0.911 | **0.898** | 0.697 |
| `large` (5×5–8×8) | 0.838 | 0.807 | 0.869 | 0.589 | 0.548 |
| `multiscale` (proposed) | 0.762 | 0.783 | 0.741 | 0.615 | 0.497 |
| `multiscale` + curriculum | 0.805 | 0.764 | 0.845 | 0.502 | 0.501 |

On this data the ordering looked decisive — a 0.193 spread, with multi-scale worst — and the
project previously concluded that "finer masking wins" and "multi-scale masking is
unsupported".

Real MVTec LOCO `pushpins`, 80 epochs, **3 seeds**, `mean ± std`:

| Arm | Image AUROC | Logical | Structural |
|---|---|---|---|
| `random_patch` | 0.632 ± 0.007 | 0.449 ± 0.003 | 0.837 ± 0.012 |
| `small` | 0.638 ± 0.003 | 0.448 ± 0.001 | 0.851 ± 0.005 |
| `large` | 0.638 ± 0.004 | 0.454 ± 0.004 | 0.845 ± 0.006 |
| `multiscale` | 0.641 ± 0.005 | 0.451 ± 0.003 | 0.855 ± 0.008 |
| `multiscale` + curriculum | **0.645 ± 0.007** | 0.453 ± 0.005 | **0.860 ± 0.010** |

**Every arm falls within 0.013 image AUROC**, against a seed σ of ~0.005 — the runner's own
significance check reports the top two as `indistinguishable (0.6σ)`. The ordering is even
mildly *reversed*, with `multiscale + curriculum` leading.

| | synthetic spread | real spread |
|---|---:|---:|
| best − worst image AUROC | **0.193** | **0.013** |

So the correct statement is not "multi-scale masking failed" but **"the masking strategy has
no measurable effect on real data, and the synthetic ordering was an artifact of one seed on
synthetic images."** This is exactly the failure mode multi-seed reporting was added to
catch.

*What did survive:* large-block masking still localizes worse than small-block masking on
synthetic data (pixel AUROC 0.589 vs 0.898) — an 8×8 target yields one error value for a
128×128 region, so coarse localization was the correct prediction.

### Study 1b · Logical anomalies score BELOW chance — the real finding

Look again at the logical column above: **0.448–0.454 across every arm**, with seed σ of
0.001–0.005 over 15 training runs. Below chance, reproducibly. Structural over the same runs
is 0.837–0.860, so the model works — the *score* is inverted for one anomaly family.

That logical AUROC pins to ~0.45 regardless of how the model was trained means the cause is in
**scoring, not training**.

**The mechanism.** A missing pushpin leaves an empty compartment, and an empty compartment is
*easier* to predict than the object that belongs there. The score counts only "harder than
normal", so a missing object registers as extra-normal and is pushed *down* the ranking. The
synthetic data never exposed this because its logical anomalies mostly *added* structure.

`anomaly.deviation: absolute` scores `|z|`, so an unexpectedly easy region counts as
surprising too. Added as Study 7 with tests reproducing both the failure and the fix.

### Study 2 · Anomaly score function — cosine wins clearly

| Distance | Image AUROC | Logical | Structural | AU-PRO |
|---|---|---|---|---|
| **cosine** (Method B) | **0.762** | **0.783** | **0.741** | **0.497** |
| combined 0.7·cos + 0.3·L2 (Method C) | 0.665 | 0.728 | 0.603 | 0.344 |
| combined 0.5/0.5 | 0.649 | 0.723 | 0.575 | 0.315 |
| L2 (Method A) | 0.618 | 0.681 | 0.555 | 0.269 |
| smooth-L1 | 0.617 | 0.685 | 0.549 | 0.271 |

Cosine beats L2 by **+0.14 image AUROC and +0.23 AU-PRO**, and mixing in any L2 monotonically
hurts. Scale-invariance is doing real work: the teacher's embedding norm tracks local
contrast, so L2 partly measures "how much texture is here" rather than "is the right thing
here". This is a confirmation of the default, not a tuning artifact.

### Study 3 · Inference sweep — the scale mix barely matters

| Sweep | Image AUROC | Logical | Structural | AU-PRO |
|---|---|---|---|---|
| multi (2,4,6) weighted toward large | **0.767** | 0.781 | **0.752** | **0.504** |
| multi (2,4,6), mean fusion | 0.762 | **0.783** | 0.741 | 0.497 |
| large only (6×6) | 0.757 | 0.761 | 0.753 | 0.511 |
| multi, max fusion | 0.756 | 0.763 | 0.749 | 0.482 |
| small only (2×2) | 0.751 | 0.779 | 0.723 | 0.481 |

All five sit within 0.016 AUROC. On this data the *inference* window scale is close to
irrelevant — a useful negative result, since sweeping three scales costs 299 forward
configurations per image versus 225 for one. If throughput matters, drop to a single scale.

### Study 4 · Grid normalization — the strongest effect measured

| Normalization | Image AUROC | Logical | Structural | Pixel AUROC | AU-PRO |
|---|---|---|---|---|---|
| **per-position (proposed)** | **0.762** | **0.783** | **0.741** | 0.615 | 0.497 |
| per-image median/MAD | 0.534 | 0.521 | 0.547 | 0.871 | 0.640 |
| none | 0.533 | 0.544 | 0.521 | 0.866 | 0.624 |
| per-image z-score | 0.461 | 0.603 | 0.319 | **0.876** | **0.655** |

This is the single largest effect in the whole study, and it is a **genuine trade-off, not a
free win**.

Prediction difficulty is strongly position-dependent even on flawless images — object
boundaries and high-frequency regions are always harder to predict than flat background — and
that positional baseline is *larger than the anomaly signal*. Dividing it out using
per-position statistics fitted on normal images turns the question from *"is this region hard
to predict?"* into *"is this region harder than it normally is **here**?"*, and image AUROC
jumps from 0.53 to 0.76.

But it costs localization: pixel AUROC drops from 0.876 to 0.615, because dividing by a small
per-position standard deviation amplifies noise in easy regions. Per-image z-scoring is the
mirror image — the sharpest heatmaps in the study and *below-chance* image detection (0.461),
because forcing every map to mean 0 and std 1 erases exactly the between-image difference an
image-level score is made of.

**Detection and localization want different normalizations.** `anomaly.normalize` exposes the
choice; pick `global` to decide whether an image is defective, `zscore` to show an operator
where.

Reproduce all of it:

```bash
py -3.12 run_ablations.py --config configs/ablations.yaml --study all
```

---

## Quick start

Requires **Python 3.12** installed globally. This project deliberately uses the global
interpreter — do not create a virtual environment.

```bash
# 1. install dependencies into the global Python 3.12
py -3.12 -m pip install -r requirements.txt

# 2. generate the synthetic dataset (or install real MVTec LOCO AD, see below)
py -3.12 scripts/download_data.py --synthetic

# 3. train on normal images only  (~4 min on an RTX 4060)
py -3.12 train.py --config configs/loco.yaml

# 4. evaluate: AUROC / AU-PRO, split by logical vs structural
py -3.12 evaluate.py --checkpoint checkpoints/logical_jepa_multiscale/screw_board/final.pt

# 5. launch the web demo
py -3.12 app/server.py --checkpoint checkpoints/logical_jepa_multiscale/screw_board/final.pt
#    -> http://127.0.0.1:5000
```

To train the best measured configuration instead of the strategy under study:

```bash
py -3.12 train.py --config configs/loco.yaml --set masking.strategy=random_patch
```

For a CUDA build of PyTorch:

```bash
py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

## Architecture

Roughly **6.4M trainable parameters** — small on purpose. The research contribution is the
masking strategy and the scoring procedure, not scale.

| Component | Configuration |
|---|---|
| Patch embedding | 256×256 input, 16×16 patches → **16×16 = 256 tokens**, dim 256 |
| Positional encoding | fixed 2-D sin-cos (not learned) |
| Context encoder | 6 transformer blocks, 8 heads, MLP ratio 4 — sees **visible patches only** |
| Target encoder | identical architecture, **EMA-updated**, no gradient, always sees the full image |
| Predictor | 3 blocks, width 192, one shared mask token + target positions |
| Loss | `1 − cos(Ẑt, Zt)`, teacher outputs LayerNorm'd |
| EMA momentum | 0.996 → 1.0, cosine-annealed |

Everything — multi-head attention, the MLP block, stochastic depth, the ViT initialization,
the 2-D sin-cos position table, the EMA mechanism, the coreset selection in the PatchCore
baseline — is implemented in this repository. **No ImageNet backbone, no pretrained I-JEPA,
DINO, CLIP or ViT weights.** Every model starts from random initialization.

Two design choices are load-bearing and worth calling out:

- **Masked patches are removed from the encoder's sequence entirely**, not replaced with a
  mask token. This forces the *predictor*, not the encoder, to do the imagining.
- **Positional encodings are fixed rather than learned.** At inference the sweep asks for
  predictions at positions that were never targets during any given training step, and a
  fixed table extrapolates consistently.

Watch two numbers during training: `cos_sim` should climb towards 1 (it reaches ~0.98), and
`target_std` should stay well above zero (~0.50). If `target_std` collapses towards 0 the
encoder has found the trivial constant solution — training warns about this explicitly.

---

## Masking strategies

Vanilla I-JEPA samples target blocks from a single scale range. The premise of this project
was that the two failure modes live at *different* scales:

- **structural** defects (scratch, chip, contamination) are a few patches wide and are only
  visible when the target window is comparably small;
- **logical** violations only become visible when the window is large enough to contain a
  whole component, so the predictor must reason about scene composition rather than texture.

`MultiScaleMask` interleaves both — and in `mixed` mode, within a single gradient step. As
Study 1 above reports, that premise was **not** supported on the synthetic data; the study is
implemented so it can be settled on the real benchmark.

| Strategy | Blocks | Sizes | Masked area | Intent |
|---|---|---|---|---|
| `random_patch` | scattered | 1×1 | 18.8% | MAE-style control — **best measured** |
| `small` | 6 | 2×2 – 3×3 | 14.5% | structural anomalies |
| `large` | 2 | 5×5 – 8×8 | 27.4% | logical anomalies |
| `multiscale` | 3 small + 1 large | both | 21.1% | proposed |

Three multi-scale sampling policies: `mixed` (both scales every step, the default),
`alternate` (one scale per step), and `curriculum` (small-heavy early, large-heavy late).

---

## Anomaly detection at inference

Training masks are random; evaluation masks are **exhaustive**. A window slides over the
whole patch grid so that every patch eventually becomes a target while everything else serves
as context:

```
Mask 1        Mask 2        Mask 3       ...
████░░░░      ░░████░░      ░░░░████
████░░░░      ░░████░░      ░░░░████
░░░░░░░░      ░░░░░░░░      ░░░░░░░░
░░░░░░░░      ░░░░░░░░      ░░░░░░░░
```

For each position `r`:  `A_r = 1 − cos(Ẑ_r, Z_r)`.

```
Input image → sweep at 3 scales (299 configs) → per-scale 16×16 error grids
           → per-position normalization (fitted on normals)
           → fuse scales → 16×16 grid → bilinear + Gaussian → 256×256 heatmap
           → top-1% mean → image score → threshold → NORMAL / ANOMALOUS
```

Sweeping *every* region is what makes visible anomalies detectable at all. If an extra or
misplaced component stays inside the context, the predictor simply copies it and the error
vanishes; only when that region becomes the target is the model forced to imagine what
should have been there.

Measured throughput: **258 ms per image** at batch 1, 242 ms at batch 4 (RTX 4060, 299 sweep
configurations per image).

**Calibration is unsupervised.** The decision threshold is `mean + 3σ` of the score over
held-out *normal* images; the per-position statistics come from the normal training set. No
anomalous image ever informs the threshold.

---

## Dataset

### MVTec LOCO AD (required for reported results)

3,651 images across five categories, designed specifically to contain both structural and
logical anomalies. Free for non-commercial research.

```bash
py -3.12 scripts/download_data.py            # download instructions
py -3.12 scripts/verify_dataset.py --root data/mvtec_loco_real --strict-loco
```

`verify_dataset.py` is the strict check: it fails loudly on a missing split, an anomalous
image with no ground-truth directory, ground truth stored as a flat file instead of a
directory, an all-zero mask union, or **any anomaly found inside `train/` or `validation/`**
— the last being the one that would silently invalidate the unsupervised claim.

**Pre-resize before training.** LOCO ships images up to 1700×1000, and decoding them costs
**84 ms each**, which makes training CPU-bound rather than GPU-bound:

```bash
py -3.12 scripts/prepare_loco.py --src data/mvtec_loco_real --dst data/mvtec_loco_256
```

This drops loading to **7.2 ms/image (11.7× faster)** and the dataset to 476 MB. It is not an
approximation — the loader would have produced the same 256×256 tensor anyway. Masks are
resampled with NEAREST, which reproduces LOCO's original label values exactly (they are *not*
`{0,255}`; they carry values in the 234–255 range, and the loader thresholds at `> 0`).

**Why the union of region masks matters.** Measured across the real dataset:

| | images |
|---|---:|
| 1 region | 906 |
| 2 regions | 65 |
| 3 regions | 10 |
| **15 regions** | 12 |

**8.8% of annotated images carry multiple regions — and 50 of 91 `pushpins` logical
anomalies do, up to 15 each.** A loader reading only the first mask file would silently lose
over half the annotation on that category, with no error and no obviously wrong output. This
is why `tests/test_union_mask.py` exists and why it is the most heavily tested path in the
repository.

Expected layout:

```
data/mvtec_loco/
  breakfast_box/
    train/good/000.png                       ← normal images ONLY
    validation/good/000.png                  ← normal, held out for calibration
    test/good/000.png
    test/logical_anomalies/000.png
    test/structural_anomalies/000.png
    ground_truth/logical_anomalies/000/000.png    ← a DIRECTORY of region masks
    ground_truth/structural_anomalies/000/000.png
  juice_bottle/  pushpins/  screw_bag/  splicing_connectors/
```

Two details of this benchmark drive the loader: anomaly *type* is a directory name (so every
sample carries `defect_type` and the two families are reported separately), and ground truth
is a **directory of region masks per image**, not a single file — a logical anomaly like "two
pushpins in one compartment" is split across several files that must be unioned.

### Synthetic stand-in (for development)

```bash
py -3.12 scripts/download_data.py --synthetic
```

Generates a `screw_board` category with the same directory layout, the same
logical/structural taxonomy and the same ground-truth format. Normal images have four screws
at fixed mounting points, a product body and a colour-coded indicator strip. Anomalies:

- **logical** — `missing_screw`, `extra_screw`, `misplaced_screw`, `wrong_count`, `swapped_indicator`
- **structural** — `scratch`, `contamination`, `chip`

None of the logical anomalies change local texture: every pixel is still a legitimate board
or screw pixel, and only the arrangement is invalid. That is precisely the regime this
project targets.

---

## Tests

```bash
py -3.12 -m pytest tests/ -q          # 245 tests, ~14 s
```

| file | tests | what it guards |
|---|---:|---|
| `test_masking.py` | 42 | context/target disjointness, sweep coverage, scale semantics |
| `test_scoring.py` | 34 | normalization trade-off, **calibration sees no anomaly** |
| `test_model.py` | 33 | **no pretrained weights**, teacher detachment, EMA arithmetic |
| `test_aggregate.py` | 27 | multi-seed stats, refusing to rank inside the noise |
| `test_vicreg.py` | 26 | both collapse modes, the vanishing-gradient limit |
| `test_dataset.py` | 23 | **train/validation stay anomaly-free**, transform policy |
| `test_metrics.py` | 17 | AUROC endpoints, AU-PRO region weighting |
| `test_union_mask.py` | 16 | **union of region masks** — the highest-risk data path |
| `test_loss_reduction.py` | 14 | per-patch vs per-block arithmetic |
| `test_decoupled_norm.py` | 13 | detection and localization paths stay separate |

The bolded rows encode project *constraints* rather than ordinary correctness. They exist so
a future change cannot quietly invalidate a research claim — a leaked anomaly in the
calibration set or a pretrained backbone would not otherwise raise an error, it would just
make every reported number optimistic.

## Repository layout

```
logical-JEPA/
├── configs/
│   ├── loco.yaml              # default configuration
│   └── ablations.yaml         # the four ablation studies
├── models/
│   ├── patch_embed.py         # patchify + fixed 2-D sin-cos positions
│   ├── transformer.py         # MHSA, MLP, DropPath, ViT init — from scratch
│   ├── context_encoder.py     # student: visible patches only
│   ├── target_encoder.py      # teacher: EMA, frozen, sees everything
│   ├── predictor.py           # narrow transformer, imagines hidden regions
│   └── logical_jepa.py        # training step + masked inference sweep
├── masking/
│   ├── base.py                # MaskSpec, block sampling, context complement
│   ├── small_mask.py          # 2×2–3×3   (ablation arm 2)
│   ├── large_mask.py          # 5×5–8×8   (ablation arm 3)
│   ├── multiscale_mask.py     # proposed  (ablation arm 4)
│   ├── random_patch_mask.py   # scattered (ablation arm 1, control)
│   └── sweep_mask.py          # exhaustive inference-time sweep
├── anomaly/
│   ├── embedding_error.py     # distances + JEPA loss (one definition, two uses)
│   ├── anomaly_map.py         # normalize → fuse scales → upsample → smooth
│   ├── scoring.py             # image score, unsupervised calibration
│   └── metrics.py             # AUROC, pixel AUROC, AU-PRO, logical/structural split
├── datasets/
│   ├── mvtec_loco.py          # the real benchmark
│   ├── synthetic_loco.py      # LOCO-shaped stand-in generator
│   └── transforms.py          # deliberately conservative augmentation
├── baselines/
│   ├── autoencoder.py         # reconstruction error
│   └── patchcore.py           # coreset memory bank + kNN
├── app/
│   ├── server.py              # Flask inference API
│   └── static/                # HTML + CSS + JS frontend
├── utils/                     # config, logging, seeding
├── scripts/download_data.py
├── train.py  evaluate.py  visualize.py  run_ablations.py
└── requirements.txt
```

---

## Usage

### Training

```bash
py -3.12 train.py --config configs/loco.yaml
py -3.12 train.py --config configs/loco.yaml --category juice_bottle
py -3.12 train.py --config configs/loco.yaml --set masking.strategy=large train.epochs=200
```

Any config key can be overridden with dotted `--set key=value` arguments.

### Evaluation

```bash
py -3.12 evaluate.py --checkpoint checkpoints/<name>/<category>/final.pt
py -3.12 evaluate.py --checkpoint <path> --set anomaly.distance=l2 anomaly.fusion=max
```

Writes `eval_results.json` plus a `calibration.json` beside the checkpoint, which the web
demo and `visualize.py` both pick up.

### Visualization

```bash
py -3.12 visualize.py --checkpoint <path> --mode all
py -3.12 visualize.py --checkpoint <path> --mode latent --defect logical_anomalies
```

- `grid` — input, heatmap, overlay, ground truth, and one column per sweep scale
- `masks` — what each masking strategy actually hides, drawn on a real image
- `latent` — predicted vs observed embedding for the peak patch of the most surprising
  region, with its cosine distance; the figure that makes "high latent discrepancy = logical
  anomaly" concrete rather than asserted

### Ablations

```bash
py -3.12 run_ablations.py --study all                          # everything
py -3.12 run_ablations.py --study masking                      # trains 5 models
py -3.12 run_ablations.py --study scoring --checkpoint <path>  # inference only
```

| Study | Question | Cost |
|---|---|---|
| 1 · masking | Does multi-scale masking beat single-scale? | 5 training runs |
| 2 · scoring | cosine vs L2 vs combined (Methods A/B/C) | inference only |
| 3 · sweep | Does the *inference* window scale matter independently? | inference only |
| 4 · normalization | Per-position vs per-image normalization | inference only |

Results land in `outputs/<experiment>/<category>/` as per-study CSVs plus a JSON summary.
The whole suite takes about 40 minutes on an RTX 4060.

### Web demo

```bash
py -3.12 app/server.py --checkpoint <path> --port 5000
```

Drag in an image (or paste from the clipboard, or click a test sample) and the page shows
the verdict, the anomaly score against its calibrated threshold, the localization heatmap,
and **the per-scale evidence**, so you can see *which kind* of anomaly fired.

---

## Baselines

Both are trained from scratch alongside the main model.

**Autoencoder** — the classical detector. It fails on logical anomalies for a specific,
predictable reason: an autoencoder is trained to *copy its input*, and convolutions are
local, so a plausible-looking screw in the wrong place is reconstructed perfectly well. The
network never has to know where screws belong. The measured profile (structural 0.879,
logical 0.672) is exactly that failure mode.

**PatchCore** — a memory bank of normal patch features with greedy coreset subsampling and
kNN scoring. A memory bank has no notion of *where* a patch was or *how many* like it there
were, so a correctly-manufactured component in the wrong position produces a feature that is
already in the bank.

> **One deliberate deviation.** The original PatchCore uses an ImageNet-pretrained
> WideResNet-50. This project forbids pretrained backbones, so features come from the same
> from-scratch target encoder the JEPA model trained. That is the fair comparison for *this*
> research question — both methods then see identical features and differ only in how they
> use them, memory lookup versus contextual prediction — but it means the absolute numbers
> are **not comparable to published PatchCore results**, and PatchCore's weak showing here
> should not be read as a criticism of the published method.

---

## Research questions

**1. Can latent-space contextual prediction detect logical industrial anomalies more
effectively than conventional local feature-based detectors?**

*Supported on this data.* Logical-JEPA scores 0.972 on logical anomalies against the
autoencoder's 0.672, and the two methods have inverted profiles: the autoencoder is 0.21
better on structural than logical, Logical-JEPA is slightly better on logical than
structural. That inversion is the qualitative signature the objective was chosen for.

**2. Does multi-scale semantic masking improve the ability to distinguish structural from
logical anomalies?**

*Not supported on this data.* Multi-scale masking was the weakest of the five arms
(0.762 vs 0.955 for scattered-patch masking). Study 1 above sets out the two candidate
explanations — patch-count loss weighting, and anomaly scale relative to block size — and
both are directly testable. A negative result is still a result, and the honest statement is
that the second research question remains open pending the real benchmark.

---

## Notes and limitations

- **These numbers come from a synthetic development dataset**, single seed, 80 epochs per
  arm. MVTec LOCO test splits are small and AUROC is correspondingly noisy; report a mean
  over several seeds for anything conclusive.
- **Resolution is the binding constraint on structural anomalies.** A 16×16 grid over a
  256×256 image gives each token a 16×16 pixel receptive field, so defects much smaller than
  that are averaged away. Raising `data.img_size` to 512 (a 32×32 grid) is the first thing to
  try for texture-scale defects, at roughly 4× the sweep cost.
- **The sweep is the expensive part of inference**, not the model. Study 3 found the scale
  mix barely matters, so dropping to a single sweep scale is close to free.
- **Detection and localization want different normalizations** (Study 4). There is no single
  setting that is best at both.
- **Augmentation is deliberately minimal.** Cropping, flipping, rotation and cutout would all
  teach the model that moved or missing content is normal, destroying exactly the signal
  being detected. Only mild photometric jitter is on by default.

---

## References

- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive
  Architecture (I-JEPA)*, CVPR 2023 — the masking and EMA-teacher mechanism this work builds on.
- Bergmann et al., *Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly
  Detection and Localization*, IJCV 2022 — the MVTec LOCO AD benchmark and the AU-PRO metric.
- Roth et al., *Towards Total Recall in Industrial Anomaly Detection (PatchCore)*, CVPR 2022 —
  the coreset memory-bank baseline.

---

## License

Released for academic and educational use. MVTec LOCO AD carries its own license; review
MVTec's terms before redistributing any data or derived results.
