# Reference comparison: one JEPA for any product

**Status: built, trained, calibrated and deployed in the web UI.** Given one or
more photos of a correct part (the *reference*) and a photo of the part to check
(the *current* image), the model reports where and how much the current part
differs. It is trained once, on 18 products, and needs no training on the
product being inspected. It ranks anomalies well on unseen products whose
normal appearance a few photos can describe (cashew 0.92, candle 0.79–0.81,
juice bottle 0.76 image AUROC), and fails on products whose good parts vary in
layout (capsules 0.57, pushpins 0.55). Its yes/no verdict on an unseen product
needs several reference photos and still varies by product.

## What it is

- **Model** (`models/reference_jepa.py`): a frozen DINO ViT-S/16 encoder and a
  3.46M-parameter predictor. The predictor sees three groups of tokens -- the
  current image's visible context, every token of the reference image(s), and a
  mask query per hidden patch -- each tagged with a learned segment embedding.
  It predicts what a hidden region of the current image should contain, given
  both its own context and the reference.
- **Training** (`train_reference.py`): pairs of *different* normal images of the
  same product, each independently jittered (±4% shift, ±3° rotation, ±5%
  scale, lighting), drawn uniformly from 18 products: the 5 MVTec LOCO AD
  categories, the 12 VisA categories (CC BY 4.0, official split, converted by
  `scripts/prepare_visa.py`) and the synthetic board. 20% of pairs use the same
  image twice. 30 epochs × 4,096 pairs, 7,680 steps, 53 min on an RTX 4060 Laptop.
- **Scoring** (`anomaly/reference_scoring.py`):
  1. align the current photo to the reference with ECC (`anomaly/alignment.py`),
     rejecting implausible warps and filling the empty border from the reference;
  2. two channels on the aligned image: the JEPA prediction error (one 4×4 sweep
     scale, 49 masked passes) and the distance of each patch to its nearest
     reference patch (memory);
  3. subtract a per-position normal baseline measured on perturbed copies of the
     reference (and on the other references, when several are given);
  4. put both channels in common units with scales fitted on training products,
     add them (weight 1 each), upsample, smooth, take the top-1% mean;
  5. divide by the reference's own normal level (the *ratio rule*) and compare
     with a threshold calibrated on training products.

## Is the JEPA channel worth it? (weight sweep on seen products)

Held-out model, its 13 training products' own test sets, one reference, 100
test images each. Image AUROC as the JEPA weight varies (memory weight 1):

| JEPA weight | 0 (memory only) | 0.25 | 0.5 | **1.0 (default)** |
|---|---:|---:|---:|---:|
| mean AUROC, 13 products | 0.733 | 0.778 | 0.789 | **0.789** |

The reference-conditioned JEPA error adds 0.056 on products of a kind it was
trained on, and on some it carries the result almost alone (screw bag 0.717 with
JEPA only against 0.472 memory only; splicing connectors 0.695 against 0.484;
macaroni2 0.727 against 0.486). The weight was chosen here, on seen products
only, and then applied unchanged to the unseen ones below.

## Unseen products (the honest test)

A second model was trained with five products held out entirely: juice bottle
and pushpins (MVTec LOCO), candle, capsules and cashew (VisA) -- chosen because
none has a near relative in training. Thresholds were calibrated on its 13
training products only. Three random reference draws per product, all test images.

| Unseen product | AUROC, 1 ref | AUROC, 3 refs | pixel AUROC (3 refs) | AU-PRO (3 refs) | TPR / FPR, 3 refs |
|---|---:|---:|---:|---:|---:|
| cashew | 0.908 ± 0.023 | **0.915 ± 0.030** | 0.938 | 0.720 | 0.51 / 0.03 |
| candle | 0.814 ± 0.031 | 0.789 ± 0.024 | 0.952 | 0.837 | 0.60 / 0.29 |
| juice bottle | 0.701 ± 0.042 | **0.762 ± 0.031** | 0.843 | 0.625 | 0.31 / 0.06 |
| capsules | 0.565 ± 0.032 | 0.573 ± 0.028 | 0.865 | 0.463 | 0.22 / 0.12 |
| pushpins | 0.491 ± 0.047 | 0.547 ± 0.012 | 0.552 | 0.421 | 0.31 / 0.24 |

On the juice bottle with three references, logical anomalies reach 0.796 and
structural 0.710 -- from a model that never saw a juice bottle. For comparison,
the few-shot model trained *on* juice bottles with 16 images scored 0.726.

**Where it fails, and why.** Capsules lie scattered at random and pushpins sit at
varying positions: no small set of reference photos shows every acceptable
arrangement, so a different arrangement reads as a difference. Single-reference
comparison suits products with one canonical appearance.

**Unseen products: memory still beats JEPA.** On four of the five, the memory
channel alone ranks better than the fused score (e.g. cashew 0.966 against 0.915
with three references). The JEPA channel helps on product kinds it was trained
on and not yet on new ones.

## The decision threshold

Products differ about fourfold in how much two good parts differ: the normal
score of a single-reference comparison, relative to the reference's own level,
has a median of 0.65 on juice bottles and 2.67 on macaroni2. Three consequences:

- A threshold pooled over all training products (95th percentile, 2.78× with
  one reference) is set by the most variable products and flagged almost nothing.
  It was replaced by the **median over products of each product's 95th
  percentile** (1.88× with one reference, 1.35× with three, for the deployed
  model) -- still fitted on training products only.
- **With one reference, verdicts on an unseen product are unreliable** (TPR
  0.00–0.04 on three products, everything flagged on capsules). One photo cannot
  tell the model how much a new product varies.
- **With three references** the normal level comes from real second instances,
  and verdicts become usable but uneven: TPR 0.22–0.60 at FPR 0.03–0.29.

The UI therefore shows the ranking-quality signal (relative score and heatmap)
prominently, accepts up to four references, and has a sensitivity control.

## Examples (deployed model, one reference)

`Downloads/Logical-JEPA_try/` holds the pairs below. Threshold 1.88×.

| Current image | Relative score | Verdict |
|---|---:|---|
| cashew, good | 1.06× | matches |
| cashew, defect a / b | 2.13× / 1.97× | different / different |
| candle, good | 1.30× | matches |
| candle, defect a / b (≈0.1% of the image) | 1.09× / 1.36× | missed / missed |
| juice bottle, good | 0.78× | matches |
| juice bottle, labels swapped | 1.89× | different |
| juice bottle, foreign object | 1.41× | missed |

## Cost

| | |
|---|---|
| trained parameters | 3.46M (predictor); the 21.6M-parameter encoder is frozen |
| training | once, 53 min, RTX 4060 Laptop; nothing per product |
| checkpoint | 100 MB (encoder included, so it works offline) |
| first comparison with a new reference | 2–4 s (encoding + normal baseline, then cached) |
| each further comparison | 0.25–0.35 s |

## Reproduce

```bash
# data: MVTec LOCO (see README) and VisA
py -3.12 scripts/prepare_visa.py --src <extracted VisA> --dst data/visa_256 \
    --split-csv <extracted VisA>/split_csv/1cls.csv
# deployed model (all 18 products) and its calibration
py -3.12 train_reference.py --config configs/reference.yaml
py -3.12 scripts/eval_reference.py --checkpoint checkpoints/reference/final.pt --calibrate --num-references 1 --products none
py -3.12 scripts/eval_reference.py --checkpoint checkpoints/reference/final.pt --calibrate --num-references 3 --products none
# held-out model and the unseen-product evaluation
py -3.12 train_reference.py --config configs/reference.yaml --set experiment.name=reference_holdout \
    "data.holdout=[juice_bottle,pushpins,candle,capsules,cashew]"
py -3.12 scripts/eval_reference.py --checkpoint checkpoints/reference_holdout/final.pt \
    --calibrate --num-references 3 --products holdout --draws 3
# web UI with both models
py -3.12 app/server.py --checkpoint checkpoints/fewshot_dino/juice_bottle/final.pt \
    --reference-checkpoint checkpoints/reference/final.pt
```

Use `data.num_workers=0` on this machine: memory is tight. Datasets: MVTec LOCO
AD (CC BY-NC-SA 4.0) and VisA (CC BY 4.0), for non-commercial research.

## Next

1. A decision rule that adapts to an unseen product's variability with fewer
   references -- e.g. ask for 3–5 good photos at setup and fit a per-product
   threshold from them.
2. Close the seen/unseen gap of the JEPA channel: more and more varied products
   (MVTec AD's 15 categories would be the next source).
3. Arrangement-tolerant comparison for products like capsules and pushpins
   (match parts, not positions).
4. Higher resolution for tiny defects such as the candle's.
