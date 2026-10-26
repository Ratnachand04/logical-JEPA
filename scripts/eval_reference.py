"""Calibrate and evaluate the reference-conditioned model.

Calibration (``--calibrate``) fits the global constants the scorer needs -- the
scale of each channel's residual and the decision threshold -- on *training*
products only: one reference from ``train/good`` is compared with that
product's ``validation/good`` images. The constants are written next to the
checkpoint (``reference_calibration.json``).

Evaluation compares every test image of each product with ``--num-references``
reference images drawn from that product's ``train/good``, for several random
draws, and reports image AUROC for the fused score and for each channel alone,
pixel AUROC and AU-PRO (ground truth warped with the same alignment), and the
detection and false-alarm rates at the calibrated threshold.

    py -3.12 scripts/eval_reference.py --checkpoint checkpoints/reference/final.pt --calibrate
    py -3.12 scripts/eval_reference.py --checkpoint checkpoints/reference_holdout/final.pt \
        --products holdout --draws 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from anomaly.alignment import warp_mask  # noqa: E402
from anomaly.metrics import compute_pro, image_auroc, pixel_auroc  # noqa: E402
from anomaly.reference_scoring import ReferenceScorer, load_calibration  # noqa: E402
from datasets.mvtec_loco import MVTecLOCO, load_union_mask  # noqa: E402
from datasets.pairs import PairSource, resolve_sources  # noqa: E402
from train_reference import load_reference_model  # noqa: E402
from utils.logging_utils import save_json  # noqa: E402


JEPA_WEIGHTS = (0.0, 0.25, 0.5, 1.0)     # memory weight fixed at 1; reported side by side


def _normals(src: PairSource, split: str) -> list[str]:
    from datasets.mvtec_loco import DEFECT_GOOD, _list_images
    return _list_images(os.path.join(src.root, src.category, split, DEFECT_GOOD))


def build_scorer(model, cfg, device, calibration=None) -> ReferenceScorer:
    node = cfg.get("scoring", {})
    return ReferenceScorer(model, device, windows=tuple(node.get("windows", [4])),
                           sigma=node.get("smooth_sigma", 4.0),
                           top_k_ratio=node.get("top_k_ratio", 0.01),
                           n_pseudo=node.get("n_pseudo", 8), align=node.get("align", True),
                           calibration=calibration)


# --------------------------------------------------------------------------- #
def calibrate(model, cfg, device, products: list[PairSource], per_product: int,
              target_fpr: float, seed: int, rule: str = "ratio", num_references: int = 1,
              jepa_weight: float = 1.0, existing: dict | None = None) -> dict:
    """Fit channel scales and both rules' thresholds on training products' normals,
    for one reference count. ``existing`` keeps thresholds fitted for other counts."""
    scorer = build_scorer(model, cfg, device)
    rng = random.Random(seed)
    residuals = []                                   # (product, res, state) for normal pairs
    for src in products:
        train, val = _normals(src, "train"), _normals(src, "validation")
        if not train or not val:
            continue
        refs = rng.sample(train, min(num_references, len(train)))
        state = scorer.prepare([Image.open(p) for p in refs])
        for path in rng.sample(val, min(per_product, len(val))):
            res = scorer.residuals(Image.open(path), state)
            residuals.append((src.name, res, state))
    if not residuals:
        raise SystemExit("no validation normals found for calibration")

    # Channel scales: spread of the residual over all positions of all normal pairs.
    jepa = torch.cat([r["jepa"].flatten() for _, r, _ in residuals])
    memory = torch.cat([r["memory"].flatten() for _, r, _ in residuals])
    calib = {"jepa_scale": float(jepa.std()), "memory_scale": float(memory.std()),
             "jepa_weight": jepa_weight, "memory_weight": 1.0}
    if existing and existing.get("by_references"):
        # Channel scales are shared across reference counts; keep the first fit.
        calib["jepa_scale"] = existing["jepa_scale"]
        calib["memory_scale"] = existing["memory_scale"]
    scorer.calibration.update(calib)
    scores = np.array([scorer.image_score(scorer.to_map(scorer.fuse(r), r["valid"]))
                       for _, r, _ in residuals])
    ratios = np.array([sc / scorer.normal_level(st) for sc, (_, _, st) in zip(scores, residuals)])
    names = np.array([name for name, _, _ in residuals])

    def typical(values):
        # Each product's own (1 - target_fpr) quantile, then the median across
        # products: a threshold for a typical product, not one set by the few
        # products whose good parts vary the most.
        per = [np.quantile(values[names == n], 1.0 - target_fpr) for n in sorted(set(names))]
        return float(np.median(per))

    thresholds = {"absolute": typical(scores), "ratio": typical(ratios)}
    pooled = {"absolute": float(np.quantile(scores, 1.0 - target_fpr)),
              "ratio": float(np.quantile(ratios, 1.0 - target_fpr))}
    by_refs = dict((existing or {}).get("by_references") or {})
    by_refs[str(num_references)] = thresholds
    pooled_by_refs = dict((existing or {}).get("pooled_by_references") or {})
    pooled_by_refs[str(num_references)] = pooled
    calib.update(rule=rule, threshold=by_refs.get("1", thresholds)[rule], thresholds=thresholds,
                 by_references=by_refs, pooled_by_references=pooled_by_refs,
                 threshold_method="median over products of each product's quantile",
                 target_fpr=target_fpr, normal_score_mean=float(scores.mean()),
                 normal_score_std=float(scores.std()), normal_ratio_mean=float(ratios.mean()),
                 fitted_on=sorted({name for name, _, _ in residuals}), n_pairs=len(residuals))
    return calib


# --------------------------------------------------------------------------- #
def evaluate_product(scorer: ReferenceScorer, src: PairSource, num_refs: int, draws: int,
                     seed: int, max_test: int | None) -> dict:
    test = MVTecLOCO(src.root, src.category, "test", 256)
    samples = test.samples
    if max_test and len(samples) > max_test:
        samples = random.Random(seed).sample(samples, max_test)
    train = _normals(src, "train")
    per_draw = []
    for draw in range(draws):
        rng = random.Random(seed * 1000 + draw)
        refs = [Image.open(p) for p in rng.sample(train, num_refs)]
        state = scorer.prepare(refs)
        level = scorer.normal_level(state)
        c = scorer.calibration
        weighted = {w: [] for w in JEPA_WEIGHTS}
        fused, jepa_only, memory_only, labels, kinds, maps, masks = [], [], [], [], [], [], []
        for s in samples:
            res = scorer.residuals(Image.open(s.image_path), state)
            amap = scorer.to_map(scorer.fuse(res), res["valid"])
            fused.append(scorer.image_score(amap))
            jepa_only.append(scorer.image_score(
                scorer.to_map(scorer.fuse(res, ("jepa",)), res["valid"])))
            memory_only.append(scorer.image_score(
                scorer.to_map(scorer.fuse(res, ("memory",)), res["valid"])))
            for w in JEPA_WEIGHTS:
                grid = w * res["jepa"] / c["jepa_scale"] + res["memory"] / c["memory_scale"]
                weighted[w].append(scorer.image_score(scorer.to_map(grid, res["valid"])))
            labels.append(s.label)
            kinds.append(s.defect_type)
            if s.label == 1 and s.mask_dir:
                gt = np.asarray(load_union_mask(s.mask_dir, (256, 256)).resize(
                    (256, 256), Image.NEAREST)) > 0
                gt = warp_mask(gt.astype(np.uint8), res["align"]) > 0
                maps.append(amap[0, 0].cpu().numpy())
                masks.append(gt)
        labels = np.array(labels)
        fused = np.array(fused)
        image_scores = {"score": [float(v) for v in fused], "label": [int(v) for v in labels],
                        "kind": list(kinds)}
        kinds = np.array(kinds)
        row = {
            "image_auroc": image_auroc(fused, labels),
            "jepa_only_auroc": image_auroc(np.array(jepa_only), labels),
            "memory_only_auroc": image_auroc(np.array(memory_only), labels),
            "normal_level": level,
            "images": image_scores,
        }
        for w in JEPA_WEIGHTS:
            row[f"auroc_jepa_w{w}"] = image_auroc(np.array(weighted[w]), labels)
        by_refs = scorer.calibration.get("by_references") or {}
        usable = sorted(int(k) for k in by_refs if int(k) <= num_refs)
        thresholds = by_refs[str(usable[-1])] if usable else scorer.calibration.get("thresholds", {})
        for rule, values in (("absolute", fused), ("ratio", fused / level)):
            thr = thresholds.get(rule, float("inf"))
            row[f"tpr_{rule}"] = float((values[labels == 1] > thr).mean()) if (labels == 1).any() else float("nan")
            row[f"fpr_{rule}"] = float((values[labels == 0] > thr).mean()) if (labels == 0).any() else float("nan")
        for kind in sorted(set(kinds) - {"good"}):
            sel = (kinds == "good") | (kinds == kind)
            row[f"{kind}_auroc"] = image_auroc(fused[sel], labels[sel])
        if maps:
            row["pixel_auroc"] = pixel_auroc(np.stack(maps), np.stack(masks))
            row["au_pro"] = compute_pro(np.stack(maps), np.stack(masks))
        per_draw.append(row)
    keys = [k for k in per_draw[0] if k != "images"]
    summary = {k: {"mean": float(np.nanmean([d[k] for d in per_draw])),
                   "std": float(np.nanstd([d[k] for d in per_draw], ddof=1)) if draws > 1 else 0.0}
               for k in keys}
    summary["n_test"] = len(samples)
    return {"product": src.name, "per_draw": per_draw, "summary": summary}


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calib-per-product", type=int, default=12)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--rule", default="ratio", choices=["ratio", "absolute"])
    parser.add_argument("--jepa-weight", type=float, default=None,
                        help="fusion weight of the JEPA channel (memory weight is 1); "
                             "defaults to scoring.jepa_weight in the config")
    parser.add_argument("--products", default="holdout",
                        help="holdout | trained | all | comma list of root:category")
    parser.add_argument("--num-references", type=int, default=1)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, payload = load_reference_model(args.checkpoint, device)
    all_sources = resolve_sources(cfg.get_path("data.sources"))
    trained = [s for s in all_sources if s.name in set(payload["products"])]
    holdout = [s for s in all_sources if s.category in set(payload.get("holdout", []))]

    if args.calibrate:
        t0 = time.time()
        path = os.path.join(os.path.dirname(args.checkpoint), "reference_calibration.json")
        existing = load_calibration(args.checkpoint) if os.path.isfile(path) else None
        weight = (args.jepa_weight if args.jepa_weight is not None
                  else cfg.get_path("scoring.jepa_weight", 1.0))
        if existing and existing.get("jepa_weight") != weight:
            existing = None              # a new weight invalidates thresholds for other counts
        calib = calibrate(model, cfg, device, trained, args.calib_per_product,
                          args.target_fpr, args.seed, args.rule, args.num_references,
                          weight, existing)
        with open(path, "w", encoding="utf-8") as h:
            json.dump(calib, h, indent=2)
        print(f"calibrated on {calib['n_pairs']} normal pairs from {len(calib['fitted_on'])} "
              f"products in {time.time() - t0:.0f}s: jepa_scale {calib['jepa_scale']:.4f} "
              f"memory_scale {calib['memory_scale']:.4f} jepa_weight {calib['jepa_weight']} "
              f"thresholds for {args.num_references} reference(s): {calib['thresholds']}",
              flush=True)

    if args.products == "none":
        return
    if args.products == "holdout":
        targets = holdout
    elif args.products == "trained":
        targets = trained
    elif args.products == "all":
        targets = holdout + trained
    else:
        targets = [PairSource(*item.rsplit(":", 1)) for item in args.products.split(",")]
    if not targets:
        raise SystemExit("no products to evaluate")

    scorer = build_scorer(model, cfg, device, load_calibration(args.checkpoint))
    results = []
    for src in targets:
        t0 = time.time()
        r = evaluate_product(scorer, src, args.num_references, args.draws, args.seed, args.max_test)
        r["seen_in_training"] = src.name in set(payload["products"])
        results.append(r)
        s = r["summary"]
        extra = "  ".join(f"{k.replace('_auroc', '')} {s[k]['mean']:.3f}" for k in s
                          if k.endswith("_anomalies_auroc") or k == "anomaly_auroc")
        sweep = " ".join(f"w{w}:{s[f'auroc_jepa_w{w}']['mean']:.3f}" for w in JEPA_WEIGHTS)
        print(f"{src.name:34s} {'seen' if r['seen_in_training'] else 'UNSEEN':6s} [{sweep}] "
              f"image {s['image_auroc']['mean']:.3f}+-{s['image_auroc']['std']:.3f}  "
              f"jepa {s['jepa_only_auroc']['mean']:.3f}  memory {s['memory_only_auroc']['mean']:.3f}  "
              f"pixel {s.get('pixel_auroc', {}).get('mean', float('nan')):.3f}  "
              f"aupro {s.get('au_pro', {}).get('mean', float('nan')):.3f}  "
              f"ratio TPR/FPR {s['tpr_ratio']['mean']:.2f}/{s['fpr_ratio']['mean']:.2f}  "
              f"abs TPR/FPR {s['tpr_absolute']['mean']:.2f}/{s['fpr_absolute']['mean']:.2f}  "
              f"{extra}  ({time.time() - t0:.0f}s)", flush=True)

    out = args.out or os.path.join(
        cfg.get_path("experiment.output_dir", "outputs"), cfg.get_path("experiment.name", "reference"),
        f"eval_{args.products if args.products in ('holdout', 'trained', 'all') else 'custom'}"
        f"_r{args.num_references}.json")
    save_json({"checkpoint": args.checkpoint, "num_references": args.num_references,
               "draws": args.draws, "results": results}, out)
    print("written", out)


if __name__ == "__main__":
    main()
