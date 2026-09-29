"""Evaluation metrics for unsupervised anomaly detection.

Three levels are reported, and the third is the one that matters most for this
benchmark:

* **Image AUROC** -- can the model tell an anomalous image from a normal one?
* **Pixel AUROC** -- does the heatmap land on the defect? Optimistic on MVTec
  LOCO, because defects are a tiny fraction of the pixels and a mostly-blank map
  already scores well.
* **AU-PRO** -- per-region overlap averaged over *connected components*, so a
  large defect cannot drown out a small one. This is the metric MVTec themselves
  report, and it is where a method that only finds big blobs gets caught.

Every metric is additionally computed **separately for logical and structural
anomalies**, which is the entire point of the study: a single averaged number
would hide precisely the effect being measured.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import auc, average_precision_score, roc_auc_score

from datasets.mvtec_loco import DEFECT_GOOD, DEFECT_LOGICAL, DEFECT_STRUCTURAL


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Image-level AUROC. Returns NaN if only one class is present."""
    labels = np.asarray(labels).reshape(-1)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores).reshape(-1)))


def image_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision -- more informative than AUROC on imbalanced splits."""
    labels = np.asarray(labels).reshape(-1)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, np.asarray(scores).reshape(-1)))


def pixel_auroc(maps: np.ndarray, masks: np.ndarray, max_pixels: int = 4_000_000) -> float:
    """Pixel-level AUROC over all images.

    Full-resolution evaluation of a whole test split is tens of millions of
    points; the arrays are subsampled (stratified by class, so every defect
    pixel is retained) above ``max_pixels`` to keep this tractable without
    biasing the estimate.
    """
    flat_scores = np.asarray(maps, dtype=np.float32).reshape(-1)
    flat_labels = (np.asarray(masks).reshape(-1) > 0.5).astype(np.uint8)

    if len(np.unique(flat_labels)) < 2:
        return float("nan")

    if flat_scores.size > max_pixels:
        pos = np.flatnonzero(flat_labels)
        neg = np.flatnonzero(flat_labels == 0)
        n_neg = max(max_pixels - pos.size, pos.size)
        rng = np.random.default_rng(0)
        neg = rng.choice(neg, size=min(n_neg, neg.size), replace=False)
        keep = np.concatenate([pos, neg])
        flat_scores, flat_labels = flat_scores[keep], flat_labels[keep]

    return float(roc_auc_score(flat_labels, flat_scores))


def compute_pro(
    maps: np.ndarray,
    masks: np.ndarray,
    max_fpr: float = 0.3,
    num_thresholds: int = 100,
) -> float:
    """Area under the Per-Region-Overlap curve, normalised to ``max_fpr``.

    For each threshold, PRO is the mean over *ground-truth connected components*
    of the fraction of that component the prediction covers. Averaging over
    regions rather than pixels is what stops one large defect dominating.

    The curve is integrated up to ``max_fpr`` (0.3 by the MVTec convention) and
    divided by it, so a perfect detector scores 1.0.

    Args:
        maps: (N, 1, H, W) or (N, H, W) anomaly maps for anomalous images.
        masks: matching binary ground truth.
        max_fpr: false-positive-rate limit of the integration.
        num_thresholds: sampling resolution of the curve.
    """
    maps = np.asarray(maps, dtype=np.float32).squeeze()
    masks = (np.asarray(masks).squeeze() > 0.5)

    if maps.ndim == 2:
        maps, masks = maps[None], masks[None]
    if masks.sum() == 0:
        return float("nan")

    # Label connected components once; 8-connectivity matches MVTec's masks.
    structure = np.ones((3, 3), dtype=int)
    regions: list[tuple[int, np.ndarray]] = []
    for i in range(masks.shape[0]):
        if not masks[i].any():
            continue
        labelled, n = ndimage.label(masks[i], structure=structure)
        for r in range(1, n + 1):
            regions.append((i, labelled == r))

    if not regions:
        return float("nan")

    normal_pixels = ~masks
    n_normal = int(normal_pixels.sum())
    if n_normal == 0:
        return float("nan")

    lo, hi = float(maps.min()), float(maps.max())
    thresholds = np.linspace(hi, lo, num_thresholds)

    fprs, pros = [], []
    for thr in thresholds:
        binary = maps >= thr

        fpr = float((binary & normal_pixels).sum()) / n_normal
        overlaps = [
            float((binary[i] & region).sum()) / float(region.sum())
            for i, region in regions
        ]

        fprs.append(fpr)
        pros.append(float(np.mean(overlaps)))

        if fpr > max_fpr:
            break

    fprs = np.asarray(fprs)
    pros = np.asarray(pros)

    order = np.argsort(fprs)
    fprs, pros = fprs[order], pros[order]

    # Clip the curve exactly at max_fpr, interpolating the final point so the
    # integration limit does not depend on threshold sampling.
    keep = fprs <= max_fpr
    if keep.sum() < 2:
        return 0.0

    fprs_c, pros_c = fprs[keep], pros[keep]
    if fprs_c[-1] < max_fpr and keep.sum() < len(fprs):
        nxt = int(keep.sum())
        span = fprs[nxt] - fprs_c[-1]
        if span > 1e-12:
            frac = (max_fpr - fprs_c[-1]) / span
            fprs_c = np.append(fprs_c, max_fpr)
            pros_c = np.append(pros_c, pros_c[-1] + frac * (pros[nxt] - pros_c[-1]))

    return float(auc(fprs_c, pros_c) / max_fpr)


def evaluate_split(
    scores: np.ndarray,
    labels: np.ndarray,
    defect_types: np.ndarray,
    maps: np.ndarray | None = None,
    masks: np.ndarray | None = None,
    compute_localization: bool = True,
    max_fpr: float = 0.3,
) -> dict:
    """Full metric report, overall and split by anomaly family.

    Every anomaly family is scored against the *same* set of normal images, so
    ``logical_auroc`` and ``structural_auroc`` are directly comparable to each
    other and to the overall figure.
    """
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    defect_types = np.asarray(defect_types).reshape(-1)

    results: dict[str, float | int] = {
        "image_auroc": image_auroc(scores, labels),
        "image_ap": image_average_precision(scores, labels),
        "n_total": int(scores.size),
        "n_normal": int((labels == 0).sum()),
        "n_anomalous": int((labels == 1).sum()),
    }

    normal_mask = defect_types == DEFECT_GOOD

    for family, key in ((DEFECT_LOGICAL, "logical"), (DEFECT_STRUCTURAL, "structural")):
        family_mask = defect_types == family
        results[f"n_{key}"] = int(family_mask.sum())

        if family_mask.sum() == 0 or normal_mask.sum() == 0:
            results[f"{key}_auroc"] = float("nan")
            continue

        subset = normal_mask | family_mask
        results[f"{key}_auroc"] = image_auroc(scores[subset], labels[subset])
        results[f"{key}_ap"] = image_average_precision(scores[subset], labels[subset])

    if compute_localization and maps is not None and masks is not None:
        anomalous = labels == 1
        if anomalous.any():
            results["pixel_auroc"] = pixel_auroc(maps[anomalous], masks[anomalous])
            results["au_pro"] = compute_pro(maps[anomalous], masks[anomalous], max_fpr=max_fpr)

            for family, key in ((DEFECT_LOGICAL, "logical"), (DEFECT_STRUCTURAL, "structural")):
                fam = defect_types == family
                if fam.any():
                    results[f"{key}_pixel_auroc"] = pixel_auroc(maps[fam], masks[fam])
                    results[f"{key}_au_pro"] = compute_pro(maps[fam], masks[fam], max_fpr=max_fpr)

    return results


def load_subtype_manifest(category_dir: str) -> dict | None:
    """``{"<family>/<file>": {"subtype", "direction"}}`` if the dataset ships one.

    The synthetic generator writes it; real MVTec LOCO does not, in which case
    the breakdown is simply skipped.
    """
    import json
    import os

    path = os.path.join(category_dir, "defect_subtypes.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def subtype_breakdown(scores: np.ndarray, labels: np.ndarray, paths: list[str],
                      manifest: dict) -> dict:
    """Image AUROC per defect subtype and per direction, each against all normals.

    Direction (``removal`` / ``addition`` / ``rearrangement``) is the grouping
    that tests the Phase 1 mechanism: if missing content is what signed scoring
    under-ranks, ``removal_auroc`` is where it shows.

    Returns flat keys (``subtype_<name>_auroc``, ``<direction>_auroc``) so they
    flow through the multi-seed aggregation like any other metric.
    """
    import os

    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    normal = labels == 0

    subtypes, directions = [], []
    for path in paths:
        parts = os.path.normpath(path).replace("\\", "/").split("/")
        entry = manifest.get("/".join(parts[-2:]), {})
        subtypes.append(entry.get("subtype", ""))
        directions.append(entry.get("direction", ""))
    subtypes, directions = np.array(subtypes), np.array(directions)

    out: dict[str, float] = {}
    # "structural" as a direction is exactly the structural family, which
    # `structural_auroc` already reports.
    for prefix, groups, skip in (("subtype_", subtypes, {""}),
                                 ("", directions, {"", "structural"})):
        for name in sorted(set(groups[~normal]) - skip):
            member = groups == name
            if normal.any() and member.any():
                subset = normal | member
                out[f"{prefix}{name}_auroc"] = image_auroc(scores[subset], labels[subset])
    return out


def summarize(results: dict, title: str = "Results") -> str:
    """Human-readable metric block for the console and the log file."""
    lines = [f"=== {title} ===",
             f"  images: {results.get('n_total', 0)} "
             f"({results.get('n_normal', 0)} normal, {results.get('n_anomalous', 0)} anomalous)"]

    def row(label: str, key: str) -> None:
        value = results.get(key)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            lines.append(f"  {label:<34s} {value:.4f}")

    row("Image AUROC (overall)", "image_auroc")
    row("Image AP    (overall)", "image_ap")
    row("Image AUROC - LOGICAL", "logical_auroc")
    row("Image AUROC - STRUCTURAL", "structural_auroc")
    row("Pixel AUROC (overall)", "pixel_auroc")
    row("AU-PRO      (overall)", "au_pro")
    row("Pixel AUROC - LOGICAL", "logical_pixel_auroc")
    row("Pixel AUROC - STRUCTURAL", "structural_pixel_auroc")
    row("AU-PRO      - LOGICAL", "logical_au_pro")
    row("AU-PRO      - STRUCTURAL", "structural_au_pro")

    breakdown = sorted(k for k in results
                       if k.endswith("_auroc") and (k.startswith("subtype_") or k in (
                           "removal_auroc", "addition_auroc", "rearrangement_auroc")))
    if breakdown:
        lines.append("  -- by defect subtype / direction --")
        for key in breakdown:
            row(key.replace("subtype_", "  ").replace("_auroc", ""), key)

    return "\n".join(lines)
