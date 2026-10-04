"""Evaluate a trained Logical-JEPA checkpoint on MVTec LOCO AD.

Reports image-level AUROC/AP, pixel-level AUROC and AU-PRO, each broken out
**separately for logical and structural anomalies** -- a single averaged number
would hide the effect this project exists to measure.

The decision threshold is fitted on held-out *normal* images only, so the
pipeline stays unsupervised end to end.

Usage::

    py -3.12 evaluate.py --checkpoint checkpoints/logical_jepa_multiscale/screw_board/final.pt
    py -3.12 evaluate.py --checkpoint <path> --set anomaly.distance=l2 anomaly.fusion=max
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch

from anomaly.metrics import (
    evaluate_split,
    load_subtype_manifest,
    subtype_breakdown,
    summarize,
)
from anomaly.scoring import build_scorer
from datasets.mvtec_loco import build_dataloaders, build_normal_loader
from masking import SweepMaskBank
from models import build_model
from utils.config import Config, deep_merge, load_config
from utils.logging_utils import get_logger, save_json
from utils.seed import seed_everything


ARCHITECTURE_KEYS = ("encoder", "predictor", "slots", "loss", "ema", "regularizer",
                     "data.img_size", "data.patch_size")


def load_model_from_checkpoint(path: str, device, cfg_override: Config | None = None):
    """Restore a model plus the config it was trained with.

    The architecture always comes from the checkpoint -- overrides may change
    *inference* settings (sweep windows, distance, fusion) but never the layer
    shapes, which would fail to load.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    trained = Config(payload["config"])
    cfg = trained

    if cfg_override:
        cfg = deep_merge(trained, cfg_override)
        # Re-assert the trained architecture: an override config that merely
        # *inherits* a different default (e.g. slots.enabled: false) must not
        # rebuild a model the saved weights do not fit.
        for key in ARCHITECTURE_KEYS:
            value = trained.get_path(key, None)
            if value is not None:
                cfg.set_path(key, copy.deepcopy(value))

    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    return model, cfg, payload


def build_sweep_bank(cfg, model, device) -> SweepMaskBank:
    return SweepMaskBank(
        grid_size=model.grid_size,
        windows=cfg.get_path("anomaly.sweep_windows", [2, 4, 6]),
        strides=cfg.get_path("anomaly.sweep_strides", None),
        device=device,
    )


@torch.no_grad()
def evaluate(cfg, checkpoint: str, logger, save_arrays: bool = False) -> dict:
    """Score the test split and compute the full metric report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.get_path("experiment.seed", 42))

    model, cfg, payload = load_model_from_checkpoint(checkpoint, device, cfg)
    category = cfg.get_path("data.category")
    logger.info(f"Loaded {checkpoint} (epoch {payload.get('epoch', '?')})")
    logger.info(model.describe())

    train_loader, val_loader, test_loader = build_dataloaders(cfg, category)
    logger.info(f"Test set: {test_loader.dataset.counts()}")

    bank = build_sweep_bank(cfg, model, device)
    scorer = build_scorer(cfg, model, bank)
    logger.info(f"Sweep: {bank.describe()}")
    logger.info(
        f"Scoring: distance={scorer.distance or model.loss_kind} fusion={scorer.fusion} "
        f"aggregation={scorer.aggregation} deviation={scorer.deviation} "
        f"cardinality={scorer.cardinality}"
        + (f" (weight {scorer.cardinality_weight})" if scorer.uses_cardinality else "")
    )

    # ---- calibration on NORMAL images only --------------------------- #
    calib_loader = val_loader if (
        cfg.get_path("eval.calibrate_on", "validation") == "validation" and val_loader
    ) else train_loader
    stats_loader = build_normal_loader(cfg, category, "train")

    # The positional statistics are a per-patch mean and std, so they are fitted
    # on the (much larger) normal training set; the threshold stays on the
    # held-out validation normals. Both are anomaly-free.
    calib = scorer.calibrate(
        calib_loader, device,
        sigma_threshold=cfg.get_path("eval.sigma_threshold", 3.0),
        stats_loader=stats_loader,
    )
    logger.info(
        f"Calibrated threshold on {calib.n_samples} normal images "
        f"(positional stats from {len(stats_loader.dataset)} train normals): "
        f"mean={calib.score_mean:.4f} std={calib.score_std:.4f} threshold={calib.threshold:.4f}"
    )

    # ---- score the test split ---------------------------------------- #
    out = scorer.score_loader(test_loader, device, collect_maps=True, progress=True)

    # Detection metrics come from the detection-normalised maps; localisation
    # metrics from the localisation-normalised ones. Study 4 showed no single
    # normalisation is best at both, so they are never mixed into one number.
    results = evaluate_split(
        scores=out["scores"],
        labels=out["labels"],
        defect_types=out["defect_types"],
        maps=out.get("localization_maps", out.get("maps")),
        masks=out.get("masks"),
    )
    results["normalize_detection"] = scorer.normalize_detection
    results["normalize_localization"] = scorer.normalize_localization
    results["cardinality"] = scorer.cardinality
    results["deviation"] = scorer.deviation

    manifest = load_subtype_manifest(
        os.path.join(cfg.get_path("data.root"), category))
    if manifest and out.get("paths"):
        results.update(subtype_breakdown(out["scores"], out["labels"], out["paths"], manifest))

    # When the two differ, also report what localisation *would* have been under
    # the detection normalisation, so the trade-off is visible in one place
    # rather than requiring a second run to see.
    if (scorer.normalize_localization != scorer.normalize_detection
            and out.get("masks") is not None and out.get("maps") is not None):
        anomalous = out["labels"] == 1
        if anomalous.any():
            from anomaly.metrics import compute_pro as _pro
            from anomaly.metrics import pixel_auroc as _px
            results["pixel_auroc_under_detection_norm"] = _px(
                out["maps"][anomalous], out["masks"][anomalous]
            )
            results["au_pro_under_detection_norm"] = _pro(
                out["maps"][anomalous], out["masks"][anomalous]
            )

    # Threshold-dependent operating point, for the demo's verdict.
    predicted = calib.is_anomalous(out["scores"])
    truth = out["labels"] == 1
    results["accuracy_at_threshold"] = float((predicted == truth).mean())
    results["tpr_at_threshold"] = float(predicted[truth].mean()) if truth.any() else float("nan")
    results["fpr_at_threshold"] = float(predicted[~truth].mean()) if (~truth).any() else float("nan")
    results["threshold"] = calib.threshold
    results["category"] = category
    results["checkpoint"] = checkpoint
    results["masking_strategy"] = cfg.get_path("masking.strategy")
    results["distance"] = scorer.distance or model.loss_kind
    results["sweep_windows"] = list(bank.scales())
    results["fusion"] = scorer.fusion

    logger.info("\n" + summarize(results, f"{category} -- {cfg.get_path('experiment.name')}"))
    logger.info(
        f"  operating point: acc={results['accuracy_at_threshold']:.4f} "
        f"TPR={results['tpr_at_threshold']:.4f} FPR={results['fpr_at_threshold']:.4f}"
    )

    # ---- persist ----------------------------------------------------- #
    out_dir = os.path.join(cfg.get_path("experiment.output_dir", "outputs"),
                           cfg.get_path("experiment.name", "run"), category)
    os.makedirs(out_dir, exist_ok=True)

    save_json({"results": results, "calibration": calib.to_dict()},
              os.path.join(out_dir, "eval_results.json"))

    # The calibration lives beside the checkpoint so the web demo can load a
    # threshold without re-running evaluation.
    save_json(calib.to_dict(),
              os.path.join(os.path.dirname(checkpoint), "calibration.json"))

    if save_arrays:
        np.savez_compressed(
            os.path.join(out_dir, "scores.npz"),
            scores=out["scores"], labels=out["labels"], defect_types=out["defect_types"],
        )
        logger.info(f"Saved raw scores to {out_dir}/scores.npz")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Logical-JEPA checkpoint")
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--config", default=None,
                        help="optional config whose inference settings override the checkpoint's")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="dotted overrides, e.g. anomaly.distance=l2")
    parser.add_argument("--save-arrays", action="store_true",
                        help="also dump raw per-image scores as .npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        cfg = load_config(args.config, args.set)
    else:
        cfg = Config()
        for item in args.set:
            key, _, value = item.partition("=")
            import yaml
            cfg.set_path(key.strip(), yaml.safe_load(value.strip()))

    logger = get_logger("evaluate")
    evaluate(cfg, args.checkpoint, logger, save_arrays=args.save_arrays)


if __name__ == "__main__":
    main()
