"""Run the Logical-JEPA ablation studies and emit the results tables.

Four studies, defined in ``configs/ablations.yaml``:

``masking``
    The headline study. Trains one model per masking strategy -- random-patch,
    small-only, large-only, multi-scale, multi-scale-curriculum -- and evaluates
    each on structural and logical anomalies separately. This is expensive
    (one full training run per arm) and is what the research claim rests on.

``scoring``
    Which latent distance makes the best anomaly score (Methods A/B/C).
    Inference only: every arm re-scores the same trained checkpoint.

``sweep``
    Whether the *inference* window scale matters independently of the training
    scale. Inference only.

``normalization``
    How the per-scale error grids are normalised before fusion. This is a real
    trade-off rather than a tuning knob: per-position statistics fitted on
    normal images are what make image-level detection work, while per-image
    normalisation gives sharper localisation but near-chance image AUROC.
    Inference only.

Usage::

    py -3.12 run_ablations.py --config configs/ablations.yaml --study all
    py -3.12 run_ablations.py --study scoring --checkpoint <path>   # no training
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np

from evaluate import evaluate
from train import train
from utils.config import Config, load_config, save_config
from utils.logging_utils import append_csv, format_table, get_logger, save_json

# Columns reported for every arm, in presentation order.
REPORT_KEYS = [
    "image_auroc",
    "logical_auroc",
    "structural_auroc",
    "pixel_auroc",
    "au_pro",
    "logical_au_pro",
    "structural_au_pro",
]


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Copy ``cfg`` and apply an arm's dotted overrides."""
    out = Config(copy.deepcopy(cfg.to_dict()))
    for key, value in (overrides or {}).items():
        out.set_path(key, value)
    return out


def _row(arm_name: str, description: str, results: dict) -> dict:
    """Flatten one arm's metrics into a CSV/table row."""
    row = {"arm": arm_name, "description": description}
    for key in REPORT_KEYS:
        value = results.get(key, float("nan"))
        row[key] = float(value) if value is not None else float("nan")
    return row


# --------------------------------------------------------------------------- #
def run_masking_study(cfg, arms, logger, out_dir, csv_path, retrain=True) -> list[dict]:
    """Train and evaluate one model per masking strategy."""
    rows = []

    for arm in arms:
        name, description = arm["name"], arm.get("description", "")
        logger.info(f"\n{'='*72}\nMASKING ARM: {name} -- {description}\n{'='*72}")

        arm_cfg = apply_overrides(cfg, arm.get("overrides", {}))
        arm_cfg.set_path("experiment.name", f"{cfg.get_path('experiment.name')}/{name}")

        category = arm_cfg.get_path("data.category")
        ckpt = os.path.join(
            arm_cfg.get_path("experiment.checkpoint_dir", "checkpoints"),
            arm_cfg.get_path("experiment.name"), category, "final.pt",
        )

        start = time.time()
        if retrain or not os.path.isfile(ckpt):
            ckpt = train(arm_cfg, logger)
        else:
            logger.info(f"Reusing existing checkpoint {ckpt}")

        results = evaluate(arm_cfg, ckpt, logger)
        results["train_minutes"] = (time.time() - start) / 60.0

        row = _row(name, description, results)
        rows.append(row)
        append_csv(row, csv_path)
        save_json(results, os.path.join(out_dir, f"masking_{name}.json"))

    return rows


def run_inference_study(cfg, arms, checkpoint, logger, out_dir, csv_path,
                        study: str) -> list[dict]:
    """Re-score one trained checkpoint under each arm's inference settings."""
    rows = []

    for arm in arms:
        name, description = arm["name"], arm.get("description", "")
        logger.info(f"\n{'='*72}\n{study.upper()} ARM: {name} -- {description}\n{'='*72}")

        arm_cfg = apply_overrides(cfg, arm.get("overrides", {}))
        arm_cfg.set_path("experiment.name", f"{cfg.get_path('experiment.name')}/{study}/{name}")

        results = evaluate(arm_cfg, checkpoint, logger)

        row = _row(name, description, results)
        rows.append(row)
        append_csv(row, csv_path)
        save_json(results, os.path.join(out_dir, f"{study}_{name}.json"))

    return rows


def run_baselines(cfg, checkpoint, logger, out_dir, csv_path) -> list[dict]:
    """Train and evaluate the autoencoder and PatchCore comparisons.

    PatchCore reuses the trained JEPA target encoder as its feature extractor,
    so it needs ``checkpoint``; the autoencoder is trained from scratch here.
    """
    import torch

    from anomaly.metrics import evaluate_split, summarize
    from baselines.autoencoder import AutoencoderScorer, train_autoencoder
    from baselines.patchcore import PatchCore
    from datasets.mvtec_loco import build_dataloaders
    from evaluate import load_model_from_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    category = cfg.get_path("data.category")
    train_loader, val_loader, test_loader = build_dataloaders(cfg, category)
    calib_loader = val_loader or train_loader
    sigma_thr = cfg.get_path("eval.sigma_threshold", 3.0)

    rows = []

    # ---- Autoencoder ------------------------------------------------- #
    logger.info(f"\n{'='*72}\nBASELINE: Convolutional Autoencoder\n{'='*72}")
    ae = train_autoencoder(cfg, train_loader, device, logger)
    ae_scorer = AutoencoderScorer(
        ae,
        sigma=cfg.get_path("anomaly.smooth_sigma", 4.0),
        aggregation=cfg.get_path("anomaly.aggregation", "topk"),
        top_k_ratio=cfg.get_path("anomaly.top_k_ratio", 0.01),
    )
    ae_scorer.calibrate(calib_loader, device, sigma_thr)
    out = ae_scorer.score_loader(test_loader, device, collect_maps=True, progress=True)
    ae_results = evaluate_split(out["scores"], out["labels"], out["defect_types"],
                                out.get("maps"), out.get("masks"))
    logger.info("\n" + summarize(ae_results, "Autoencoder baseline"))

    rows.append(_row("baseline_autoencoder", "Conv AE reconstruction error", ae_results))
    append_csv(rows[-1], csv_path)
    save_json(ae_results, os.path.join(out_dir, "baseline_autoencoder.json"))

    # ---- PatchCore --------------------------------------------------- #
    logger.info(f"\n{'='*72}\nBASELINE: PatchCore (from-scratch features)\n{'='*72}")
    model, model_cfg, _ = load_model_from_checkpoint(checkpoint, device, cfg)
    pc = PatchCore(
        model,
        coreset_ratio=cfg.get_path("baseline.coreset_ratio", 0.01),
        n_neighbors=cfg.get_path("baseline.n_neighbors", 3),
        sigma=cfg.get_path("anomaly.smooth_sigma", 4.0),
        aggregation=cfg.get_path("anomaly.aggregation", "topk"),
        top_k_ratio=cfg.get_path("anomaly.top_k_ratio", 0.01),
    )
    pc.fit(train_loader, device, logger)
    pc.calibrate(calib_loader, device, sigma_thr)
    out = pc.score_loader(test_loader, device, collect_maps=True, progress=True)
    pc_results = evaluate_split(out["scores"], out["labels"], out["defect_types"],
                                out.get("maps"), out.get("masks"))
    logger.info("\n" + summarize(pc_results, "PatchCore baseline"))

    rows.append(_row("baseline_patchcore", "Patch memory bank + coreset kNN", pc_results))
    append_csv(rows[-1], csv_path)
    save_json(pc_results, os.path.join(out_dir, "baseline_patchcore.json"))

    return rows


# --------------------------------------------------------------------------- #
def print_study(rows: list[dict], title: str, logger) -> None:
    """Log a study's table, with the best arm per metric called out."""
    if not rows:
        return

    display = [
        {k: v for k, v in row.items() if k != "description"}
        for row in rows
    ]
    logger.info(f"\n\n### {title}\n" + format_table(display))

    best_lines = []
    for key in REPORT_KEYS:
        values = np.array([row.get(key, np.nan) for row in rows], dtype=float)
        if np.all(np.isnan(values)):
            continue
        winner = rows[int(np.nanargmax(values))]["arm"]
        best_lines.append(f"    best {key:<20s} -> {winner} ({np.nanmax(values):.4f})")

    if best_lines:
        logger.info("  winners:\n" + "\n".join(best_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Logical-JEPA ablation studies")
    parser.add_argument("--config", default="configs/ablations.yaml")
    parser.add_argument("--study", default="all",
                        choices=["all", "masking", "scoring", "sweep",
                                 "normalization", "baselines"])
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint for the inference-only studies and PatchCore; "
                             "defaults to the multi-scale masking arm's checkpoint")
    parser.add_argument("--category", default=None)
    parser.add_argument("--no-retrain", action="store_true",
                        help="reuse existing masking-arm checkpoints when present")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    if args.category:
        cfg.set_path("data.category", args.category)

    category = cfg.get_path("data.category")
    out_dir = os.path.join(cfg.get_path("experiment.output_dir", "outputs"),
                           cfg.get_path("experiment.name", "ablation_study"), category)
    os.makedirs(out_dir, exist_ok=True)

    logger = get_logger("ablations", os.path.join(out_dir, "ablations.log"))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    logger.info(f"Ablation study on '{category}' -> {out_dir}")

    studies = (["masking", "scoring", "sweep", "normalization", "baselines"]
               if args.study == "all" else [args.study])
    all_rows: dict[str, list[dict]] = {}

    # The masking study produces the checkpoint the other studies re-score.
    checkpoint = args.checkpoint

    if "masking" in studies:
        arms = cfg.get_path("ablation.masking", [])
        csv_path = os.path.join(out_dir, "masking_study.csv")
        all_rows["Study 1: masking strategy"] = run_masking_study(
            cfg, arms, logger, out_dir, csv_path, retrain=not args.no_retrain
        )

        if checkpoint is None:
            multiscale = next((a for a in arms if "multiscale" in a["name"]), arms[-1] if arms else None)
            if multiscale:
                checkpoint = os.path.join(
                    cfg.get_path("experiment.checkpoint_dir", "checkpoints"),
                    cfg.get_path("experiment.name"), multiscale["name"], category, "final.pt",
                )

    if checkpoint is None or not os.path.isfile(checkpoint):
        remaining = [s for s in studies if s != "masking"]
        if remaining:
            logger.error(
                f"Studies {remaining} need a trained checkpoint. Pass --checkpoint <path>, "
                f"or run --study masking first."
            )
            checkpoint = None

    if checkpoint:
        for study, title in (("scoring", "Study 2: anomaly score function"),
                             ("sweep", "Study 3: inference sweep configuration"),
                             ("normalization", "Study 4: grid normalisation")):
            if study in studies:
                arms = cfg.get_path(f"ablation.{study}", [])
                csv_path = os.path.join(out_dir, f"{study}_study.csv")
                all_rows[title] = run_inference_study(
                    cfg, arms, checkpoint, logger, out_dir, csv_path, study
                )

        if "baselines" in studies:
            csv_path = os.path.join(out_dir, "baselines.csv")
            all_rows["Baselines"] = run_baselines(cfg, checkpoint, logger, out_dir, csv_path)

    logger.info("\n\n" + "=" * 72 + "\nABLATION SUMMARY\n" + "=" * 72)
    for title, rows in all_rows.items():
        print_study(rows, title, logger)

    save_json({title: rows for title, rows in all_rows.items()},
              os.path.join(out_dir, "ablation_summary.json"))
    logger.info(f"\nAll results written to {out_dir}")


if __name__ == "__main__":
    main()
