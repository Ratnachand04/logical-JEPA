"""Qualitative visualisation of Logical-JEPA predictions.

Produces the figures used in the report and the presentation:

``--mode grid`` (default)
    Per-sample panels: input, anomaly heatmap, overlay, ground truth, and one
    column per sweep scale. The per-scale columns are the interesting part --
    they show the small windows lighting up on scratches while the large windows
    light up on missing / misplaced components, which is the visual form of the
    ablation result.

``--mode masks``
    What the masking strategies actually hide, drawn on a real image. Useful for
    explaining the method before showing any numbers.

``--mode latent``
    The "why" figure: for one masked region, the predicted embedding against the
    observed one, per channel, with their cosine distance. This is the picture
    that makes "high latent discrepancy = logical anomaly" concrete rather than
    asserted.

Usage::

    py -3.12 visualize.py --checkpoint checkpoints/<name>/<category>/final.pt
    py -3.12 visualize.py --checkpoint <path> --mode latent --defect logical_anomalies
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")   # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import torch

from anomaly.scoring import build_scorer
from datasets.mvtec_loco import MVTecLOCO
from datasets.transforms import to_numpy_image
from evaluate import build_sweep_bank, load_model_from_checkpoint
from masking import build_mask_generator
from models.patch_embed import gather_tokens
from utils.config import Config
from utils.logging_utils import get_logger


def _overlay(image: np.ndarray, heat: np.ndarray, alpha: float = 0.5,
             cmap: str = "jet") -> np.ndarray:
    """Blend a normalised heatmap over an RGB image."""
    lo, hi = float(heat.min()), float(heat.max())
    norm = (heat - lo) / max(hi - lo, 1e-6)
    colour = (plt.get_cmap(cmap)(norm)[..., :3] * 255).astype(np.uint8)
    return (image * (1 - alpha) + colour * alpha).astype(np.uint8)


def _pick_samples(dataset: MVTecLOCO, num: int, defect: str | None) -> list[int]:
    """Choose sample indices, spread evenly across the requested defect type."""
    candidates = [
        i for i, s in enumerate(dataset.samples)
        if defect is None or s.defect_type == defect
    ]
    if not candidates:
        raise ValueError(f"No samples with defect_type='{defect}'")
    step = max(len(candidates) // num, 1)
    return candidates[::step][:num]


# --------------------------------------------------------------------------- #
def visualize_grid(model, scorer, dataset, indices, out_path, device,
                   alpha=0.5, cmap="jet") -> str:
    """Input / heatmap / overlay / GT / per-scale panels for each sample."""
    scales = list(scorer.mask_bank.scales())
    n_cols = 4 + len(scales)
    fig, axes = plt.subplots(len(indices), n_cols,
                             figsize=(2.4 * n_cols, 2.4 * len(indices)))
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(indices):
        item = dataset[idx]
        image = item["image"].unsqueeze(0).to(device)

        out = scorer.predict(image)
        heat = out["maps"][0, 0].cpu().numpy()
        rgb = to_numpy_image(item["image"])

        verdict = "ANOMALOUS" if out["is_anomalous"][0] else "NORMAL"
        truth = item["defect_type"]

        panels = [
            (rgb, f"{truth}", None),
            (heat, f"anomaly map\nscore={out['score'][0]:.3f}", cmap),
            (_overlay(rgb, heat, alpha, cmap), f"{verdict}  (z={out['z_score'][0]:+.1f})", None),
        ]

        gt = item["mask"][0].numpy() if "mask" in item else np.zeros_like(heat)
        panels.append((gt, "ground truth", "gray"))

        for window in scales:
            grid = out["grids"][window][0].cpu().numpy()
            panels.append((grid, f"scale {window}x{window}", cmap))

        for col, (data, title, cm) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(data, cmap=cm) if cm else ax.imshow(data)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    fig.suptitle(
        f"Logical-JEPA -- {dataset.category}  "
        f"(threshold z=0 at score {scorer.calibration.threshold:.3f})",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize_masks(cfg, dataset, out_path, grid_size=16, seed=0) -> str:
    """Draw what each masking strategy hides, over a real training image."""
    strategies = ["random_patch", "small", "large", "multiscale"]
    item = dataset[0]
    rgb = to_numpy_image(item["image"])
    patch = rgb.shape[0] // grid_size

    fig, axes = plt.subplots(2, len(strategies), figsize=(3.2 * len(strategies), 6.6))

    for col, strategy in enumerate(strategies):
        node = Config(cfg.get("masking", {}).to_dict())
        node["strategy"] = strategy
        gen = build_mask_generator(node, grid_size, seed)
        spec = gen()

        covered = spec.target_mask_grid().numpy()
        # Upsample the patch-grid mask to pixels so it lines up with the image.
        big = np.kron(covered > 0, np.ones((patch, patch), dtype=bool))

        masked = rgb.copy()
        masked[big] = (masked[big] * 0.15).astype(np.uint8)   # hidden = darkened

        axes[0, col].imshow(masked)
        axes[0, col].set_title(
            f"{strategy}\n{spec.num_targets} blocks, "
            f"{spec.num_target_patches}/{grid_size**2} patches hidden",
            fontsize=9,
        )
        axes[0, col].axis("off")

        axes[1, col].imshow(covered, cmap="viridis", vmin=0, vmax=1)
        axes[1, col].set_title("target mask on 16x16 grid", fontsize=8)
        axes[1, col].axis("off")

    fig.suptitle("Masking strategies -- what the context encoder is NOT allowed to see",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


@torch.no_grad()
def visualize_latent(model, dataset, indices, out_path, device, window=6,
                     scale_stats=None) -> str:
    """Predicted vs observed embedding for the most surprising region.

    For each sample the sweep window with the largest error is selected, and the
    predicted and observed embedding of its single most surprising *patch* are
    plotted channel by channel. A normal patch gives two nearly-overlapping
    curves; a logical anomaly gives two that diverge -- the model expected one
    thing and found another.

    Two details make this figure honest rather than decorative:

    * The peak is chosen on the **position-normalised** error when calibration
      statistics are available, so the region shown is the one the heatmap
      actually flagged. Ranking on raw error instead just finds whichever
      position is hardest to predict on every image, anomalous or not.
    * A single patch is plotted rather than the block mean. Averaging 36 patch
      embeddings cancels the disagreement and reports a cosine distance near
      zero even for a clearly detected anomaly.
    """
    from masking.sweep_mask import SweepMaskBank

    bank = SweepMaskBank(model.grid_size, windows=(window,), device=device)
    stat = (scale_stats or {}).get(window)

    fig, axes = plt.subplots(len(indices), 3, figsize=(13, 3.0 * len(indices)))
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(indices):
        item = dataset[idx]
        image = item["image"].unsqueeze(0).to(device)
        rgb = to_numpy_image(item["image"])

        full_target = model.encode_targets(image)

        # Per-position baseline difficulty, as a flat (num_patches,) tensor.
        if stat is not None:
            mean_t = torch.as_tensor(stat[0], device=device, dtype=torch.float32).reshape(-1)
            std_t = torch.as_tensor(stat[1], device=device, dtype=torch.float32).reshape(-1)
        else:
            mean_t = std_t = None

        best = {"err": -float("inf")}
        for ctx_idx, tgt_idx, boxes in bank.batched(window, chunk=64):
            err, _ = model.sweep_scale(image, ctx_idx, tgt_idx, full_target=full_target)
            per_patch = err[0]                                     # (M, K)

            if mean_t is not None:
                idx_dev = tgt_idx.to(device)
                per_patch = (per_patch - mean_t[idx_dev]) / std_t[idx_dev].clamp_min(1e-6)

            per_pos = per_patch.mean(dim=1)                        # (M,)
            top = int(per_pos.argmax().item())
            if float(per_pos[top]) > best["err"]:
                best = {
                    "err": float(per_pos[top]),
                    "box": boxes[top],
                    "ctx": ctx_idx[top : top + 1],
                    "tgt": tgt_idx[top : top + 1],
                    # Which patch inside the block disagreed most.
                    "patch": int(per_patch[top].argmax().item()),
                    "patch_z": float(per_patch[top].max().item()),
                }

        # Re-run the winning position to recover the actual vectors.
        tokens = model.context_encoder.embed_patches(image)
        ctx_tokens = gather_tokens(tokens, best["ctx"].to(device))
        encoded = model.context_encoder.forward_tokens(ctx_tokens)
        pred = model.predictor(encoded, best["ctx"].to(device), best["tgt"].to(device))
        actual = gather_tokens(full_target, best["tgt"].to(device))

        k = best["patch"]
        p = pred[0, k].float().cpu().numpy()
        a = actual[0, k].float().cpu().numpy()
        cos = float(np.dot(p, a) / (np.linalg.norm(p) * np.linalg.norm(a) + 1e-8))

        top_r, left_c, h, w = best["box"]
        patch = rgb.shape[0] // model.grid_size
        marked = rgb.copy()
        y0, x0, y1, x1 = top_r * patch, left_c * patch, (top_r + h) * patch, (left_c + w) * patch
        marked[y0:y1, x0:x1] = (marked[y0:y1, x0:x1] * 0.45).astype(np.uint8)
        marked[y0:y1, x0:x0 + 3] = marked[y0:y1, x1 - 3:x1] = [255, 40, 40]
        marked[y0:y0 + 3, x0:x1] = marked[y1 - 3:y1, x0:x1] = [255, 40, 40]

        # Restore full brightness inside the single peak patch and ring it in
        # gold, so the plotted vectors are tied to a visible location.
        pr, pc = top_r + k // w, left_c + k % w
        py0, px0 = pr * patch, pc * patch
        marked[py0:py0 + patch, px0:px0 + patch] = rgb[py0:py0 + patch, px0:px0 + patch]
        gold = [255, 214, 0]
        marked[py0:py0 + patch, px0:px0 + 3] = gold
        marked[py0:py0 + patch, px0 + patch - 3:px0 + patch] = gold
        marked[py0:py0 + 3, px0:px0 + patch] = gold
        marked[py0 + patch - 3:py0 + patch, px0:px0 + patch] = gold

        axes[row, 0].imshow(marked)
        axes[row, 0].set_title(
            f"{item['defect_type']}\nmost surprising {w}x{h} region "
            f"(peak +{best['patch_z']:.1f}σ)",
            fontsize=9,
        )
        axes[row, 0].axis("off")

        axes[row, 1].plot(a, label="observed  $Z_t$", lw=1.0, color="#2b6cb0")
        axes[row, 1].plot(p, label="predicted $\\hat{Z}_t$", lw=1.0, color="#c53030", alpha=0.85)
        axes[row, 1].set_title(
            f"peak patch (gold box) -- cosine distance = {1 - cos:.3f}", fontsize=9
        )
        axes[row, 1].set_xlabel("embedding channel", fontsize=8)
        axes[row, 1].legend(fontsize=7)
        axes[row, 1].tick_params(labelsize=7)

        axes[row, 2].bar(range(len(p)), np.abs(p - a), width=1.0, color="#805ad5")
        axes[row, 2].set_title("per-channel |discrepancy|", fontsize=9)
        axes[row, 2].set_xlabel("embedding channel", fontsize=8)
        axes[row, 2].tick_params(labelsize=7)

    fig.suptitle("Why a logical anomaly is detected: predicted vs observed latent", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path



@torch.no_grad()
def visualize_slots(model, slot_ckpt_path, dataset, indices, out_path, device) -> str:
    """Render each slot's attention map over real images -- the Phase 3a gate.

    This is the check that decides whether the bottleneck may be wired into the
    predictor. The numeric gate in ``train_slots.py`` can only detect total
    collapse; only looking at the maps shows whether slots have latched onto
    *components* (a screw, the board, the indicator strip) rather than carving
    the image into arbitrary stripes.

    Slot order is arbitrary and differs per image -- see the module docstring of
    ``models.slot_bottleneck`` for why that is inherent, and what the project
    does about it.
    """
    from models.slot_bottleneck import SlotBottleneck

    payload = torch.load(slot_ckpt_path, map_location=device, weights_only=False)
    num_slots = payload["num_slots"]

    bottleneck = SlotBottleneck(
        token_dim=payload["token_dim"], slot_dim=payload["token_dim"],
        num_slots=num_slots,
    ).to(device)
    bottleneck.load_state_dict(payload["model"])
    bottleneck.eval()

    grid = model.grid_size
    n_cols = num_slots + 2
    fig, axes = plt.subplots(len(indices), n_cols,
                             figsize=(1.9 * n_cols, 2.1 * len(indices)))
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(indices):
        item = dataset[idx]
        image = item["image"].unsqueeze(0).to(device)
        rgb = to_numpy_image(item["image"])

        tokens = model.encode_targets(image)
        _slots, attn = bottleneck.encode(tokens, return_attn=True)   # (1, S, N)

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(item["defect_type"], fontsize=8)
        axes[row, 0].axis("off")

        # Hard assignment: which slot claims each patch.
        assignment = attn[0].argmax(dim=0).reshape(grid, grid).cpu().numpy()
        axes[row, 1].imshow(assignment, cmap="tab10", vmin=0, vmax=max(num_slots - 1, 1),
                            interpolation="nearest")
        axes[row, 1].set_title("slot assignment", fontsize=8)
        axes[row, 1].axis("off")

        for slot in range(num_slots):
            heat = attn[0, slot].reshape(grid, grid).cpu().numpy()
            ax = axes[row, slot + 2]
            ax.imshow(rgb, alpha=0.35)
            ax.imshow(
                np.kron(heat, np.ones((rgb.shape[0] // grid, rgb.shape[1] // grid))),
                cmap="inferno", alpha=0.65, vmin=0.0, vmax=1.0,
            )
            ax.set_title(f"slot {slot}  ({heat.sum() / heat.size:.2f})", fontsize=7)
            ax.axis("off")

    fig.suptitle(
        f"Slot attention -- {num_slots} slots. Slot order is arbitrary and "
        f"differs per image (see models/slot_bottleneck.py).",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise Logical-JEPA results")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", default="grid",
                        choices=["grid", "masks", "latent", "slots", "all"])
    parser.add_argument("--slot-checkpoint", default=None,
                        help="slot bottleneck .pt from train_slots.py (--mode slots)")
    parser.add_argument("--defect", default=None,
                        help="restrict to good / logical_anomalies / structural_anomalies")
    parser.add_argument("--num", type=int, default=6, help="samples to draw")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("visualize")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    override = Config()
    for item in args.set:
        import yaml
        key, _, value = item.partition("=")
        override.set_path(key.strip(), yaml.safe_load(value.strip()))

    model, cfg, _ = load_model_from_checkpoint(args.checkpoint, device, override)
    category = cfg.get_path("data.category")
    root = cfg.get_path("data.root", "data/mvtec_loco")

    out_dir = args.out or os.path.join(
        cfg.get_path("experiment.output_dir", "outputs"),
        cfg.get_path("experiment.name", "run"), category, "figures",
    )

    modes = ["grid", "masks", "latent"] if args.mode == "all" else [args.mode]

    if "slots" in modes:
        if not args.slot_checkpoint:
            raise SystemExit(
                "--mode slots needs --slot-checkpoint (produced by train_slots.py)"
            )
        test_set = MVTecLOCO(root, category, "test", cfg.get_path("data.img_size", 256))
        indices = _pick_samples(test_set, args.num, args.defect)
        path = visualize_slots(
            model, args.slot_checkpoint, test_set, indices,
            os.path.join(out_dir, "slot_attention.png"), device,
        )
        logger.info(f"wrote {path}")

    if "masks" in modes:
        train_set = MVTecLOCO(root, category, "train", cfg.get_path("data.img_size", 256),
                              train_augment=False)
        path = visualize_masks(cfg, train_set, os.path.join(out_dir, "masking_strategies.png"),
                               grid_size=model.grid_size)
        logger.info(f"wrote {path}")

    if "grid" in modes or "latent" in modes:
        test_set = MVTecLOCO(root, category, "test", cfg.get_path("data.img_size", 256))

        if "grid" in modes:
            bank = build_sweep_bank(cfg, model, device)
            scorer = build_scorer(cfg, model, bank)

            # Load the threshold produced by evaluate.py, if it exists.
            calib_path = os.path.join(os.path.dirname(args.checkpoint), "calibration.json")
            if os.path.isfile(calib_path):
                import json

                from anomaly.scoring import Calibration
                with open(calib_path, encoding="utf-8") as handle:
                    scorer.calibration = Calibration.from_dict(json.load(handle))
                logger.info(f"loaded calibration from {calib_path}")
            else:
                logger.warning("no calibration.json found -- run evaluate.py first for "
                               "meaningful NORMAL/ANOMALOUS verdicts")

            indices = _pick_samples(test_set, args.num, args.defect)
            path = visualize_grid(
                model, scorer, test_set, indices,
                os.path.join(out_dir, f"anomaly_maps_{args.defect or 'mixed'}.png"),
                device,
                alpha=cfg.get_path("visualize.overlay_alpha", 0.5),
                cmap=cfg.get_path("visualize.colormap", "jet"),
            )
            logger.info(f"wrote {path}")

        if "latent" in modes:
            # Reuse the fitted per-position statistics so the region highlighted
            # here is the one the heatmap actually flagged.
            scale_stats = None
            calib_path = os.path.join(os.path.dirname(args.checkpoint), "calibration.json")
            if os.path.isfile(calib_path):
                import json

                from anomaly.scoring import Calibration
                with open(calib_path, encoding="utf-8") as handle:
                    scale_stats = Calibration.from_dict(json.load(handle)).scale_stats
            else:
                logger.warning("no calibration.json -- the latent figure will rank "
                               "regions by raw error, which favours positions that are "
                               "hard to predict on every image")

            indices = _pick_samples(test_set, min(args.num, 4), args.defect)
            path = visualize_latent(
                model, test_set, indices,
                os.path.join(out_dir, f"latent_discrepancy_{args.defect or 'mixed'}.png"),
                device, scale_stats=scale_stats,
            )
            logger.info(f"wrote {path}")


if __name__ == "__main__":
    main()
