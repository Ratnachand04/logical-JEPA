"""Diagnostics behind the few-shot findings (outputs/fewshot/SUMMARY.md).

Three questions, each answerable without training a new model:

``layers``  -- *Will a pretrained few-shot model work on this category at all?*
    No checkpoint needed. Patch features of K normal images from every probed
    DINO layer form a memory bank; each test image is scored by its patch
    farthest from the bank (PatchCore-style). If no layer separates the test
    set, no amount of JEPA training on frozen features will -- this takes a
    minute per category against hours for a study.

``stats <checkpoint>``  -- *Is the calibration the problem?*
    Re-scores the test set with the per-position statistics fitted on (a) the
    K images the predictor trained on, as evaluation does, and (b) normal
    images it never saw. Caches the expensive sweep once.

``memory <checkpoint>``  -- *Does a memory-bank channel on the JEPA's own
    teacher features help?* Fuses it with the JEPA error. The bank holds the
    same K training images; its statistics use leave-one-image-out distances,
    so no extra normal image is used.

    py -3.12 scripts/diagnose_fewshot.py layers --category pushpins --k 16
    py -3.12 scripts/diagnose_fewshot.py stats  --checkpoint checkpoints/fewshot/fs_dino_k16/seed_0/pushpins/final.pt
    py -3.12 scripts/diagnose_fewshot.py memory --checkpoint <same>
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from anomaly.anomaly_map import fuse_scales, upsample_map  # noqa: E402
from anomaly.metrics import image_auroc  # noqa: E402
from anomaly.scoring import _MomentAccumulator, aggregate_score, build_scorer  # noqa: E402
from datasets.mvtec_loco import MVTecLOCO, select_few_shot  # noqa: E402
from utils.config import Config  # noqa: E402

PROBE_LAYERS = (3, 6, 8, 9, 10, 11, 12)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
class TestSplit:
    """Labels of a test split, and AUROC overall / logical / structural."""

    def __init__(self, dataset: MVTecLOCO):
        self.labels = np.array([s.label for s in dataset.samples])
        defects = np.array([s.defect_type for s in dataset.samples])
        good = defects == "good"
        self.logical = good | (defects == "logical_anomalies")
        self.structural = good | (defects == "structural_anomalies")

    def line(self, name: str, scores: np.ndarray) -> str:
        y = self.labels
        return (f"{name:38s} image {image_auroc(scores, y):.4f}  "
                f"logical {image_auroc(scores[self.logical], y[self.logical]):.4f}  "
                f"structural {image_auroc(scores[self.structural], y[self.structural]):.4f}")


def _loader(dataset, batch_size=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _fit_stats(grids: dict) -> dict:
    acc = _MomentAccumulator()
    acc.add(grids)
    return acc.finish(1e-4)


def _load(checkpoint: str, root: str | None, device):
    from evaluate import build_sweep_bank, load_model_from_checkpoint

    override = Config({"data": {"num_workers": 0}})
    if root:
        override.set_path("data.root", root)
    model, cfg, _ = load_model_from_checkpoint(checkpoint, device, override)
    k = cfg.get_path("data.num_train_images", None)
    if k is None:
        raise SystemExit("This diagnostic is for few-shot checkpoints (data.num_train_images).")
    scorer = build_scorer(cfg, model, build_sweep_bank(cfg, model, device))
    return model, cfg, scorer, int(k), int(cfg.get_path("data.subset_seed", 0))


class _Scorer:
    """Score cached grids with any statistics, exactly as the scorer would."""

    def __init__(self, scorer, device):
        self.s, self.device = scorer, device

    def fused(self, grids, stats, deviation=None):
        return fuse_scales(grids, mode=self.s.fusion, weights=self.s.weights,
                           normalize="global", scale_stats=stats,
                           deviation=deviation or self.s.deviation)

    def image_scores(self, fused):
        out = []
        for i in range(0, fused.size(0), 16):
            maps = upsample_map(fused[i:i + 16].to(self.device),
                                out_size=self.s.out_size, sigma=self.s.sigma)
            out.append(aggregate_score(maps, self.s.aggregation,
                                       self.s.top_k_ratio).float().cpu().numpy())
        return np.concatenate(out)


@torch.no_grad()
def _sweep(scorer, dataset, device, with_teacher=False):
    """Raw per-scale JEPA error grids (and optionally teacher features)."""
    grids, feats = [], []
    for batch in _loader(dataset):
        images = batch["image"].to(device)
        g, _ = scorer._grids(images)
        grids.append({w: t.float() for w, t in g.items()})
        if with_teacher:
            feats.append(F.normalize(scorer.model.encode_targets(images).float(), dim=-1))
    merged = {w: torch.cat([o[w] for o in grids]) for w in grids[0]}
    return (merged, torch.cat(feats)) if with_teacher else merged


# --------------------------------------------------------------------------- #
# layers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cmd_layers(args, device):
    from models import pretrained as P
    from models.context_encoder import build_context_encoder

    enc = build_context_encoder({"pretrained": args.backbone, "img_size": args.img_size})
    enc = enc.to(device).eval()
    P.load_pretrained_into(enc, args.backbone)

    train = MVTecLOCO(args.root, args.category, "train", args.img_size, train_augment=False)
    train.samples = select_few_shot(train.samples, args.k, args.subset_seed)
    test = MVTecLOCO(args.root, args.category, "test", args.img_size)
    split = TestSplit(test)

    def features(dataset):
        per_layer = {layer: [] for layer in PROBE_LAYERS if layer <= len(enc.blocks)}
        for batch in _loader(dataset, 16):
            x = enc.embed_patches(batch["image"].to(device))
            if enc.cls_embed is not None:
                x = torch.cat([enc.cls_embed.expand(x.size(0), -1, -1), x], 1)
            for i, block in enumerate(enc.blocks, start=1):
                x = block(x)
                if i in per_layer:
                    h = enc.norm(x) if i == len(enc.blocks) else F.layer_norm(x, x.shape[-1:])
                    per_layer[i].append(h[:, 1:].float() if enc.cls_embed is not None
                                        else h.float())
        return {layer: torch.cat(v) for layer, v in per_layer.items()}

    bank_feats, test_feats = features(train), features(test)
    print(f"{args.category}: {args.backbone} @ {args.img_size}px, {len(train)}-shot "
          f"(subset_seed {args.subset_seed}), {len(test)} test images")
    for layer, bank in bank_feats.items():
        bank = bank.reshape(-1, bank.shape[-1])
        scores = np.array([torch.cdist(img, bank).min(1).values.max().item()
                           for img in test_feats[layer]])
        print(split.line(f"layer {layer:2d}  nearest-neighbour", scores))


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def cmd_stats(args, device):
    _model, cfg, scorer, k, seed = _load(args.checkpoint, args.root, device)
    root, cat, size = cfg.get_path("data.root"), cfg.get_path("data.category"), cfg.get_path("data.img_size")

    train = MVTecLOCO(root, cat, "train", size, train_augment=False)
    test = MVTecLOCO(root, cat, "test", size)
    seen = {s.image_path for s in select_few_shot(train.samples, k, seed)}
    seen_idx = [i for i, s in enumerate(train.samples) if s.image_path in seen]
    unseen_idx = [i for i, s in enumerate(train.samples) if s.image_path not in seen]

    grids_train = _sweep(scorer, train, device)
    grids_test = _sweep(scorer, test, device)
    split, sc = TestSplit(test), _Scorer(scorer, device)
    rng = np.random.default_rng(0)
    probe = sorted(rng.choice(unseen_idx, min(64, len(unseen_idx)), replace=False).tolist())

    print(f"{cat}: {k}-shot checkpoint (subset_seed {seed}), {len(test)} test images")
    variants = [(f"stats on the {k} images it trained on", seen_idx),
                (f"stats on {k} images it never saw", probe[:k]),
                (f"stats on {len(unseen_idx)} images it never saw", unseen_idx)]
    for name, idx in variants:
        stats = _fit_stats({w: t[idx] for w, t in grids_train.items()})
        held_out = [i for i in probe if i not in set(idx)] or probe
        normal = sc.image_scores(sc.fused({w: t[held_out] for w, t in grids_train.items()}, stats))
        test_scores = sc.image_scores(sc.fused(grids_test, stats))
        print(split.line(name, test_scores), f" held-out normal score {normal.mean():6.2f}")


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #
def cmd_memory(args, device):
    model, cfg, scorer, k, seed = _load(args.checkpoint, args.root, device)
    root, cat, size = cfg.get_path("data.root"), cfg.get_path("data.category"), cfg.get_path("data.img_size")

    train = MVTecLOCO(root, cat, "train", size, train_augment=False)
    train.samples = select_few_shot(train.samples, k, seed)
    test = MVTecLOCO(root, cat, "test", size)

    grids_train, feats_train = _sweep(scorer, train, device, with_teacher=True)
    grids_test, feats_test = _sweep(scorer, test, device, with_teacher=True)
    g = model.grid_size
    bank = feats_train.reshape(-1, feats_train.shape[-1])
    owner = torch.arange(len(train), device=device).repeat_interleave(g * g)

    def nn_grid(feats, leave_out=False):
        out = []
        for i, img in enumerate(feats):
            d = torch.cdist(img, bank)
            if leave_out:
                d[:, owner == i] = float("inf")
            out.append(d.min(1).values.reshape(g, g))
        return torch.stack(out)

    mem_train, mem_test = nn_grid(feats_train, leave_out=True), nn_grid(feats_test)
    jepa_stats, mem_stats = _fit_stats(grids_train), _fit_stats({0: mem_train})
    split, sc = TestSplit(test), _Scorer(scorer, device)

    jepa = sc.fused(grids_test, jepa_stats)
    memory = sc.fused({0: mem_test}, mem_stats, deviation="signed")
    print(f"{cat}: {k}-shot checkpoint (subset_seed {seed}), {len(test)} test images")
    print(split.line("JEPA error (as evaluated)", sc.image_scores(jepa)))
    print(split.line("memory bank alone", sc.image_scores(memory)))
    for w in (0.5, 1.0, 2.0):
        print(split.line(f"JEPA + {w} x memory bank", sc.image_scores(jepa + w * memory)))


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    layers = sub.add_parser("layers", help="probe pretrained features, no training")
    layers.add_argument("--category", default="pushpins")
    layers.add_argument("--root", default="data/mvtec_loco_256")
    layers.add_argument("--k", type=int, default=16)
    layers.add_argument("--subset-seed", type=int, default=0)
    layers.add_argument("--img-size", type=int, default=256)
    layers.add_argument("--backbone", default="dino_vits16")

    for name in ("stats", "memory"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--root", default=None, help="override data.root, e.g. a moved dataset")

    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    {"layers": cmd_layers, "stats": cmd_stats, "memory": cmd_memory}[args.command](args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
