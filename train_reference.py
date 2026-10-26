"""Train the reference-conditioned Logical-JEPA on normal pairs from many products.

    py -3.12 train_reference.py --config configs/reference.yaml
    py -3.12 train_reference.py --config configs/reference.yaml \
        --set data.holdout=[juice_bottle,pushpins,candle,pcb1,macaroni1] experiment.name=reference_holdout

Only the predictor trains; the DINO encoder is frozen. ``data.holdout`` names
products that are kept out of training entirely, so they can be used to measure
how the model does on objects it has never seen.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.pairs import ReferencePairs, resolve_sources
from masking import build_mask_generator
from models.reference_jepa import build_reference_model
from train import apply_schedules, build_param_groups, lr_at, wd_at
from utils.config import load_config, save_config
from utils.logging_utils import get_logger, save_json
from utils.seed import seed_everything


def train_reference(cfg, logger) -> str:
    seed = seed_everything(cfg.get_path("experiment.seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = cfg.get_path("experiment.name", "reference")
    ckpt_dir = os.path.join(cfg.get_path("experiment.checkpoint_dir", "checkpoints"), name)
    out_dir = os.path.join(cfg.get_path("experiment.output_dir", "outputs"), name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    holdout = list(cfg.get_path("data.holdout", []) or [])
    sources = resolve_sources(cfg.get_path("data.sources"), exclude=holdout)
    dataset = ReferencePairs(
        sources,
        img_size=cfg.get_path("data.img_size", 256),
        samples_per_epoch=cfg.get_path("train.samples_per_epoch", 4096),
        num_references=cfg.get_path("data.num_references", 1),
        same_image_prob=cfg.get_path("data.same_image_prob", 0.2),
        split="train",
        augment=dict(cfg.get_path("data.augment", {}) or {}),
        seed=seed,
    )
    batch = cfg.get_path("train.batch_size", 16)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, drop_last=True,
                        num_workers=cfg.get_path("data.num_workers", 0),
                        pin_memory=torch.cuda.is_available())
    logger.info(f"Training pairs from {dataset.describe()}; held out: {holdout or 'none'}")
    logger.info("Products: " + ", ".join(s.name for s in dataset.sources))

    model = build_reference_model(cfg).to(device)
    logger.info(model.describe())
    mask_gen = build_mask_generator(cfg.get("masking", {}), model.grid_size, seed=seed)

    epochs = cfg.get_path("train.epochs", 30)
    total = epochs * len(loader)
    warmup = int(cfg.get_path("train.warmup_epochs", 2) * len(loader))
    base_lr = cfg.get_path("train.lr", 5e-4)
    min_lr = cfg.get_path("train.min_lr", 1e-6)
    wd0 = cfg.get_path("train.weight_decay", 0.04)
    wd1 = cfg.get_path("train.final_weight_decay", 0.4)
    clip = cfg.get_path("train.grad_clip", 3.0)
    optimizer = torch.optim.AdamW(build_param_groups(model.predictor, wd0), lr=base_lr,
                                  betas=(0.9, 0.95))
    use_amp = device.type == "cuda" and cfg.get_path("train.amp", True)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    logger.info(f"{epochs} epochs x {len(loader)} steps = {total} steps, batch {batch}, "
                f"AMP {amp_dtype if use_amp else 'off'}")

    history, step, start = [], 0, time.time()
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.train()
        sums = {"loss": 0.0, "cos_sim": 0.0}
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch_data in bar:
            apply_schedules(optimizer, lr_at(step, total, warmup, base_lr, min_lr),
                            wd_at(step, total, wd0, wd1))
            current = batch_data["current"].to(device, non_blocking=True)
            refs = batch_data["references"].to(device, non_blocking=True)
            spec = mask_gen()
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss, stats = model(current, refs, spec)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), clip)
            optimizer.step()
            step += 1
            sums["loss"] += stats["loss"]
            sums["cos_sim"] += stats["cos_sim"]
            bar.set_postfix(loss=f"{stats['loss']:.4f}", cos=f"{stats['cos_sim']:.3f}")
        n = max(len(loader), 1)
        record = {"epoch": epoch + 1, "loss": sums["loss"] / n, "cos_sim": sums["cos_sim"] / n,
                  "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)
        if epoch == 0 or (epoch + 1) % cfg.get_path("train.log_interval", 5) == 0 or epoch + 1 == epochs:
            logger.info(f"epoch {epoch + 1}/{epochs}  loss={record['loss']:.4f}  "
                        f"cos_sim={record['cos_sim']:.4f}  lr={record['lr']:.2e}")

    path = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "products": [s.name for s in dataset.sources],
        "holdout": holdout,
        "steps": step,
    }, path)
    save_json(history, os.path.join(out_dir, "train_history.json"))
    logger.info(f"Training finished in {(time.time() - start) / 60:.1f} min -> {path}")
    return path


def load_reference_model(checkpoint: str, device) -> tuple:
    """Rebuild a trained model from its checkpoint, without downloading weights."""
    from utils.config import Config

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(payload["config"])
    model = build_reference_model(cfg, load_pretrained=False)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), cfg, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/reference.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    logger = get_logger("train_reference", os.path.join(
        cfg.get_path("experiment.output_dir", "outputs"),
        cfg.get_path("experiment.name", "reference"), "train.log"))
    train_reference(cfg, logger)


if __name__ == "__main__":
    main()
