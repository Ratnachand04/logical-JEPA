"""Train Logical-JEPA on the normal images of one MVTec LOCO category.

Training is fully self-supervised: only ``train/good`` is ever read, and no
anomaly label is used at any point.

Usage::

    py -3.12 train.py --config configs/loco.yaml
    py -3.12 train.py --config configs/loco.yaml --category juice_bottle
    py -3.12 train.py --config configs/loco.yaml --set masking.strategy=large train.epochs=80
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
from tqdm import tqdm

from datasets.mvtec_loco import build_dataloaders
from masking import build_mask_generator
from models import build_model
from utils.config import load_config, save_config
from utils.logging_utils import MetricTracker, get_logger, save_json
from utils.seed import seed_everything


# --------------------------------------------------------------------------- #
# Optimisation schedules
# --------------------------------------------------------------------------- #
def build_param_groups(model, weight_decay: float) -> list[dict]:
    """Split parameters into decayed and non-decayed groups.

    Biases, LayerNorm gains and the predictor's mask token are excluded from
    weight decay. Decaying a 1-D gain or the single mask token pulls it towards
    zero for no regularisation benefit and measurably hurts JEPA training.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "mask_token" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay, "wd_scale": 1.0},
        {"params": no_decay, "weight_decay": 0.0, "wd_scale": 0.0},
    ]


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def wd_at(step: int, total_steps: int, base_wd: float, final_wd: float) -> float:
    """Cosine *increase* of weight decay, following I-JEPA.

    Light regularisation early lets the representation form; heavier decay later
    keeps it from drifting as the EMA teacher slows down.
    """
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return final_wd + (base_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_schedules(optimizer, lr: float, wd: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
        if group.get("wd_scale", 0.0) > 0:
            group["weight_decay"] = wd


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model, optimizer, cfg, epoch: int, step: int, stats: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "config": cfg.to_dict(),
            "epoch": epoch,
            "step": step,
            "stats": stats,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, device="cpu") -> dict:
    """Restore a checkpoint. Returns the payload minus the tensors."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    return {k: v for k, v in payload.items() if k not in ("model", "optimizer")}


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(cfg, logger) -> str:
    """Run training for one category. Returns the final checkpoint path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    category = cfg.get_path("data.category")

    seed_everything(cfg.get_path("experiment.seed", 42),
                    cfg.get_path("experiment.deterministic", False))

    # ---- data -------------------------------------------------------- #
    train_loader, _val_loader, _test_loader = build_dataloaders(cfg, category)
    logger.info(f"Category '{category}': {len(train_loader.dataset)} normal training images")

    # ---- model ------------------------------------------------------- #
    model = build_model(cfg).to(device)
    logger.info(model.describe())

    mask_gen = build_mask_generator(
        cfg.masking, grid_size=model.grid_size, seed=cfg.get_path("experiment.seed", 42)
    )
    logger.info(f"Masking: {mask_gen.describe()}")

    # ---- optimiser --------------------------------------------------- #
    base_lr = cfg.get_path("train.lr", 1e-3)
    min_lr = cfg.get_path("train.min_lr", 1e-6)
    base_wd = cfg.get_path("train.weight_decay", 0.04)
    final_wd = cfg.get_path("train.final_weight_decay", 0.4)
    epochs = cfg.get_path("train.epochs", 150)
    grad_clip = cfg.get_path("train.grad_clip", 3.0)
    masks_per_batch = cfg.get_path("train.masks_per_batch", 1)

    # Only the student and predictor are optimised; the teacher is EMA-driven.
    trainable = torch.nn.ModuleList([model.context_encoder, model.predictor])
    optimizer = torch.optim.AdamW(build_param_groups(trainable, base_wd), lr=base_lr)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * cfg.get_path("train.warmup_epochs", 15)

    use_amp = cfg.get_path("train.amp", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    # GradScaler is only needed for fp16; bf16 has the dynamic range of fp32.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    logger.info(
        f"Device {device} | AMP {'off' if not use_amp else amp_dtype} | "
        f"{total_steps} steps ({epochs} epochs x {steps_per_epoch})"
    )

    # ---- run --------------------------------------------------------- #
    out_dir = os.path.join(cfg.get_path("experiment.output_dir", "outputs"),
                           cfg.get_path("experiment.name", "run"), category)
    ckpt_dir = os.path.join(cfg.get_path("experiment.checkpoint_dir", "checkpoints"),
                            cfg.get_path("experiment.name", "run"), category)
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    history: list[dict] = []
    global_step = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        tracker = MetricTracker()
        progress = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)

        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)

            lr = lr_at(global_step, total_steps, warmup_steps, base_lr, min_lr)
            wd = wd_at(global_step, total_steps, base_wd, final_wd)
            apply_schedules(optimizer, lr, wd)

            # Curriculum masking needs to know how far along training is.
            if hasattr(mask_gen, "set_progress"):
                mask_gen.set_progress(global_step / max(total_steps, 1))

            optimizer.zero_grad(set_to_none=True)

            # Several independent mask draws per image batch give a lower
            # variance gradient without paying for another data load.
            step_stats: dict = {}
            for _ in range(masks_per_batch):
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    loss, stats = model(images, mask_gen())
                    loss = loss / masks_per_batch

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                step_stats = stats

            if grad_clip and grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            momentum = model.update_target_encoder(global_step, total_steps)
            global_step += 1

            tracker.update(
                loss=step_stats["loss"],
                cos_sim=step_stats["cos_sim"],
                target_std=step_stats["target_std"],
                masked=step_stats["masked_ratio"],
            )
            progress.set_postfix(
                loss=f"{tracker['loss'].smooth:.4f}",
                cos=f"{tracker['cos_sim'].smooth:.3f}",
                lr=f"{lr:.2e}",
                m=f"{momentum:.4f}",
            )

        epoch_stats = tracker.averages()
        epoch_stats.update(epoch=epoch + 1, lr=lr, wd=wd, momentum=momentum,
                           elapsed=time.time() - start)
        history.append(epoch_stats)

        if (epoch + 1) % cfg.get_path("train.log_interval", 20) == 0 or epoch == 0:
            logger.info(
                f"epoch {epoch+1}/{epochs}  loss={epoch_stats['loss']:.4f}  "
                f"cos_sim={epoch_stats['cos_sim']:.4f}  "
                f"target_std={epoch_stats['target_std']:.4f}  "
                f"lr={lr:.2e}  m={momentum:.5f}"
            )

        # target_std collapsing to ~0 means the encoder found the constant
        # solution and the run is dead -- worth saying out loud, not silently
        # producing a useless checkpoint.
        if epoch_stats["target_std"] < 0.05:
            logger.warning(
                f"target_std={epoch_stats['target_std']:.4f} -- representation may be "
                f"collapsing. Consider raising ema.base_momentum or lowering train.lr."
            )

        save_interval = cfg.get_path("train.save_interval", 25)
        if save_interval and (epoch + 1) % save_interval == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pt"),
                            model, optimizer, cfg, epoch + 1, global_step, epoch_stats)

    final_path = os.path.join(ckpt_dir, "final.pt")
    save_checkpoint(final_path, model, optimizer, cfg, epochs, global_step,
                    history[-1] if history else {})
    save_json({"history": history, "config": cfg.to_dict()},
              os.path.join(out_dir, "train_history.json"))

    logger.info(f"Training finished in {(time.time()-start)/60:.1f} min -> {final_path}")
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Logical-JEPA on normal images")
    parser.add_argument("--config", default="configs/loco.yaml", help="YAML config path")
    parser.add_argument("--category", default=None, help="override data.category")
    parser.add_argument("--name", default=None, help="override experiment.name")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="dotted config overrides, e.g. train.epochs=50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)

    if args.category:
        cfg.set_path("data.category", args.category)
    if args.name:
        cfg.set_path("experiment.name", args.name)

    log_path = os.path.join(
        cfg.get_path("experiment.output_dir", "outputs"),
        cfg.get_path("experiment.name", "run"),
        cfg.get_path("data.category", "category"),
        "train.log",
    )
    logger = get_logger("train", log_path)
    logger.info(f"Config: {args.config}  overrides: {args.set or 'none'}")

    train(cfg, logger)


if __name__ == "__main__":
    main()
