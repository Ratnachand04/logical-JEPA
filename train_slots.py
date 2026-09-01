"""Isolation gate for the slot bottleneck (Phase 3a).

Trains the bottleneck **alone**, as

    patch tokens --> slots --> reconstructed patch tokens

with no predictor, no masking and no anomaly objective. Nothing downstream may
be built until this passes, for a concrete reason: if slots are not routing
sensibly, no later loss can repair that, and wiring it into the predictor first
means debugging two broken things at once.

Tokens come from the frozen target encoder of an existing Logical-JEPA
checkpoint, so the bottleneck is validated on exactly the representation it
will have to compress in the full model -- not on raw pixels.

Pass / fail criteria
--------------------
``entropy_ratio``
    Mean entropy of each token's distribution over slots, divided by
    ``log(num_slots)``. 1.0 means every token is spread evenly across all slots
    -- the bottleneck has collapsed to an averaging layer and is not
    segmenting. Below ~0.75 indicates tokens are committing to slots.

``dead_slots``
    Slots receiving a negligible share of attention. If most slots are dead the
    effective capacity is far below ``num_slots`` and any count read off the
    set is meaningless.

``recon_r2``
    How much of the token variance survives the bottleneck. A slot set that
    cannot reconstruct the tokens has thrown away the content the predictor
    needs.

Usage::

    py -3.12 train_slots.py --checkpoint checkpoints/<name>/<cat>/final.pt
    py -3.12 train_slots.py --checkpoint <path> --num-slots 4,6,8,12 --epochs 40
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.mvtec_loco import build_normal_loader
from models.slot_bottleneck import SlotBottleneck, slot_attention_entropy, slot_usage
from utils.logging_utils import get_logger, save_json
from utils.seed import seed_everything

# Gate thresholds. Deliberately permissive -- they are meant to catch a
# collapsed bottleneck, not to certify a good one, which is what the rendered
# attention maps are for.
ENTROPY_RATIO_MAX = 0.75
DEAD_SLOT_SHARE_MAX = 0.5
RECON_R2_MIN = 0.30
DEAD_SLOT_THRESHOLD = 0.02          # share of attention mass below which a slot is dead


@torch.no_grad()
def _collect_tokens(model, loader, device, max_batches: int | None = None):
    """Cache frozen target-encoder tokens so epochs do not re-run the encoder."""
    chunks = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        chunks.append(model.encode_targets(batch["image"].to(device)).cpu())
    return torch.cat(chunks, dim=0)


def evaluate_gate(bottleneck, tokens, pos_embed, device, batch_size: int = 16) -> dict:
    """Run the gate diagnostics over a token set."""
    bottleneck.eval()

    entropies, recon_err, token_var, usages = [], [], [], []

    with torch.no_grad():
        for start in range(0, tokens.size(0), batch_size):
            batch = tokens[start : start + batch_size].to(device)
            recon, _slots, attn = bottleneck(batch, pos_embed, return_attn=True)

            entropies.append(slot_attention_entropy(attn).item())
            usages.append(slot_usage(attn).mean(dim=0).cpu())
            recon_err.append(F.mse_loss(recon, batch, reduction="mean").item())
            token_var.append(batch.var(dim=(0, 1), unbiased=False).mean().item())

    usage = torch.stack(usages).mean(dim=0)
    entropy = sum(entropies) / len(entropies)
    mse = sum(recon_err) / len(recon_err)
    variance = sum(token_var) / len(token_var)

    dead = int((usage < DEAD_SLOT_THRESHOLD).sum())
    n_slots = bottleneck.num_slots

    return {
        "entropy": entropy,
        "entropy_ratio": entropy / math.log(n_slots),
        "recon_mse": mse,
        "recon_r2": 1.0 - mse / max(variance, 1e-9),
        "dead_slots": dead,
        "dead_slot_share": dead / n_slots,
        "slot_usage": [round(float(u), 4) for u in usage],
        "num_slots": n_slots,
    }


def gate_verdict(metrics: dict) -> tuple[bool, list[str]]:
    """Apply the pass/fail criteria. Returns ``(passed, reasons)``."""
    reasons = []

    if metrics["entropy_ratio"] > ENTROPY_RATIO_MAX:
        reasons.append(
            f"attention is near-uniform over slots "
            f"(entropy ratio {metrics['entropy_ratio']:.3f} > {ENTROPY_RATIO_MAX}) -- "
            f"the bottleneck is averaging, not segmenting"
        )
    if metrics["dead_slot_share"] > DEAD_SLOT_SHARE_MAX:
        reasons.append(
            f"{metrics['dead_slots']}/{metrics['num_slots']} slots are unused -- "
            f"effective slot count is far below the configured one"
        )
    if metrics["recon_r2"] < RECON_R2_MIN:
        reasons.append(
            f"reconstruction R2 {metrics['recon_r2']:.3f} < {RECON_R2_MIN} -- "
            f"the slot set does not retain enough token content"
        )

    return (not reasons), reasons


def train_bottleneck(tokens, pos_embed, num_slots, device, logger,
                     epochs=30, batch_size=16, lr=4e-4, iters=3, slot_dim=None):
    """Train one bottleneck to reconstruct the cached tokens."""
    token_dim = tokens.size(-1)
    bottleneck = SlotBottleneck(
        token_dim=token_dim, slot_dim=slot_dim or token_dim,
        num_slots=num_slots, iters=iters,
    ).to(device)
    logger.info(bottleneck.describe())

    optimizer = torch.optim.AdamW(bottleneck.parameters(), lr=lr, weight_decay=1e-4)
    n = tokens.size(0)
    total_steps = max(epochs * math.ceil(n / batch_size), 1)
    step = 0

    for epoch in range(epochs):
        bottleneck.train()
        order = torch.randperm(n)
        running, count = 0.0, 0

        for start in tqdm(range(0, n, batch_size),
                          desc=f"slots={num_slots} epoch {epoch+1}/{epochs}", leave=False):
            batch = tokens[order[start : start + batch_size]].to(device)

            # Cosine schedule, matching the main training loop's shape.
            for group in optimizer.param_groups:
                group["lr"] = lr * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))

            recon, _ = bottleneck(batch, pos_embed)
            loss = F.mse_loss(recon, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bottleneck.parameters(), 1.0)
            optimizer.step()

            running += float(loss.detach())
            count += 1
            step += 1

        if (epoch + 1) % max(epochs // 5, 1) == 0 or epoch == 0:
            logger.info(f"  slots={num_slots} epoch {epoch+1}/{epochs} "
                        f"recon_mse={running/max(count,1):.5f}")

    return bottleneck


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slot bottleneck isolation gate")
    parser.add_argument("--checkpoint", required=True,
                        help="trained Logical-JEPA checkpoint supplying the tokens")
    parser.add_argument("--num-slots", default="4,6,8,12",
                        help="comma-separated slot counts to sweep")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from evaluate import load_model_from_checkpoint
    from utils.config import Config

    override = Config()
    for item in args.set:
        import yaml
        key, _, value = item.partition("=")
        override.set_path(key.strip(), yaml.safe_load(value.strip()))

    model, cfg, _ = load_model_from_checkpoint(args.checkpoint, device, override)
    category = cfg.get_path("data.category")
    seed_everything(cfg.get_path("experiment.seed", 42))

    out_dir = args.out or os.path.join(
        cfg.get_path("experiment.output_dir", "outputs"), "phase3a_slots", category
    )
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger("train_slots", os.path.join(out_dir, "slots.log"))

    logger.info(f"Isolation gate for the slot bottleneck -- category '{category}'")
    logger.info(f"Tokens from the frozen target encoder of {args.checkpoint}")

    loader = build_normal_loader(cfg, category, "train", num_workers=0)
    tokens = _collect_tokens(model, loader, device)
    logger.info(f"Cached {tokens.size(0)} images x {tokens.size(1)} tokens "
                f"x {tokens.size(2)} dims")

    pos_embed = model.context_encoder.pos_embed.to(device)
    if pos_embed.size(-1) != tokens.size(-1):
        raise RuntimeError("positional table and token width disagree")

    results = {}
    for num_slots in [int(s) for s in args.num_slots.replace(",", " ").split()]:
        logger.info(f"\n{'='*60}\nnum_slots = {num_slots}\n{'='*60}")

        bottleneck = train_bottleneck(
            tokens, pos_embed, num_slots, device, logger,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, iters=args.iters,
        )
        metrics = evaluate_gate(bottleneck, tokens, pos_embed, device, args.batch_size)
        passed, reasons = gate_verdict(metrics)
        metrics["passed"] = passed
        metrics["reasons"] = reasons
        results[num_slots] = metrics

        logger.info(
            f"  entropy_ratio={metrics['entropy_ratio']:.3f}  "
            f"recon_r2={metrics['recon_r2']:.3f}  "
            f"dead_slots={metrics['dead_slots']}/{num_slots}"
        )
        logger.info(f"  usage: {metrics['slot_usage']}")
        logger.info(f"  GATE: {'PASS' if passed else 'FAIL'}")
        for reason in reasons:
            logger.info(f"    - {reason}")

        torch.save(
            {"model": bottleneck.state_dict(), "num_slots": num_slots,
             "token_dim": tokens.size(-1), "metrics": metrics},
            os.path.join(out_dir, f"slots_{num_slots}.pt"),
        )

    # ---- summary ------------------------------------------------------ #
    logger.info("\n" + "=" * 60 + "\nISOLATION GATE SUMMARY\n" + "=" * 60)
    logger.info(f"{'slots':>6s} {'entropy_ratio':>14s} {'recon_r2':>10s} "
                f"{'dead':>6s} {'gate':>6s}")
    for num_slots, m in sorted(results.items()):
        logger.info(
            f"{num_slots:6d} {m['entropy_ratio']:14.3f} {m['recon_r2']:10.3f} "
            f"{m['dead_slots']:6d} {'PASS' if m['passed'] else 'FAIL':>6s}"
        )

    passing = [s for s, m in results.items() if m["passed"]]
    if passing:
        # Reconstruction R2 saturates near 0.94 for every workable slot count,
        # so it cannot discriminate. Entropy can: it measures how decisively
        # tokens are assigned, which is the property the downstream cardinality
        # question actually depends on.
        best = min(passing, key=lambda s: results[s]["entropy_ratio"])
        logger.info(
            f"\nBest passing configuration: num_slots={best} "
            f"(entropy_ratio {results[best]['entropy_ratio']:.3f}, "
            f"recon_r2 {results[best]['recon_r2']:.3f})"
        )
        logger.info(
            "  Selected on entropy: reconstruction R2 is saturated across all "
            "passing counts and does not discriminate between them."
        )
        logger.info("Next: render the attention maps with "
                    "`visualize.py --mode slots` before wiring into the predictor.")
    else:
        logger.warning(
            "\nNO configuration passed the gate. Per the phase plan, STOP here: "
            "do not wire the bottleneck into the predictor. Report the failure and "
            "the diagnostics above."
        )

    save_json({str(k): v for k, v in results.items()},
              os.path.join(out_dir, "gate_results.json"))
    logger.info(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
