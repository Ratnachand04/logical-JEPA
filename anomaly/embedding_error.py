"""Latent discrepancy between predicted and observed embeddings.

The same quantity plays two roles in Logical-JEPA:

* during **training** it is the loss -- minimise the gap between what the
  predictor imagines for a hidden region and what the teacher actually encodes
  there;
* during **inference** it is the anomaly score -- a region whose true content
  cannot be predicted from its context is, by construction, a region that
  violates the structure the model learned from normal images.

Keeping both in one module guarantees the score is measured with exactly the
same function the model was optimised against.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Distances available for both the loss and the anomaly score.
DISTANCES = ("cosine", "l2", "l1", "smooth_l1", "combined")


def cosine_distance(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - cos(pred, target), elementwise over the last dim. Returns (..., K).

    Scale-invariant, which is why it is the default: the teacher's embedding
    norm varies with local contrast, and we want the *direction* of the
    representation (what is there) rather than its magnitude.
    """
    return 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=eps)


def l2_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Euclidean distance per token. Returns (..., K)."""
    return torch.linalg.vector_norm(pred - target, ord=2, dim=-1)


def l1_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute deviation per token. Returns (..., K)."""
    return (pred - target).abs().mean(dim=-1)


def smooth_l1_distance(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Huber distance per token -- less sensitive to a few extreme channels."""
    return F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(dim=-1)


def combined_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    l2_scale: float | None = None,
) -> torch.Tensor:
    """``alpha * cosine + (1 - alpha) * l2``.

    The two terms answer different questions -- cosine asks *what* is there, L2
    also reacts to *how strongly*. Because their ranges differ (cosine is in
    [0, 2], L2 is unbounded), the L2 term is divided by ``l2_scale``; when that
    is ``None`` the batch mean is used, making the mixture scale-free.

    This is Method C of the scoring ablation.
    """
    cos = cosine_distance(pred, target)
    l2 = l2_distance(pred, target)

    if l2_scale is None:
        l2_scale = l2.detach().mean().clamp_min(1e-6)
    l2 = l2 / l2_scale

    return alpha * cos + (1.0 - alpha) * l2


def patch_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str = "cosine",
    alpha: float = 0.7,
    beta: float = 1.0,
    l2_scale: float | None = None,
) -> torch.Tensor:
    """Dispatch to a named distance. ``(B, K, D) x (B, K, D) -> (B, K)``."""
    if kind == "cosine":
        return cosine_distance(pred, target)
    if kind == "l2":
        return l2_distance(pred, target)
    if kind == "l1":
        return l1_distance(pred, target)
    if kind == "smooth_l1":
        return smooth_l1_distance(pred, target, beta=beta)
    if kind == "combined":
        return combined_distance(pred, target, alpha=alpha, l2_scale=l2_scale)
    raise ValueError(f"Unknown distance '{kind}'. Available: {DISTANCES}")


def jepa_loss(
    pred_blocks: list[torch.Tensor],
    target_blocks: list[torch.Tensor],
    kind: str = "cosine",
    alpha: float = 0.7,
    beta: float = 1.0,
    reduction: str = "per_patch",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean prediction error over the target blocks.

    Blocks are weighted by their patch count rather than averaged
    block-by-block. This matters for the multi-scale strategy: a 6x6 block holds
    nine times the patches of a 2x2 one, and per-block averaging would silently
    down-weight the large-scale (logical) signal it exists to provide.

    Args:
        pred_blocks: list of (B, K_m, D) predictions.
        target_blocks: list of (B, K_m, D) teacher embeddings.
        kind: distance name, see :data:`DISTANCES`.
        alpha, beta: parameters of ``combined`` / ``smooth_l1``.
        reduction: how blocks are combined.

            ``per_patch`` (default, preserves the historical behaviour)
                Every target patch contributes equally, so a block's influence
                scales with its area.
            ``per_block``
                Average within each block first, then across blocks, so every
                block counts once regardless of size.

            This choice is not cosmetic. Under ``per_patch`` a single 6x6 block
            (36 patches) outweighs three 2x2-3x3 blocks (~19 patches combined)
            by roughly 2:1, which means the ``multiscale`` masking strategy
            effectively trains on its large-scale objective alone -- the leading
            explanation for that arm losing Study 1. ``per_block`` is the
            controlled test of that explanation.

    Returns:
        ``(loss, stats)`` where ``stats`` holds detached diagnostics -- notably
        ``cos_sim``, which should climb towards 1 as training converges, and is
        the quickest way to spot representation collapse. ``stats`` also
        reports ``loss_small`` / ``loss_large`` when block sizes differ, so a
        scale whose term has stalled is visible during training instead of only
        in the final metrics.
    """
    if not pred_blocks:
        raise ValueError("jepa_loss received no target blocks")
    if reduction not in ("per_patch", "per_block"):
        raise ValueError(
            f"Unknown loss reduction '{reduction}'. Use 'per_patch' or 'per_block'."
        )

    total = pred_blocks[0].new_zeros(())
    n_patches = 0
    cos_accum = pred_blocks[0].new_zeros(())
    block_means: list[torch.Tensor] = []
    block_sizes: list[int] = []

    for pred, target in zip(pred_blocks, target_blocks):
        dist = patch_distance(pred, target, kind=kind, alpha=alpha, beta=beta)
        total = total + dist.sum()
        n_patches += dist.numel()

        block_means.append(dist.mean())
        block_sizes.append(pred.size(1))

        with torch.no_grad():
            cos_accum = cos_accum + F.cosine_similarity(
                pred.detach(), target.detach(), dim=-1
            ).sum()

    if reduction == "per_block":
        loss = torch.stack(block_means).mean()
    else:
        loss = total / max(n_patches, 1)

    with torch.no_grad():
        # Variance of the teacher embeddings across the batch. If this collapses
        # towards 0 the model has found the trivial constant solution.
        flat = torch.cat([t.detach().reshape(-1, t.size(-1)) for t in target_blocks], dim=0)
        target_std = flat.std(dim=0).mean()

        stats = {
            "loss": float(loss.detach()),
            "cos_sim": float(cos_accum / max(n_patches, 1)),
            "target_std": float(target_std),
            "n_target_patches": n_patches,
            "reduction": reduction,
        }

        # Per-scale breakdown, so a stalled scale is visible in the logs.
        # 9 patches is the largest "small" block (3x3) in the default config.
        small = [m for m, s in zip(block_means, block_sizes) if s <= 9]
        large = [m for m, s in zip(block_means, block_sizes) if s > 9]
        if small:
            stats["loss_small"] = float(torch.stack(small).mean())
        if large:
            stats["loss_large"] = float(torch.stack(large).mean())

    return loss, stats
