"""VICReg-style collapse regulariser (Phase 3d).

Training currently *watches* ``target_std`` and warns when it drifts towards
zero. That is a smoke alarm, not a sprinkler: by the time it fires the run is
already wasted. This module adds the two terms that make non-collapse an
objective rather than an observation.

**Variance term.** Pushes each embedding dimension's standard deviation, across
the batch, above a floor. A dimension that carries the same value for every
sample carries no information; hinging at the floor means the term is exactly
zero once a dimension is healthy, so it stops pulling as soon as it has done its
job.

**Covariance term.** Drives off-diagonal covariances towards zero. Without it a
model can satisfy the variance floor while packing all its variance into one
direction repeated across dimensions -- technically not collapsed, informationally
almost as bad.

Weighting, and why it is low by default
--------------------------------------
These terms fight the JEPA objective. The predictive loss *wants* an embedding
where a region is predictable from its context, which necessarily means
structure -- correlations included. Over-weighting VICReg buys non-collapse by
destroying exactly that structure, and the anomaly scores degrade even though
``target_std`` looks healthier than ever. They are a safety net, so the defaults
are small and :func:`collapse_report` keeps the original monitor running as the
regulariser's own unit test: if ``target_std`` still drifts towards zero with
VICReg active, it is mis-weighted, not unnecessary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0,
                  eps: float = 1e-4) -> torch.Tensor:
    """Hinge each dimension's batch standard deviation up to ``gamma``.

    Args:
        embeddings: (N, D) -- flatten tokens into the batch axis before calling.
        gamma: target standard deviation floor.
        eps: variance epsilon, inside the sqrt for gradient stability at zero.

    Returns:
        Scalar. Zero when every dimension already exceeds the floor.
    """
    if embeddings.size(0) < 2:
        return embeddings.new_zeros(())

    std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """Sum of squared off-diagonal covariances, normalised by dimension.

    Decorrelates the embedding dimensions so variance cannot be satisfied by
    one direction duplicated across channels.
    """
    n, d = embeddings.shape
    if n < 2:
        return embeddings.new_zeros(())

    centred = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (centred.T @ centred) / (n - 1)

    off_diagonal = cov.pow(2).sum() - cov.pow(2).diagonal().sum()
    return off_diagonal / d


def vicreg_loss(
    embeddings: torch.Tensor,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined variance + covariance regulariser.

    Args:
        embeddings: (B, K, D) or (N, D). Token axes are folded into the batch,
            so the statistics are taken over *all* predicted tokens.
        var_weight / cov_weight: term weights. See the module docstring for why
            these stay small.
        gamma: variance floor.

    Returns:
        ``(loss, stats)`` with the two terms reported separately -- a combined
        number hides which one is actually active.
    """
    if embeddings.dim() == 3:
        embeddings = embeddings.reshape(-1, embeddings.size(-1))

    var = variance_loss(embeddings, gamma=gamma)
    cov = covariance_loss(embeddings)
    loss = var_weight * var + cov_weight * cov

    with torch.no_grad():
        stats = {
            "vicreg_var": float(var),
            "vicreg_cov": float(cov),
            "vicreg_total": float(loss),
            "embed_std": float(embeddings.std(dim=0).mean()),
        }

    return loss, stats


@torch.no_grad()
def collapse_report(embeddings: torch.Tensor, gamma: float = 1.0) -> dict[str, float]:
    """Diagnostics for how close a representation is to collapsing.

    Kept deliberately separate from the loss: this is the regulariser's unit
    test at training time, and it must stay meaningful whether or not VICReg is
    enabled.

    ``dead_dim_frac``
        Share of dimensions whose batch std is below 10% of the floor. This is
        the number that should stay near zero; if it climbs while VICReg is on,
        the weights are too low.
    ``effective_rank``
        ``exp(entropy of the normalised eigenvalue spectrum)``. Roughly "how
        many dimensions are actually in use". A representation with 256 channels
        and an effective rank of 3 has collapsed in every sense that matters,
        even if no single dimension is dead.
    """
    if embeddings.dim() == 3:
        embeddings = embeddings.reshape(-1, embeddings.size(-1))

    if embeddings.size(0) < 2:
        return {"embed_std": 0.0, "dead_dim_frac": 1.0, "effective_rank": 0.0}

    std = embeddings.std(dim=0)
    centred = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (centred.T @ centred) / (embeddings.size(0) - 1)

    eigenvalues = torch.linalg.eigvalsh(cov.float()).clamp_min(0)
    total = eigenvalues.sum()
    if total <= 1e-12:
        effective_rank = 0.0
    else:
        p = (eigenvalues / total).clamp_min(1e-12)
        effective_rank = float(torch.exp(-(p * p.log()).sum()))

    return {
        "embed_std": float(std.mean()),
        "dead_dim_frac": float((std < 0.1 * gamma).float().mean()),
        "effective_rank": effective_rank,
    }


def build_vicreg(cfg) -> dict | None:
    """Read the ``regularizer.vicreg`` config node.

    Returns ``None`` when disabled, so the training loop can skip the term
    entirely rather than multiplying by a zero weight.
    """
    node = dict((cfg or {}).get("vicreg", {}) or {})
    if not node.get("enabled", False):
        return None

    return {
        "var_weight": node.get("var_weight", 1.0),
        "cov_weight": node.get("cov_weight", 0.04),
        "gamma": node.get("gamma", 1.0),
    }
