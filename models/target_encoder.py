"""Target encoder: the EMA (teacher) copy of the context encoder.

The target encoder has exactly the same architecture as the context encoder but
receives no gradient. Its weights follow the student through an exponential
moving average

    theta_t <- m * theta_t + (1 - m) * theta_c

with the momentum ``m`` annealed from ``base_momentum`` (~0.996) to
``final_momentum`` (~1.0) over training. This is the mechanism that stops the
JEPA objective collapsing to a constant representation: the target is a slowly
moving, non-differentiable version of the student.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from .context_encoder import ContextEncoder


class TargetEncoder(nn.Module):
    """EMA wrapper around a frozen structural copy of the context encoder."""

    def __init__(self, context_encoder: ContextEncoder,
                 base_momentum: float = 0.996,
                 final_momentum: float = 1.0):
        super().__init__()

        # Deep-copy so the teacher starts identical to the student, then detach
        # every parameter from autograd.
        self.encoder = copy.deepcopy(context_encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.register_buffer("_momentum", torch.tensor(float(base_momentum)))

    # ------------------------------------------------------------------ #
    # EMA update
    # ------------------------------------------------------------------ #
    def momentum_at(self, step: int, total_steps: int) -> float:
        """Cosine schedule from ``base_momentum`` up to ``final_momentum``."""
        if total_steps <= 0:
            return self.base_momentum
        progress = min(max(step / total_steps, 0.0), 1.0)
        span = self.final_momentum - self.base_momentum
        return self.final_momentum - span * (math.cos(math.pi * progress) + 1.0) / 2.0

    @torch.no_grad()
    def update(self, context_encoder: ContextEncoder, momentum: float | None = None) -> float:
        """Apply one EMA step from the student weights.

        Buffers (the fixed sin-cos position table) are copied outright rather
        than averaged, since they are constants, not learned state.
        """
        m = self.base_momentum if momentum is None else float(momentum)
        m = min(max(m, 0.0), 1.0)

        for tgt_p, src_p in zip(self.encoder.parameters(), context_encoder.parameters()):
            tgt_p.mul_(m).add_(src_p.detach(), alpha=1.0 - m)

        for tgt_b, src_b in zip(self.encoder.buffers(), context_encoder.buffers()):
            tgt_b.copy_(src_b)

        self._momentum.fill_(m)
        return m

    @property
    def momentum(self) -> float:
        """Momentum used by the most recent update (logged during training)."""
        return float(self._momentum.item())

    # ------------------------------------------------------------------ #
    # Forward passes (always no-grad)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, images: torch.Tensor, index: torch.Tensor | None = None) -> torch.Tensor:
        """Encode the *full* image and optionally gather a subset of tokens.

        Unlike the context encoder, the teacher always sees every patch: the
        target representation of a region must be conditioned on the real image
        content there, which is exactly what makes the prediction error a
        meaningful anomaly signal at test time.
        """
        self.encoder.eval()
        tokens = self.encoder(images, context_idx=None)   # (B, N, D)

        if index is not None:
            from .patch_embed import gather_tokens
            tokens = gather_tokens(tokens, index)

        return tokens

    def train(self, mode: bool = True) -> "TargetEncoder":
        """Keep the teacher in eval mode regardless of the parent's mode.

        Dropout / stochastic depth in the teacher would inject noise straight
        into the regression target.
        """
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters())


def normalize_targets(targets: torch.Tensor, mode: str = "layernorm") -> torch.Tensor:
    """Normalise teacher outputs before they are used as regression targets.

    I-JEPA applies LayerNorm to the target patches; this removes per-patch scale
    from the objective and empirically prevents the loss from being dominated by
    high-norm tokens.

    Args:
        targets: (B, K, D) teacher embeddings.
        mode: ``layernorm``, ``l2`` or ``none``.
    """
    if mode == "layernorm":
        return nn.functional.layer_norm(targets, (targets.size(-1),))
    if mode == "l2":
        return nn.functional.normalize(targets, dim=-1)
    if mode in ("none", None):
        return targets
    raise ValueError(f"Unknown target normalisation mode: {mode}")
