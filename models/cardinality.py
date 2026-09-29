"""Cardinality head (Phase 3c).

Question it answers: *given everything outside this region, how much component
mass should be inside it?* The answer is compared with how much there actually
is, measured on the teacher's slot attention.

Why it exists
-------------
Phase 1 measured logical AUROC *below chance* on real LOCO ``pushpins``: a
missing pin leaves an empty compartment that is easier to predict than a pin, so
a "harder than normal" score ranks it as extra-normal. A count-like quantity has
no such bias -- too little and too much are both a mismatch.

Constraints from the Phase 3a gate, all honoured here
-----------------------------------------------------
* **Scalar per region, never per component type.** Slots group by *type* (all
  screws share one slot), so per-instance or per-type counts are unavailable.
  The target is the region's non-background attention share,
  :func:`models.slot_bottleneck.region_foreground_mass`.
* **Stop-gradient into the slots.** The head reads detached slot attention; its
  loss cannot reshape the grouping to make its own job easier.
* **Normalised, sorted usage vector as input**, not raw attention mass, so the
  feature is comparable across images despite arbitrary slot order.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .patch_embed import build_2d_sincos_pos_embed, gather_pos_embed
from .transformer import init_vit_weights


class CardinalityHead(nn.Module):
    """MLP: (sorted context usage, region position, region size) -> mass in [0, 1].

    Args:
        num_slots: length of the usage vector.
        grid_size: patch-grid side, for the positional table.
        pos_dim: width of the fixed sin-cos table used to describe the region.
        hidden_dim: MLP width.
    """

    def __init__(self, num_slots: int, grid_size: int, pos_dim: int = 64,
                 hidden_dim: int = 128):
        super().__init__()
        self.num_slots = num_slots
        self.grid_size = grid_size
        self.num_patches = grid_size**2

        # Fixed, like every other positional table in the model: the sweep asks
        # about regions no training mask ever produced.
        self.register_buffer("pos_embed", build_2d_sincos_pos_embed(pos_dim, grid_size),
                             persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(num_slots + pos_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_vit_weights)

    def forward(self, context_usage: torch.Tensor, region_idx: torch.Tensor) -> torch.Tensor:
        """Predict the region's foreground mass.

        Args:
            context_usage: (B, S) sorted, normalised slot usage of the context.
            region_idx: (B, K) patch indices of the hidden region.

        Returns:
            (B,) predictions in [0, 1].
        """
        B = context_usage.size(0)
        pos = gather_pos_embed(self.pos_embed, region_idx, B).mean(dim=1)   # (B, P)
        size = torch.full((B, 1), region_idx.size(-1) / self.num_patches,
                          device=context_usage.device, dtype=pos.dtype)
        features = torch.cat([context_usage.to(pos.dtype), pos, size], dim=-1)
        return torch.sigmoid(self.mlp(features).squeeze(-1))

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
