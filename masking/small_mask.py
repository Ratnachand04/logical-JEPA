"""Small-block masking (ablation arm 2).

Small targets cover 2x2 to 3x3 patches, i.e. 32x32 to 48x48 pixels at the
default 256/16 resolution. At that scale the predictor is asked a *local*
question -- "is the texture and micro-structure here what the neighbourhood
implies?" -- which is the right question for structural anomalies: scratches,
cracks, contamination, a chipped edge.

The hypothesis this arm tests is that small-only masking gives strong structural
localisation but cannot see logical violations, because a 3x3 window never
contains a whole component and so never forces the model to reason about how
many of something should exist.
"""

from __future__ import annotations

import torch

from .base import MaskGenerator, sample_blocks

# Block shapes in patch units. Non-square shapes are included so the predictor
# does not overfit to a single aspect ratio.
SMALL_SIZES: list[tuple[int, int]] = [(2, 2), (2, 3), (3, 2), (3, 3)]


class SmallBlockMask(MaskGenerator):
    """Sample several small rectangular target blocks.

    Args:
        grid_size: patch-grid side.
        num_targets: how many blocks to hide per batch. More small blocks are
            used than large ones so the total masked area stays comparable
            across ablation arms (~15-25% of the grid).
        sizes: candidate ``(height, width)`` block shapes.
        seed: RNG seed for reproducible mask sequences.
    """

    name = "small"

    def __init__(
        self,
        grid_size: int = 16,
        num_targets: int = 6,
        sizes: list[tuple[int, int]] | None = None,
        seed: int | None = None,
        max_overlap: float = 0.25,
    ):
        super().__init__(grid_size=grid_size, num_targets=num_targets, seed=seed)
        self.sizes = [tuple(s) for s in (sizes or SMALL_SIZES)]
        self.max_overlap = max_overlap

    def sample_targets(self):
        blocks, boxes = sample_blocks(
            grid_size=self.grid_size,
            num_blocks=self.num_targets,
            sizes=self.sizes,
            rng=self.rng,
            max_overlap=self.max_overlap,
        )
        return blocks, boxes, ["small"] * len(blocks)

    def describe(self) -> str:
        return f"small(grid={self.grid_size}, targets={self.num_targets}, sizes={self.sizes})"


def build_small_mask(cfg, grid_size: int = 16, seed: int | None = None) -> SmallBlockMask:
    """Construct a :class:`SmallBlockMask` from a config node."""
    # Dedicated key: `num_small_targets` belongs to the multi-scale generator,
    # where small blocks are only part of the budget. The small-only arm needs
    # more of them to mask a comparable area, so it reads its own key.
    sizes = cfg.get("small_sizes", None)
    if sizes is not None:
        sizes = [tuple(s) for s in sizes]
    return SmallBlockMask(
        grid_size=grid_size,
        num_targets=cfg.get("num_small_only_targets", 6),
        sizes=sizes,
        seed=seed,
        max_overlap=cfg.get("max_overlap", 0.25),
    )
