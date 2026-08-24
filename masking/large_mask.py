"""Large-block masking (ablation arm 3).

Large targets cover 5x5 to 8x8 patches, i.e. 80x80 to 128x128 pixels. A block
that size can swallow an entire component -- a screw, a connector, one of the
fruits in the LOCO ``breakfast_box`` -- so the predictor can no longer solve the
task by extrapolating texture. It has to answer the *logical* question: given
the rest of the scene, what object belongs here, and how many of them?

This is the arm that should carry logical-anomaly performance, at the cost of
coarse localisation: an 8x8 target produces one error value for a 128x128 pixel
region, which blurs the pixel-level metrics.
"""

from __future__ import annotations

import torch

from .base import MaskGenerator, sample_blocks

LARGE_SIZES: list[tuple[int, int]] = [(5, 5), (6, 6), (5, 7), (7, 5), (8, 8)]


class LargeBlockMask(MaskGenerator):
    """Sample a small number of large rectangular target blocks.

    Fewer blocks are used than in the small-scale arm because each one already
    removes a large fraction of the grid; with ``num_targets=2`` and 6x6 blocks
    roughly 28% of the image is hidden, which keeps enough context to make the
    prediction well-posed.
    """

    name = "large"

    def __init__(
        self,
        grid_size: int = 16,
        num_targets: int = 2,
        sizes: list[tuple[int, int]] | None = None,
        seed: int | None = None,
        max_overlap: float = 0.35,
        max_masked_ratio: float = 0.55,
    ):
        super().__init__(grid_size=grid_size, num_targets=num_targets, seed=seed)
        self.sizes = [tuple(s) for s in (sizes or LARGE_SIZES)]
        self.max_overlap = max_overlap
        self.max_masked_ratio = max_masked_ratio

    def sample_targets(self):
        blocks, boxes = sample_blocks(
            grid_size=self.grid_size,
            num_blocks=self.num_targets,
            sizes=self.sizes,
            rng=self.rng,
            max_overlap=self.max_overlap,
        )

        # Guard against a degenerate draw (e.g. two 8x8 blocks side by side)
        # leaving almost no context: drop blocks until the budget is respected.
        total = self.grid_size**2
        while len(blocks) > 1:
            claimed = set()
            for block in blocks:
                claimed.update(block.tolist())
            if len(claimed) / total <= self.max_masked_ratio:
                break
            blocks.pop()
            boxes.pop()

        return blocks, boxes, ["large"] * len(blocks)

    def describe(self) -> str:
        return f"large(grid={self.grid_size}, targets={self.num_targets}, sizes={self.sizes})"


def build_large_mask(cfg, grid_size: int = 16, seed: int | None = None) -> LargeBlockMask:
    """Construct a :class:`LargeBlockMask` from a config node."""
    # Dedicated key -- see build_small_mask for why this is not shared with
    # the multi-scale generator's `num_large_targets`.
    sizes = cfg.get("large_sizes", None)
    if sizes is not None:
        sizes = [tuple(s) for s in sizes]
    return LargeBlockMask(
        grid_size=grid_size,
        num_targets=cfg.get("num_large_only_targets", 2),
        sizes=sizes,
        seed=seed,
        max_overlap=cfg.get("max_overlap", 0.35),
        max_masked_ratio=cfg.get("max_masked_ratio", 0.55),
    )
