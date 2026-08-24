"""Random independent-patch masking (ablation arm 1 -- the baseline).

Every masked patch is drawn independently, MAE-style, instead of as a
contiguous block. This is the control condition for the masking study: it hides
the same *fraction* of the image as the block strategies but never hides a
contiguous region.

The expected outcome is that this arm is the weakest on logical anomalies, and
the reason is worth stating explicitly: when masked patches are scattered, every
hidden patch still has un-masked immediate neighbours, so the predictor can
solve the task by local interpolation and is never forced to learn the scene's
composition. I-JEPA reports the same effect for representation quality; this
arm checks whether it also governs *anomaly* performance.
"""

from __future__ import annotations

import torch

from .base import MaskGenerator, complement_indices


class RandomPatchMask(MaskGenerator):
    """Hide a random subset of individual patches.

    Args:
        grid_size: patch-grid side.
        mask_ratio: fraction of the grid to hide.
        num_targets: how many groups the masked patches are split into. The
            predictor consumes targets block-by-block, so splitting keeps the
            per-block token count in the same range as the block strategies.
        seed: RNG seed.
    """

    name = "random_patch"

    def __init__(
        self,
        grid_size: int = 16,
        mask_ratio: float = 0.20,
        num_targets: int = 4,
        seed: int | None = None,
    ):
        super().__init__(grid_size=grid_size, num_targets=num_targets, seed=seed)
        self.mask_ratio = mask_ratio

    def sample_targets(self):
        total = self.grid_size**2
        n_mask = max(int(round(total * self.mask_ratio)), self.num_targets)

        order = list(range(total))
        self.rng.shuffle(order)
        masked = sorted(order[:n_mask])

        # Split into equal-sized groups so every block has the same token count
        # and the predictor can batch them in one pass.
        per_block = n_mask // self.num_targets
        blocks, boxes, tags = [], [], []
        for i in range(self.num_targets):
            chunk = masked[i * per_block : (i + 1) * per_block]
            if not chunk:
                continue
            blocks.append(torch.tensor(chunk, dtype=torch.long))
            # Scattered patches have no meaningful bounding block; record the
            # enclosing box purely for visualisation.
            rows = [c // self.grid_size for c in chunk]
            cols = [c % self.grid_size for c in chunk]
            boxes.append(
                (min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)
            )
            tags.append("scattered")

        return blocks, boxes, tags

    def describe(self) -> str:
        return f"random_patch(grid={self.grid_size}, ratio={self.mask_ratio})"


def build_random_patch_mask(cfg, grid_size: int = 16, seed: int | None = None) -> RandomPatchMask:
    """Construct a :class:`RandomPatchMask` from a config node."""
    return RandomPatchMask(
        grid_size=grid_size,
        mask_ratio=cfg.get("mask_ratio", 0.20),
        num_targets=cfg.get("num_random_targets", 4),
        seed=seed,
    )
