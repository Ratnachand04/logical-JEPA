"""Masking strategies for Logical-JEPA.

Four training strategies (the four ablation arms) plus the inference-time sweep:

===================  ======================================================
``random_patch``     scattered individual patches (MAE-style control)
``small``            2x2 - 3x3 blocks, targets structural anomalies
``large``            5x5 - 8x8 blocks, targets logical anomalies
``multiscale``       both scales jointly -- the proposed method
===================  ======================================================
"""

from .base import (
    MaskGenerator,
    MaskSpec,
    block_to_indices,
    complement_indices,
    sample_block,
    sample_blocks,
)
from .large_mask import LARGE_SIZES, LargeBlockMask, build_large_mask
from .multiscale_mask import MultiScaleMask, build_multiscale_mask
from .random_patch_mask import RandomPatchMask, build_random_patch_mask
from .small_mask import SMALL_SIZES, SmallBlockMask, build_small_mask
from .sweep_mask import SweepMaskBank, build_sweep_masks, coverage_map, sweep_positions

MASK_REGISTRY = {
    "random_patch": build_random_patch_mask,
    "small": build_small_mask,
    "large": build_large_mask,
    "multiscale": build_multiscale_mask,
}


def build_mask_generator(cfg, grid_size: int = 16, seed: int | None = None) -> MaskGenerator:
    """Instantiate the masking strategy named by ``cfg.strategy``.

    Args:
        cfg: config node with a ``strategy`` key plus strategy-specific options.
        grid_size: patch-grid side.
        seed: RNG seed for the generator.

    Raises:
        ValueError: if ``cfg.strategy`` is not one of :data:`MASK_REGISTRY`.
    """
    strategy = cfg.get("strategy", "multiscale")
    if strategy not in MASK_REGISTRY:
        raise ValueError(
            f"Unknown masking strategy '{strategy}'. "
            f"Available: {sorted(MASK_REGISTRY)}"
        )
    return MASK_REGISTRY[strategy](cfg, grid_size=grid_size, seed=seed)


__all__ = [
    "MaskSpec",
    "MaskGenerator",
    "block_to_indices",
    "complement_indices",
    "sample_block",
    "sample_blocks",
    "SmallBlockMask",
    "LargeBlockMask",
    "MultiScaleMask",
    "RandomPatchMask",
    "SMALL_SIZES",
    "LARGE_SIZES",
    "build_small_mask",
    "build_large_mask",
    "build_multiscale_mask",
    "build_random_patch_mask",
    "build_mask_generator",
    "MASK_REGISTRY",
    "SweepMaskBank",
    "build_sweep_masks",
    "sweep_positions",
    "coverage_map",
]
