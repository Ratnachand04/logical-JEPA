"""Shared primitives for every Logical-JEPA masking strategy.

All strategies work on the 16x16 patch grid and return a :class:`MaskSpec`:
a set of rectangular *target* blocks plus the *context* index (everything the
context encoder is allowed to see).

Masks are sampled once per batch and shared across the samples in it. That is
the I-JEPA collator convention and it matters practically: identical block
shapes across the batch let the predictor fold the target blocks into the batch
dimension in a single pass.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch


@dataclass
class MaskSpec:
    """One sampled masking configuration on the patch grid.

    Attributes:
        context_idx: (Kc,) long tensor of visible patch indices.
        target_blocks: list of (Kt_m,) long tensors, one per target region.
        boxes: the ``(top, left, height, width)`` of each block, in patch units,
            kept for visualisation and for the ablation logs.
        grid_size: side of the patch grid the indices refer to.
        scale_tags: a label per block (``small`` / ``large``) so the multi-scale
            strategy can report what it actually sampled.
    """

    context_idx: torch.Tensor
    target_blocks: list[torch.Tensor]
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    grid_size: int = 16
    scale_tags: list[str] = field(default_factory=list)

    @property
    def num_targets(self) -> int:
        return len(self.target_blocks)

    @property
    def num_target_patches(self) -> int:
        return sum(int(t.numel()) for t in self.target_blocks)

    def to(self, device) -> "MaskSpec":
        """Move every index tensor to ``device``."""
        return MaskSpec(
            context_idx=self.context_idx.to(device),
            target_blocks=[t.to(device) for t in self.target_blocks],
            boxes=list(self.boxes),
            grid_size=self.grid_size,
            scale_tags=list(self.scale_tags),
        )

    def target_mask_grid(self) -> torch.Tensor:
        """(grid, grid) uint8 map of how many blocks cover each patch."""
        grid = torch.zeros(self.grid_size * self.grid_size, dtype=torch.uint8)
        for block in self.target_blocks:
            grid[block.cpu()] += 1
        return grid.view(self.grid_size, self.grid_size)


def block_to_indices(top: int, left: int, height: int, width: int, grid_size: int) -> torch.Tensor:
    """Flatten a rectangular patch block into row-major token indices."""
    rows = torch.arange(top, top + height, dtype=torch.long)
    cols = torch.arange(left, left + width, dtype=torch.long)
    idx = rows.unsqueeze(1) * grid_size + cols.unsqueeze(0)
    return idx.reshape(-1)


def sample_block(
    grid_size: int,
    height: int,
    width: int,
    rng: random.Random | None = None,
) -> tuple[int, int, int, int]:
    """Uniformly place an ``height x width`` block inside the grid."""
    rng = rng or random
    height = min(height, grid_size)
    width = min(width, grid_size)
    top = rng.randint(0, grid_size - height)
    left = rng.randint(0, grid_size - width)
    return top, left, height, width


def sample_blocks(
    grid_size: int,
    num_blocks: int,
    sizes: list[tuple[int, int]],
    rng: random.Random | None = None,
    max_overlap: float = 0.5,
    tries: int = 20,
) -> tuple[list[torch.Tensor], list[tuple[int, int, int, int]]]:
    """Sample ``num_blocks`` blocks, rejecting heavily overlapping placements.

    Overlapping targets are wasteful (the predictor is asked the same question
    twice) so a placement is retried while more than ``max_overlap`` of its
    patches are already claimed. After ``tries`` attempts the block is accepted
    anyway, which keeps sampling bounded on small grids.
    """
    rng = rng or random
    blocks: list[torch.Tensor] = []
    boxes: list[tuple[int, int, int, int]] = []
    claimed: set[int] = set()

    for _ in range(num_blocks):
        best_idx, best_box, best_frac = None, None, 2.0

        for _ in range(tries):
            h, w = sizes[rng.randrange(len(sizes))]
            box = sample_block(grid_size, h, w, rng)
            idx = block_to_indices(*box, grid_size=grid_size)
            overlap = len(claimed.intersection(idx.tolist())) / max(idx.numel(), 1)

            if overlap < best_frac:
                best_idx, best_box, best_frac = idx, box, overlap
            if overlap <= max_overlap:
                break

        assert best_idx is not None and best_box is not None
        blocks.append(best_idx)
        boxes.append(best_box)
        claimed.update(best_idx.tolist())

    return blocks, boxes


def complement_indices(
    grid_size: int,
    target_blocks: list[torch.Tensor],
    min_context: int = 16,
) -> torch.Tensor:
    """Context = every patch not claimed by any target block.

    I-JEPA samples a separate context block and then subtracts the targets from
    it. Here the full complement is used instead: for anomaly detection we want
    the predictor conditioned on *all* remaining evidence, since the question at
    test time is precisely "given everything else, does this region belong?".

    If the targets swallow too much of the grid, some patches are returned to
    the context so the encoder is never handed a near-empty sequence.
    """
    total = grid_size * grid_size
    claimed: set[int] = set()
    for block in target_blocks:
        claimed.update(block.tolist())

    context = [i for i in range(total) if i not in claimed]

    if len(context) < min_context:
        spare = sorted(claimed)
        random.shuffle(spare)
        context.extend(spare[: min_context - len(context)])
        context.sort()

    return torch.tensor(context, dtype=torch.long)


class MaskGenerator:
    """Base class for masking strategies.

    Subclasses implement :meth:`sample_targets`; the base class turns those
    blocks into a full :class:`MaskSpec` with the matching context.
    """

    name = "base"

    def __init__(self, grid_size: int = 16, num_targets: int = 4, seed: int | None = None):
        self.grid_size = grid_size
        self.num_targets = num_targets
        self.rng = random.Random(seed)

    def sample_targets(self) -> tuple[list[torch.Tensor], list[tuple[int, int, int, int]], list[str]]:
        raise NotImplementedError

    def __call__(self) -> MaskSpec:
        blocks, boxes, tags = self.sample_targets()
        return MaskSpec(
            context_idx=complement_indices(self.grid_size, blocks),
            target_blocks=blocks,
            boxes=boxes,
            grid_size=self.grid_size,
            scale_tags=tags,
        )

    def describe(self) -> str:
        return f"{self.name}(grid={self.grid_size}, targets={self.num_targets})"
