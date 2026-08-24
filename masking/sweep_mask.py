"""Inference-time mask sweep.

Training masks are random; *evaluation* masks must be exhaustive. For a test
image we slide a target window across the whole patch grid so that every patch
eventually becomes a target while the rest of the image serves as context.

This is what makes the method work on anomalies that are *visible*. If an extra
or misplaced component stays inside the context, the predictor can simply copy
it and the error vanishes. Sweeping guarantees that each region is, at some
point, hidden from the predictor and must be imagined from the surrounding
(normal) structure -- which is exactly when the discrepancy appears.

The sweep is run at several window sizes, mirroring the multi-scale training
masks: small windows localise structural defects sharply, large windows expose
logical violations.
"""

from __future__ import annotations

import torch

from .base import MaskSpec, block_to_indices, complement_indices


def sweep_positions(grid_size: int, window: int, stride: int) -> list[tuple[int, int]]:
    """Top-left corners of a sliding window covering the whole grid.

    The last row/column is always included even when ``stride`` does not divide
    the grid evenly, so no patch is left uncovered.
    """
    if window > grid_size:
        window = grid_size

    coords = list(range(0, grid_size - window + 1, stride))
    last = grid_size - window
    if coords and coords[-1] != last:
        coords.append(last)
    elif not coords:
        coords = [0]

    return [(r, c) for r in coords for c in coords]


def build_sweep_masks(
    grid_size: int = 16,
    window: int = 4,
    stride: int = 2,
    device: torch.device | str = "cpu",
) -> list[MaskSpec]:
    """One :class:`MaskSpec` per sliding-window position.

    Each spec hides exactly one ``window x window`` block and exposes everything
    else as context.
    """
    specs: list[MaskSpec] = []
    for top, left in sweep_positions(grid_size, window, stride):
        block = block_to_indices(top, left, window, window, grid_size)
        specs.append(
            MaskSpec(
                context_idx=complement_indices(grid_size, [block]).to(device),
                target_blocks=[block.to(device)],
                boxes=[(top, left, window, window)],
                grid_size=grid_size,
                scale_tags=["small" if window <= 3 else "large"],
            )
        )
    return specs


class SweepMaskBank:
    """Pre-computed sweep masks for a set of window scales.

    Building the index tensors is pure python, so doing it once per evaluation
    run rather than once per image is a large saving: a 16x16 grid with windows
    (2, 4, 6) and stride 2 gives 64 + 49 + 36 = 149 forward configurations per
    image.

    Args:
        grid_size: patch-grid side.
        windows: target window sizes in patches.
        strides: stride per window; a single int applies to all. Defaults to
            ``max(1, window // 2)``, i.e. 50% overlap, so each patch is scored
            by several different contexts and the accumulated map is smooth.
        device: where the index tensors live.
    """

    def __init__(
        self,
        grid_size: int = 16,
        windows: list[int] | tuple[int, ...] = (2, 4, 6),
        strides: list[int] | int | None = None,
        device: torch.device | str = "cpu",
    ):
        self.grid_size = grid_size
        self.windows = list(windows)

        if strides is None:
            self.strides = [max(1, w // 2) for w in self.windows]
        elif isinstance(strides, int):
            self.strides = [strides] * len(self.windows)
        else:
            self.strides = list(strides)

        if len(self.strides) != len(self.windows):
            raise ValueError("strides must match the number of windows")

        self.device = torch.device(device)
        self.masks_per_scale: dict[int, list[MaskSpec]] = {
            w: build_sweep_masks(grid_size, w, s, self.device)
            for w, s in zip(self.windows, self.strides)
        }

    def to(self, device) -> "SweepMaskBank":
        """Move every cached index tensor to ``device``."""
        self.device = torch.device(device)
        self.masks_per_scale = {
            w: [m.to(self.device) for m in masks]
            for w, masks in self.masks_per_scale.items()
        }
        return self

    def scales(self) -> list[int]:
        return list(self.windows)

    def __getitem__(self, window: int) -> list[MaskSpec]:
        return self.masks_per_scale[window]

    def __len__(self) -> int:
        return sum(len(v) for v in self.masks_per_scale.values())

    def batched(self, window: int, chunk: int = 32):
        """Yield ``(context_idx, target_idx, boxes)`` batches for one scale.

        All windows at a given scale share a shape, so a chunk of sweep
        positions can be stacked and evaluated as one batch:

            context_idx: (n, Kc)   target_idx: (n, w*w)

        where ``n <= chunk`` is the number of positions in the chunk.
        """
        masks = self.masks_per_scale[window]
        for start in range(0, len(masks), chunk):
            group = masks[start : start + chunk]
            ctx = torch.stack([m.context_idx for m in group], dim=0)
            tgt = torch.stack([m.target_blocks[0] for m in group], dim=0)
            boxes = [m.boxes[0] for m in group]
            yield ctx, tgt, boxes

    def describe(self) -> str:
        parts = [
            f"w{w}/s{s}:{len(self.masks_per_scale[w])}"
            for w, s in zip(self.windows, self.strides)
        ]
        return f"SweepMaskBank(grid={self.grid_size}, {', '.join(parts)}, total={len(self)})"


def coverage_map(bank: SweepMaskBank, window: int) -> torch.Tensor:
    """(grid, grid) count of how many sweep windows cover each patch.

    Used to sanity-check that a window/stride pair leaves no patch unscored.
    """
    counts = torch.zeros(bank.grid_size**2)
    for spec in bank[window]:
        counts[spec.target_blocks[0].cpu()] += 1
    return counts.view(bank.grid_size, bank.grid_size)
