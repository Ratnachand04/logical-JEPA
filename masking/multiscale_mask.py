"""Multi-Scale Contextual Masking -- the main contribution of Logical-JEPA.

Vanilla I-JEPA samples target blocks from a single scale range. That is a
reasonable choice for representation learning, but for anomaly detection the two
failure modes we care about live at different scales:

* **structural** anomalies (scratch, chip, contamination) are a few patches wide
  and are only visible when the target window is comparably small;
* **logical** anomalies (missing / extra / misplaced component, wrong count) only
  become visible when the target window is big enough to contain a whole part,
  so that the predictor is forced to reason about scene composition rather than
  local texture.

A model trained on one scale can only be asked one of those questions. This
generator interleaves both within training, and -- in ``mixed`` mode -- inside a
single step, so the shared encoder has to produce a representation that supports
local *and* compositional prediction at once.

Three sampling policies are provided:

``alternate``
    Each step draws entirely small or entirely large blocks, with probability
    ``p_small``. Cheapest, and closest to running the two ablation arms in
    alternation.
``mixed`` (default)
    Every step contains both scales: ``num_large_targets`` large blocks and
    ``num_small_targets`` small ones. The gradient at each step therefore
    carries both signals, which is what the ablation is meant to show matters.
``curriculum``
    Starts small-heavy and anneals towards large-heavy. Local prediction is the
    easier task, so this warms the encoder up before asking compositional
    questions. ``set_progress()`` must be called from the training loop.
"""

from __future__ import annotations

import torch

from .base import MaskGenerator, sample_blocks
from .large_mask import LARGE_SIZES
from .small_mask import SMALL_SIZES


class MultiScaleMask(MaskGenerator):
    """Sample target blocks from small and large scale ranges jointly.

    Args:
        grid_size: patch-grid side.
        num_small_targets: small blocks per step (``mixed``/``alternate`` mode).
        num_large_targets: large blocks per step.
        small_sizes / large_sizes: candidate block shapes per scale.
        mode: ``mixed``, ``alternate`` or ``curriculum``.
        p_small: probability of drawing the small scale in ``alternate`` mode,
            and the *initial* small-weighting in ``curriculum`` mode.
        seed: RNG seed.
    """

    name = "multiscale"

    def __init__(
        self,
        grid_size: int = 16,
        num_small_targets: int = 3,
        num_large_targets: int = 1,
        small_sizes: list[tuple[int, int]] | None = None,
        large_sizes: list[tuple[int, int]] | None = None,
        mode: str = "mixed",
        p_small: float = 0.5,
        seed: int | None = None,
        max_overlap: float = 0.3,
        max_masked_ratio: float = 0.6,
    ):
        super().__init__(
            grid_size=grid_size,
            num_targets=num_small_targets + num_large_targets,
            seed=seed,
        )
        if mode not in ("mixed", "alternate", "curriculum"):
            raise ValueError(f"Unknown multi-scale mode: {mode}")

        self.num_small_targets = num_small_targets
        self.num_large_targets = num_large_targets
        self.small_sizes = [tuple(s) for s in (small_sizes or SMALL_SIZES)]
        self.large_sizes = [tuple(s) for s in (large_sizes or LARGE_SIZES)]
        self.mode = mode
        self.p_small = p_small
        self.max_overlap = max_overlap
        self.max_masked_ratio = max_masked_ratio
        self._progress = 0.0

    # ------------------------------------------------------------------ #
    def set_progress(self, progress: float) -> None:
        """Report training progress in [0, 1] (used by ``curriculum`` mode)."""
        self._progress = min(max(float(progress), 0.0), 1.0)

    def _current_p_small(self) -> float:
        if self.mode != "curriculum":
            return self.p_small
        # Anneal from mostly-small (0.85) down to mostly-large (0.25).
        return 0.85 + (0.25 - 0.85) * self._progress

    # ------------------------------------------------------------------ #
    def _sample_scale(self, sizes, count, tag):
        if count <= 0:
            return [], [], []
        blocks, boxes = sample_blocks(
            grid_size=self.grid_size,
            num_blocks=count,
            sizes=sizes,
            rng=self.rng,
            max_overlap=self.max_overlap,
        )
        return blocks, boxes, [tag] * len(blocks)

    def _trim_to_budget(self, blocks, boxes, tags):
        """Drop the largest blocks until the masked fraction is acceptable.

        Keeps at least one block so a step is never target-free.
        """
        total = self.grid_size**2
        while len(blocks) > 1:
            claimed: set[int] = set()
            for block in blocks:
                claimed.update(block.tolist())
            if len(claimed) / total <= self.max_masked_ratio:
                break
            biggest = max(range(len(blocks)), key=lambda i: blocks[i].numel())
            blocks.pop(biggest)
            boxes.pop(biggest)
            tags.pop(biggest)
        return blocks, boxes, tags

    def sample_targets(self):
        if self.mode == "mixed":
            # Large blocks are placed first so they get the freest choice of
            # position; the small ones then fill the gaps around them.
            lb, lx, lt = self._sample_scale(
                self.large_sizes, self.num_large_targets, "large"
            )
            sb, sx, st = self._sample_scale(
                self.small_sizes, self.num_small_targets, "small"
            )
            blocks, boxes, tags = lb + sb, lx + sx, lt + st

        else:  # alternate / curriculum -- one scale per step
            use_small = self.rng.random() < self._current_p_small()
            if use_small:
                blocks, boxes, tags = self._sample_scale(
                    self.small_sizes, max(self.num_small_targets, 1), "small"
                )
            else:
                blocks, boxes, tags = self._sample_scale(
                    self.large_sizes, max(self.num_large_targets, 1), "large"
                )

        return self._trim_to_budget(blocks, boxes, tags)

    def describe(self) -> str:
        return (
            f"multiscale(mode={self.mode}, small={self.num_small_targets}"
            f"x{self.small_sizes}, large={self.num_large_targets}x{self.large_sizes})"
        )


def build_multiscale_mask(cfg, grid_size: int = 16, seed: int | None = None) -> MultiScaleMask:
    """Construct a :class:`MultiScaleMask` from a config node."""
    small_sizes = cfg.get("small_sizes", None)
    large_sizes = cfg.get("large_sizes", None)
    return MultiScaleMask(
        grid_size=grid_size,
        num_small_targets=cfg.get("num_small_targets", 3),
        num_large_targets=cfg.get("num_large_targets", 1),
        small_sizes=[tuple(s) for s in small_sizes] if small_sizes else None,
        large_sizes=[tuple(s) for s in large_sizes] if large_sizes else None,
        mode=cfg.get("mode", "mixed"),
        p_small=cfg.get("p_small", 0.5),
        seed=seed,
        max_overlap=cfg.get("max_overlap", 0.3),
        max_masked_ratio=cfg.get("max_masked_ratio", 0.6),
    )
