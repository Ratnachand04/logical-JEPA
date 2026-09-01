"""Invariants every masking strategy must hold.

These are properties, not golden values: the strategies are stochastic, so the
tests assert what must be true of *every* draw rather than pinning one sample.
The invariants that matter for correctness are:

* context and targets are disjoint -- a leaked target patch would let the
  predictor copy the answer straight out of the context, silently inflating
  every metric downstream;
* the context is never empty -- an encoder handed zero tokens produces
  degenerate predictions;
* indices stay inside the grid -- an out-of-range gather is a hard crash at
  train time, but only for some random draws, so it must be caught here.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from masking import MASK_REGISTRY, build_mask_generator  # noqa: E402
from masking.base import block_to_indices, complement_indices, sample_blocks  # noqa: E402
from masking.sweep_mask import SweepMaskBank, coverage_map, sweep_positions  # noqa: E402

GRID = 16
STRATEGIES = sorted(MASK_REGISTRY)
DRAWS = 60


def _gen(strategy: str, seed: int = 0, **extra):
    cfg = {"strategy": strategy}
    cfg.update(extra)
    return build_mask_generator(cfg, grid_size=GRID, seed=seed)


# --------------------------------------------------------------------------- #
# Per-strategy invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_context_and_targets_are_disjoint(strategy):
    """A target patch visible in the context defeats the whole objective."""
    gen = _gen(strategy)
    for _ in range(DRAWS):
        spec = gen()
        claimed = set()
        for block in spec.target_blocks:
            claimed.update(block.tolist())
        assert not (set(spec.context_idx.tolist()) & claimed), (
            f"{strategy}: target patches leaked into the context"
        )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_indices_are_within_the_grid(strategy):
    gen = _gen(strategy)
    total = GRID * GRID
    for _ in range(DRAWS):
        spec = gen()
        assert spec.context_idx.min() >= 0 and spec.context_idx.max() < total
        for block in spec.target_blocks:
            assert block.min() >= 0 and block.max() < total
            assert block.dtype == torch.long


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_context_is_never_empty(strategy):
    gen = _gen(strategy)
    for _ in range(DRAWS):
        assert gen().context_idx.numel() >= 16


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_at_least_one_target_block(strategy):
    gen = _gen(strategy)
    for _ in range(DRAWS):
        spec = gen()
        assert spec.num_targets >= 1
        assert spec.num_target_patches >= 1


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_masked_fraction_is_in_a_sane_band(strategy):
    """Every arm should hide a comparable slice of the image.

    If one strategy hides 5% and another 60%, the masking ablation is really
    measuring difficulty, not scale -- so the arms would not be comparable.
    """
    gen = _gen(strategy)
    ratios = []
    for _ in range(DRAWS):
        spec = gen()
        claimed = set()
        for block in spec.target_blocks:
            claimed.update(block.tolist())
        ratios.append(len(claimed) / (GRID * GRID))

    mean_ratio = sum(ratios) / len(ratios)
    assert 0.08 <= mean_ratio <= 0.45, f"{strategy}: mean masked ratio {mean_ratio:.3f}"
    assert max(ratios) <= 0.70, f"{strategy}: a draw hid {max(ratios):.3f} of the grid"


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_same_seed_reproduces_the_same_masks(strategy):
    a = [_gen(strategy, seed=7)() for _ in range(5)]
    b = [_gen(strategy, seed=7)() for _ in range(5)]
    for x, y in zip(a, b):
        assert torch.equal(x.context_idx, y.context_idx)
        assert len(x.target_blocks) == len(y.target_blocks)
        for bx, by in zip(x.target_blocks, y.target_blocks):
            assert torch.equal(bx, by)


def test_different_seeds_give_different_masks():
    a, b = _gen("multiscale", seed=1)(), _gen("multiscale", seed=2)()
    assert not torch.equal(a.context_idx, b.context_idx)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown masking strategy"):
        build_mask_generator({"strategy": "does_not_exist"}, GRID, 0)


# --------------------------------------------------------------------------- #
# Scale semantics
# --------------------------------------------------------------------------- #
def test_small_strategy_only_emits_small_blocks():
    gen = _gen("small")
    for _ in range(DRAWS):
        spec = gen()
        assert set(spec.scale_tags) == {"small"}
        for block in spec.target_blocks:
            assert block.numel() <= 9      # at most 3x3


def test_large_strategy_only_emits_large_blocks():
    gen = _gen("large")
    for _ in range(DRAWS):
        spec = gen()
        assert set(spec.scale_tags) == {"large"}
        for block in spec.target_blocks:
            assert block.numel() >= 25     # at least 5x5


def test_multiscale_mixed_emits_both_scales():
    gen = _gen("multiscale", mode="mixed")
    seen = set()
    for _ in range(DRAWS):
        seen.update(gen().scale_tags)
    assert seen == {"small", "large"}, f"mixed mode produced only {seen}"


def test_curriculum_shifts_from_small_to_large():
    """Progress 0 must be small-heavy; progress 1 must be large-heavy."""
    gen = _gen("multiscale", mode="curriculum", seed=3)

    gen.set_progress(0.0)
    early = [gen().scale_tags[0] for _ in range(200)]
    gen.set_progress(1.0)
    late = [gen().scale_tags[0] for _ in range(200)]

    early_small = early.count("small") / len(early)
    late_small = late.count("small") / len(late)
    assert early_small > 0.7, f"curriculum start not small-heavy ({early_small:.2f})"
    assert late_small < 0.4, f"curriculum end not large-heavy ({late_small:.2f})"


# --------------------------------------------------------------------------- #
# Block geometry helpers
# --------------------------------------------------------------------------- #
def test_block_to_indices_is_row_major():
    idx = block_to_indices(0, 0, 2, 2, GRID)
    assert idx.tolist() == [0, 1, GRID, GRID + 1]

    idx = block_to_indices(3, 5, 3, 3, GRID)
    assert idx[0].item() == 3 * GRID + 5
    assert idx[-1].item() == 5 * GRID + 7
    assert idx.numel() == 9


def test_sampled_blocks_stay_inside_the_grid():
    import random

    rng = random.Random(0)
    for _ in range(200):
        blocks, boxes = sample_blocks(GRID, 3, [(2, 2), (5, 5), (8, 8)], rng)
        for (top, left, h, w) in boxes:
            assert 0 <= top and top + h <= GRID
            assert 0 <= left and left + w <= GRID


def test_complement_is_the_exact_set_difference():
    blocks = [block_to_indices(0, 0, 4, 4, GRID)]
    ctx = complement_indices(GRID, blocks)
    assert ctx.numel() == GRID * GRID - 16
    assert not (set(ctx.tolist()) & set(blocks[0].tolist()))


def test_complement_guards_against_a_near_empty_context():
    """Targets covering almost everything must still leave a usable context."""
    blocks = [block_to_indices(0, 0, GRID, GRID, GRID)]      # the whole grid
    ctx = complement_indices(GRID, blocks, min_context=16)
    assert ctx.numel() >= 16


# --------------------------------------------------------------------------- #
# Inference sweep
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("window,stride", [(2, 1), (2, 2), (4, 2), (6, 3), (8, 4)])
def test_sweep_covers_every_patch(window, stride):
    """No patch may go unscored, or its anomaly value is whatever the
    accumulator was initialised to rather than a measurement."""
    bank = SweepMaskBank(GRID, windows=(window,), strides=(stride,))
    cov = coverage_map(bank, window)
    assert cov.min() >= 1, f"window={window} stride={stride} left a patch unscored"


def test_sweep_positions_include_the_final_row_and_column():
    """A stride that does not divide the grid must still reach the far edge."""
    pos = sweep_positions(GRID, window=6, stride=5)
    tops = {r for r, _ in pos}
    assert max(tops) == GRID - 6


def test_sweep_target_and_context_are_disjoint():
    bank = SweepMaskBank(GRID, windows=(2, 4, 6))
    for window in bank.scales():
        for spec in bank[window]:
            ctx = set(spec.context_idx.tolist())
            tgt = set(spec.target_blocks[0].tolist())
            assert not (ctx & tgt)
            assert len(ctx) + len(tgt) == GRID * GRID


def test_sweep_batching_preserves_shapes_and_count():
    bank = SweepMaskBank(GRID, windows=(4,), strides=(2,))
    total = 0
    for ctx, tgt, boxes in bank.batched(4, chunk=8):
        assert ctx.dim() == 2 and tgt.dim() == 2
        assert ctx.size(0) == tgt.size(0) == len(boxes)
        assert tgt.size(1) == 16              # 4x4 window
        total += ctx.size(0)
    assert total == len(bank[4])
