"""Tests for the ``per_patch`` / ``per_block`` loss reduction (Phase 2a).

The whole point of adding ``per_block`` is to remove a specific asymmetry:
under ``per_patch`` a block's influence on the gradient scales with its area,
so one 6x6 block (36 patches) outweighs three small blocks (~19 patches
combined) by roughly 2:1. That is the leading explanation for the multi-scale
masking arm losing Study 1.

These tests pin the arithmetic exactly, so the two modes cannot silently become
the same thing (which would make Study 5 a no-op that still produces a table).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly.embedding_error import jepa_loss  # noqa: E402
from masking import build_mask_generator  # noqa: E402
from models import build_model  # noqa: E402
from utils.config import Config  # noqa: E402


def _block(n_patches: int, distance: float, dim: int = 8):
    """A (pred, target) pair whose cosine distance is exactly ``distance``.

    Uses collinear one-hot vectors: identical -> 0, opposite -> 2.
    """
    pred = torch.zeros(1, n_patches, dim)
    target = torch.zeros(1, n_patches, dim)
    pred[..., 0] = 1.0
    target[..., 0] = 1.0 if distance == 0.0 else -1.0
    return pred, target


def test_per_patch_weights_blocks_by_area():
    """One 36-patch block at distance 0, one 4-patch block at distance 2."""
    big_p, big_t = _block(36, 0.0)
    small_p, small_t = _block(4, 2.0)

    loss, stats = jepa_loss([big_p, small_p], [big_t, small_t], reduction="per_patch")

    assert loss.item() == pytest.approx((0.0 * 36 + 2.0 * 4) / 40, abs=1e-5)
    assert stats["reduction"] == "per_patch"


def test_per_block_weights_blocks_equally():
    """Same inputs, but each block now contributes exactly half."""
    big_p, big_t = _block(36, 0.0)
    small_p, small_t = _block(4, 2.0)

    loss, stats = jepa_loss([big_p, small_p], [big_t, small_t], reduction="per_block")

    assert loss.item() == pytest.approx((0.0 + 2.0) / 2, abs=1e-5)
    assert stats["reduction"] == "per_block"


def test_the_two_modes_actually_differ_on_unequal_blocks():
    """Guards against Study 5 becoming a no-op."""
    big_p, big_t = _block(36, 0.0)
    small_p, small_t = _block(4, 2.0)

    per_patch, _ = jepa_loss([big_p, small_p], [big_t, small_t], reduction="per_patch")
    per_block, _ = jepa_loss([big_p, small_p], [big_t, small_t], reduction="per_block")
    assert abs(per_patch.item() - per_block.item()) > 0.5


def test_the_two_modes_agree_when_all_blocks_are_the_same_size():
    """Single-scale strategies must be unaffected -- this is the Study 5 control."""
    a_p, a_t = _block(9, 0.0)
    b_p, b_t = _block(9, 2.0)

    per_patch, _ = jepa_loss([a_p, b_p], [a_t, b_t], reduction="per_patch")
    per_block, _ = jepa_loss([a_p, b_p], [a_t, b_t], reduction="per_block")
    assert per_patch.item() == pytest.approx(per_block.item(), abs=1e-6)


def test_a_single_block_is_identical_under_both_modes():
    pred, target = _block(16, 2.0)
    a, _ = jepa_loss([pred], [target], reduction="per_patch")
    b, _ = jepa_loss([pred], [target], reduction="per_block")
    assert a.item() == pytest.approx(b.item(), abs=1e-6)


def test_per_block_raises_the_weight_of_the_small_scale():
    """The mechanism Study 5 tests, stated as an inequality.

    With the large block easy (distance 0) and the small blocks hard
    (distance 2), moving to per_block must increase the loss -- i.e. the small
    scale now carries more of the gradient.
    """
    large_p, large_t = _block(36, 0.0)
    smalls = [_block(6, 2.0) for _ in range(3)]

    preds = [large_p] + [p for p, _ in smalls]
    targets = [large_t] + [t for _, t in smalls]

    per_patch, _ = jepa_loss(preds, targets, reduction="per_patch")
    per_block, _ = jepa_loss(preds, targets, reduction="per_block")
    assert per_block.item() > per_patch.item()


def test_unknown_reduction_is_rejected():
    pred, target = _block(4, 0.0)
    with pytest.raises(ValueError, match="Unknown loss reduction"):
        jepa_loss([pred], [target], reduction="per_image")


def test_default_reduction_is_per_patch():
    """The historical behaviour every measured README number was produced under."""
    big_p, big_t = _block(36, 0.0)
    small_p, small_t = _block(4, 2.0)

    default, stats = jepa_loss([big_p, small_p], [big_t, small_t])
    explicit, _ = jepa_loss([big_p, small_p], [big_t, small_t], reduction="per_patch")
    assert default.item() == pytest.approx(explicit.item())
    assert stats["reduction"] == "per_patch"


# --------------------------------------------------------------------------- #
# Per-scale logging
# --------------------------------------------------------------------------- #
def test_per_scale_losses_are_reported_separately():
    """A stalled scale must be visible in the logs, not only in final metrics."""
    large_p, large_t = _block(36, 0.0)
    small_p, small_t = _block(4, 2.0)

    _, stats = jepa_loss([large_p, small_p], [large_t, small_t])
    assert stats["loss_small"] == pytest.approx(2.0, abs=1e-4)
    assert stats["loss_large"] == pytest.approx(0.0, abs=1e-4)


def test_single_scale_reports_only_its_own_term():
    small_p, small_t = _block(4, 1.0)
    _, stats = jepa_loss([small_p], [small_t])
    assert "loss_small" in stats and "loss_large" not in stats


# --------------------------------------------------------------------------- #
# Wiring through the model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reduction", ["per_patch", "per_block"])
def test_model_honours_the_configured_reduction(reduction):
    cfg = Config({
        "data": {"img_size": 256, "patch_size": 16},
        "encoder": {"embed_dim": 64, "depth": 2, "num_heads": 4},
        "predictor": {"predictor_dim": 64, "depth": 2, "num_heads": 4},
        "loss": {"kind": "cosine", "target_norm": "layernorm", "reduction": reduction},
        "ema": {"base_momentum": 0.996, "final_momentum": 1.0},
    })
    model = build_model(cfg)
    assert model.loss_reduction == reduction

    gen = build_mask_generator({"strategy": "multiscale"}, model.grid_size, seed=0)
    loss, stats = model(torch.randn(2, 3, 256, 256), gen())

    assert torch.isfinite(loss)
    assert stats["reduction"] == reduction


def test_model_defaults_to_per_patch_when_unset():
    cfg = Config({
        "data": {"img_size": 256, "patch_size": 16},
        "encoder": {"embed_dim": 64, "depth": 2, "num_heads": 4},
        "predictor": {"predictor_dim": 64, "depth": 2, "num_heads": 4},
        "loss": {"kind": "cosine"},
    })
    assert build_model(cfg).loss_reduction == "per_patch"


def test_gradients_flow_under_per_block():
    cfg = Config({
        "data": {"img_size": 256, "patch_size": 16},
        "encoder": {"embed_dim": 64, "depth": 2, "num_heads": 4},
        "predictor": {"predictor_dim": 64, "depth": 2, "num_heads": 4},
        "loss": {"kind": "cosine", "reduction": "per_block"},
    })
    model = build_model(cfg)
    gen = build_mask_generator({"strategy": "multiscale"}, model.grid_size, seed=0)

    model.zero_grad(set_to_none=True)
    loss, _ = model(torch.randn(2, 3, 256, 256), gen())
    loss.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.context_encoder.parameters())
