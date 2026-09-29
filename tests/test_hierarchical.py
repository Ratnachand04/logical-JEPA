"""Tests for the hierarchical slot loss (Phase 3b).

The constraints that matter here are about *where gradient goes*:

* the slot term must reach the predictor, or it trains nothing;
* it must NOT reach the slot module -- otherwise the cheapest way to shrink it
  is to flatten the grouping until every slot set looks alike;
* at lambda 0 it must not touch the JEPA at all, so the Study 8 control arm is
  a genuine control.

And about the comparison itself: slot order is arbitrary, so the distance must
be permutation-invariant.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from masking.base import complement_indices  # noqa: E402
from models import build_model  # noqa: E402
from models.slot_bottleneck import chamfer_slot_distance  # noqa: E402
from utils.config import Config  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID = 16


def _cfg(**slots):
    node = {"enabled": True, "num_slots": 4, "iters": 2,
            "cardinality": {"enabled": False}}
    node.update(slots)
    return Config({
        "data": {"img_size": 256, "patch_size": 16},
        "encoder": {"embed_dim": 64, "depth": 2, "num_heads": 4},
        "predictor": {"predictor_dim": 64, "depth": 2, "num_heads": 4},
        "loss": {"kind": "cosine"},
        "slots": node,
    })


def _inputs(model, B=2):
    """Teacher tokens, one 2x2 target block, its context, and leaf predictions."""
    torch.manual_seed(0)
    D = model.embed_dim
    block = torch.tensor([0, 1, GRID, GRID + 1], device=DEVICE)
    ctx = complement_indices(GRID, [block.cpu()]).to(DEVICE)
    full_target = torch.randn(B, GRID * GRID, D, device=DEVICE)
    preds = torch.randn(B, block.numel(), D, device=DEVICE, requires_grad=True)
    return (full_target, ctx.unsqueeze(0).expand(B, -1),
            [block.unsqueeze(0).expand(B, -1)], [preds])


def _slot_grad_norm(model) -> float:
    return sum(float(p.grad.abs().sum()) for p in model.slots.parameters()
               if p.grad is not None)


# --------------------------------------------------------------------------- #
# The set distance
# --------------------------------------------------------------------------- #
def test_chamfer_is_zero_for_identical_sets():
    a = torch.randn(3, 5, 16)
    assert torch.allclose(chamfer_slot_distance(a, a), torch.zeros(3), atol=1e-6)


def test_chamfer_is_permutation_invariant():
    """Slot i is never compared with slot i -- reordering must not matter."""
    a, b = torch.randn(2, 6, 16), torch.randn(2, 6, 16)
    perm = torch.randperm(6)
    assert torch.allclose(chamfer_slot_distance(a, b),
                          chamfer_slot_distance(a, b[:, perm]), atol=1e-6)
    assert torch.allclose(chamfer_slot_distance(a, b),
                          chamfer_slot_distance(a[:, perm], b), atol=1e-6)


def test_chamfer_is_symmetric():
    a, b = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    assert torch.allclose(chamfer_slot_distance(a, b), chamfer_slot_distance(b, a))


def test_chamfer_penalises_an_unmatched_slot():
    """A set that gained a component must be further away than a copy."""
    a = torch.eye(4).unsqueeze(0)                       # 4 orthogonal slots
    b = a.clone()
    b[0, 3] = torch.tensor([1.0, 1.0, 0.0, 0.0])        # slot 3 now duplicates 0/1
    assert chamfer_slot_distance(a, b) > chamfer_slot_distance(a, a) + 1e-3


# --------------------------------------------------------------------------- #
# Lambda schedule
# --------------------------------------------------------------------------- #
def test_lambda_is_zero_before_start_then_ramps():
    model = build_model(_cfg(hierarchical_weight=0.5, hierarchical_start=0.3,
                             hierarchical_ramp=0.2))
    model.set_progress(0.1)
    assert model.hierarchical_lambda() == 0.0
    model.set_progress(0.4)
    assert model.hierarchical_lambda() == pytest.approx(0.25)
    model.set_progress(0.9)
    assert model.hierarchical_lambda() == pytest.approx(0.5)


def test_lambda_zero_weight_stays_zero():
    model = build_model(_cfg(hierarchical_weight=0.0))
    model.set_progress(1.0)
    assert model.hierarchical_lambda() == 0.0


# --------------------------------------------------------------------------- #
# Gradient routing -- the design constraints
# --------------------------------------------------------------------------- #
def test_slot_term_reaches_the_predictions():
    model = build_model(_cfg(hierarchical_weight=1.0, recon_weight=0.0)).to(DEVICE)
    model.set_progress(1.0)
    full_target, ctx, blocks, preds = _inputs(model)

    loss, stats = model._slot_losses(full_target, ctx, blocks, preds)
    loss.backward()

    assert preds[0].grad is not None and preds[0].grad.abs().sum() > 0
    assert stats["hier_lambda"] == pytest.approx(1.0)


def test_slot_term_does_not_train_the_slot_module():
    """Otherwise the module could flatten its grouping to zero the term."""
    model = build_model(_cfg(hierarchical_weight=1.0, recon_weight=0.0)).to(DEVICE)
    model.set_progress(1.0)
    full_target, ctx, blocks, preds = _inputs(model)

    loss, _ = model._slot_losses(full_target, ctx, blocks, preds)
    loss.backward()

    assert _slot_grad_norm(model) == 0.0


def test_lambda_zero_leaves_the_jepa_untouched():
    """The Study 8 control: slot stage present, zero effect on the JEPA."""
    model = build_model(_cfg(hierarchical_weight=0.0, recon_weight=1.0)).to(DEVICE)
    model.set_progress(1.0)
    full_target, ctx, blocks, preds = _inputs(model)

    loss, stats = model._slot_losses(full_target, ctx, blocks, preds)
    loss.backward()

    assert preds[0].grad is None
    assert "loss_hier" in stats                     # still logged, for comparison


def test_reconstruction_trains_the_slot_module():
    model = build_model(_cfg(hierarchical_weight=0.0, recon_weight=1.0)).to(DEVICE)
    full_target, ctx, blocks, preds = _inputs(model)

    loss, _ = model._slot_losses(full_target, ctx, blocks, preds)
    loss.backward()

    assert _slot_grad_norm(model) > 0.0


# --------------------------------------------------------------------------- #
# Integration and backwards compatibility
# --------------------------------------------------------------------------- #
def test_training_step_reports_slot_stats():
    from masking import build_mask_generator

    model = build_model(_cfg(hierarchical_weight=0.5)).to(DEVICE)
    model.set_progress(1.0)
    gen = build_mask_generator({"strategy": "multiscale"}, grid_size=GRID, seed=0)
    loss, stats = model(torch.randn(2, 3, 256, 256, device=DEVICE), gen())

    assert torch.isfinite(loss)
    for key in ("loss_slot_recon", "loss_hier", "hier_lambda", "slot_entropy_ratio"):
        assert key in stats


def test_disabled_slots_add_no_parameters():
    """Checkpoints trained before Phase 3b must still load into a default model."""
    cfg = _cfg()
    cfg.set_path("slots.enabled", False)
    model = build_model(cfg)
    assert model.slots is None and model.card_head is None
    assert not any(k.startswith(("slots.", "card_head.")) for k in model.state_dict())
    assert model.aux_parameters() == []


def test_aux_parameters_are_disjoint_from_the_jepa():
    model = build_model(_cfg(cardinality={"enabled": True}))
    jepa = {id(p) for p in model.trainable_parameters()}
    aux = {id(p) for p in model.aux_parameters()}
    assert aux and not (jepa & aux)
