"""Tests for the Logical-JEPA model itself.

Two of these encode project *constraints* rather than ordinary correctness, and
they exist so a future change cannot quietly break the research claim:

* :func:`test_no_pretrained_weights_are_loaded` -- every parameter must be
  randomly initialised. Silently introducing a pretrained backbone would
  invalidate the headline result rather than improve it.
* :func:`test_teacher_receives_no_gradient` -- the EMA teacher must stay
  detached. If gradient reached it, the objective would collapse to the
  trivial constant solution and the loss would still look like it was falling.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly.embedding_error import DISTANCES, jepa_loss, patch_distance  # noqa: E402
from masking import build_mask_generator  # noqa: E402
from masking.sweep_mask import SweepMaskBank  # noqa: E402
from models import build_model  # noqa: E402
from models.patch_embed import PatchEmbed, build_2d_sincos_pos_embed  # noqa: E402
from models.target_encoder import normalize_targets  # noqa: E402
from utils.config import Config  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_cfg(**overrides):
    """A small but structurally faithful model, so tests stay fast."""
    cfg = Config({
        "data": {"img_size": 256, "patch_size": 16},
        "encoder": {"embed_dim": 64, "depth": 2, "num_heads": 4},
        "predictor": {"predictor_dim": 64, "depth": 2, "num_heads": 4},
        "loss": {"kind": "cosine", "target_norm": "layernorm"},
        "ema": {"base_momentum": 0.996, "final_momentum": 1.0},
    })
    for key, value in overrides.items():
        cfg.set_path(key, value)
    return cfg


@pytest.fixture(scope="module")
def model():
    return build_model(_tiny_cfg()).to(DEVICE)


# --------------------------------------------------------------------------- #
# Project constraints
# --------------------------------------------------------------------------- #
def test_no_pretrained_weights_are_loaded():
    """Every *randomly initialised* tensor must differ across two builds.

    Only weight matrices (Linear / Conv, ``dim >= 2``) are sampled from a
    distribution. Biases are zeroed and LayerNorm gains are set to one by the
    standard ViT initialisation, so those are deterministic *by design* and are
    checked separately in
    :func:`test_norm_and_bias_use_the_canonical_scratch_init`.
    """
    a = build_model(_tiny_cfg())
    b = build_model(_tiny_cfg())

    compared = 0
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        if pa.dim() >= 2:                       # weight matrices only
            compared += 1
            assert not torch.allclose(pa, pb), f"{na} identical across inits"

    assert compared > 20, f"only {compared} weight matrices inspected"


def test_norm_and_bias_use_the_canonical_scratch_init():
    """LayerNorm gains are exactly 1 and biases exactly 0 at initialisation.

    This is the fingerprint of a from-scratch build. Any pretrained checkpoint
    would carry trained, non-trivial values here, so this doubles as a guard
    against a backbone being loaded somewhere upstream.
    """
    model = build_model(_tiny_cfg())

    checked_norm = checked_bias = 0
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            checked_bias += 1
            assert torch.count_nonzero(param) == 0, f"{name} is not zero-initialised"
        elif "norm" in name and param.dim() == 1:
            checked_norm += 1
            assert torch.allclose(param, torch.ones_like(param)), f"{name} != 1"

    assert checked_norm > 0 and checked_bias > 0


def test_teacher_receives_no_gradient(model):
    for param in model.target_encoder.parameters():
        assert not param.requires_grad, "teacher parameter is trainable"


def test_teacher_is_excluded_from_the_optimiser_parameter_list(model):
    trainable = {id(p) for p in model.trainable_parameters()}
    for param in model.target_encoder.parameters():
        assert id(param) not in trainable


def test_teacher_stays_in_eval_mode_even_when_parent_trains(model):
    model.train()
    assert not model.target_encoder.encoder.training, (
        "teacher dropout/droppath active would inject noise into the target"
    )


# --------------------------------------------------------------------------- #
# Shapes and wiring
# --------------------------------------------------------------------------- #
def test_patch_embed_token_count():
    pe = PatchEmbed(img_size=256, patch_size=16, embed_dim=32)
    assert pe.grid_size == 16 and pe.num_patches == 256
    out = pe(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, 256, 32)


def test_patch_embed_rejects_mismatched_input():
    pe = PatchEmbed(img_size=256, patch_size=16, embed_dim=32)
    with pytest.raises(ValueError):
        pe(torch.randn(1, 3, 128, 128))


@pytest.mark.parametrize("img_size,patch,expected", [(256, 16, 16), (384, 16, 24), (512, 16, 32)])
def test_patch_embed_supports_larger_resolutions(img_size, patch, expected):
    pe = PatchEmbed(img_size=img_size, patch_size=patch, embed_dim=32)
    assert pe.grid_size == expected
    out = pe(torch.randn(1, 3, img_size, img_size))
    assert out.shape == (1, expected * expected, 32)


def test_positional_embedding_is_deterministic_and_unique_per_position():
    a = build_2d_sincos_pos_embed(64, 16)
    b = build_2d_sincos_pos_embed(64, 16)
    assert torch.equal(a, b), "fixed sin-cos table must be deterministic"

    flat = a[0]
    # No two grid positions may share an encoding, or the predictor cannot tell
    # them apart when asked to imagine a region.
    dists = torch.cdist(flat, flat)
    dists.fill_diagonal_(float("inf"))
    assert dists.min() > 1e-4


def test_forward_produces_a_finite_scalar_loss(model):
    gen = build_mask_generator({"strategy": "multiscale"}, model.grid_size, seed=0)
    images = torch.randn(2, 3, 256, 256, device=DEVICE)
    loss, stats = model(images, gen())

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0
    for key in ("loss", "cos_sim", "target_std", "masked_ratio", "num_blocks"):
        assert key in stats


@pytest.mark.parametrize("strategy", ["random_patch", "small", "large", "multiscale"])
def test_forward_works_for_every_masking_strategy(model, strategy):
    gen = build_mask_generator({"strategy": strategy}, model.grid_size, seed=0)
    images = torch.randn(2, 3, 256, 256, device=DEVICE)
    loss, _ = model(images, gen())
    assert torch.isfinite(loss)


def test_gradients_reach_student_and_predictor_but_not_teacher(model):
    gen = build_mask_generator({"strategy": "small"}, model.grid_size, seed=0)
    images = torch.randn(2, 3, 256, 256, device=DEVICE)

    model.zero_grad(set_to_none=True)
    loss, _ = model(images, gen())
    loss.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.context_encoder.parameters()), "student got no gradient"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.predictor.parameters()), "predictor got no gradient"
    assert all(p.grad is None for p in model.target_encoder.parameters()), \
        "teacher accumulated a gradient"


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
def test_ema_starts_identical_to_the_student():
    m = build_model(_tiny_cfg())
    for tp, sp in zip(m.target_encoder.encoder.parameters(), m.context_encoder.parameters()):
        assert torch.allclose(tp, sp)


def test_ema_moves_a_known_fraction_towards_the_student():
    m = build_model(_tiny_cfg())
    with torch.no_grad():
        for p in m.context_encoder.parameters():
            p.add_(1.0)                       # student is now student0 + 1

    before = [p.clone() for p in m.target_encoder.encoder.parameters()]
    m.target_encoder.update(m.context_encoder, momentum=0.9)

    # theta_t <- 0.9*theta_t + 0.1*theta_c, and theta_c = theta_t + 1
    for old, new in zip(before, m.target_encoder.encoder.parameters()):
        assert torch.allclose(new, old + 0.1, atol=1e-6)


def test_ema_momentum_schedule_is_monotone_to_one():
    m = build_model(_tiny_cfg())
    values = [m.target_encoder.momentum_at(s, 100) for s in range(0, 101, 10)]
    assert values[0] == pytest.approx(0.996)
    assert values[-1] == pytest.approx(1.0)
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:])), "schedule not monotone"


# --------------------------------------------------------------------------- #
# Loss / distance semantics
# --------------------------------------------------------------------------- #
def test_cosine_distance_endpoints():
    x = torch.randn(2, 5, 16)
    assert patch_distance(x, x.clone()).abs().max() < 1e-5
    assert patch_distance(x, -x).mean() == pytest.approx(2.0, abs=1e-4)


@pytest.mark.parametrize("kind", DISTANCES)
def test_every_distance_is_non_negative_and_right_shaped(kind):
    a, b = torch.randn(3, 7, 32), torch.randn(3, 7, 32)
    d = patch_distance(a, b, kind=kind)
    assert d.shape == (3, 7)
    assert torch.isfinite(d).all()
    assert (d >= -1e-6).all()


def test_jepa_loss_weights_blocks_by_patch_count():
    """Default reduction is per-patch, so a big block dominates a small one."""
    big_pred = torch.zeros(1, 36, 8)
    big_tgt = torch.zeros(1, 36, 8)
    big_pred[..., 0], big_tgt[..., 0] = 1.0, 1.0          # cosine distance 0

    small_pred = torch.zeros(1, 4, 8)
    small_tgt = torch.zeros(1, 4, 8)
    small_pred[..., 0], small_tgt[..., 0] = 1.0, -1.0     # cosine distance 2

    loss, stats = jepa_loss([big_pred, small_pred], [big_tgt, small_tgt])
    expected = (0.0 * 36 + 2.0 * 4) / 40
    assert loss.item() == pytest.approx(expected, abs=1e-5)
    assert stats["n_target_patches"] == 40


def test_target_layernorm_standardises_each_token():
    raw = torch.randn(2, 5, 64) * 9.0 + 4.0
    out = normalize_targets(raw, "layernorm")
    assert out.mean(dim=-1).abs().max() < 1e-4
    assert (out.std(dim=-1, unbiased=False) - 1.0).abs().max() < 1e-2


def test_target_norm_none_is_a_passthrough():
    raw = torch.randn(2, 3, 16)
    assert torch.equal(normalize_targets(raw, "none"), raw)


# --------------------------------------------------------------------------- #
# Inference sweep
# --------------------------------------------------------------------------- #
def test_anomaly_grids_have_one_value_per_patch(model):
    bank = SweepMaskBank(model.grid_size, windows=(4,), device=DEVICE)
    images = torch.randn(2, 3, 256, 256, device=DEVICE)
    grids = model.anomaly_grids(images, bank, chunk=8)

    assert set(grids) == {4}
    assert grids[4].shape == (2, model.grid_size, model.grid_size)
    assert torch.isfinite(grids[4]).all()


def test_anomaly_grids_are_deterministic_in_eval(model):
    bank = SweepMaskBank(model.grid_size, windows=(4,), device=DEVICE)
    images = torch.randn(1, 3, 256, 256, device=DEVICE)
    a = model.anomaly_grids(images, bank, chunk=8)[4]
    b = model.anomaly_grids(images, bank, chunk=8)[4]
    assert torch.allclose(a, b, atol=1e-5), "sweep is not deterministic at eval"


def test_sweep_chunking_does_not_change_the_result(model):
    """Chunk size is a memory knob and must never alter the measurement."""
    bank = SweepMaskBank(model.grid_size, windows=(4,), device=DEVICE)
    images = torch.randn(1, 3, 256, 256, device=DEVICE)
    small = model.anomaly_grids(images, bank, chunk=4)[4]
    large = model.anomaly_grids(images, bank, chunk=32)[4]
    assert torch.allclose(small, large, atol=1e-5)


def test_sweep_runs_under_no_grad(model):
    bank = SweepMaskBank(model.grid_size, windows=(2,), device=DEVICE)
    images = torch.randn(1, 3, 256, 256, device=DEVICE)
    grids = model.anomaly_grids(images, bank, chunk=16)
    assert not grids[2].requires_grad
