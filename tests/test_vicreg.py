"""Tests for the VICReg collapse regulariser (Phase 3d).

The regulariser has to do two things and no more: punish a collapsed
representation, and leave a healthy one alone. The second half matters as much
as the first -- a term that keeps pulling once the embedding is fine is
actively fighting the JEPA objective, which is why the variance term is hinged
rather than quadratic.

:func:`test_collapse_report_detects_a_collapsed_representation` keeps the
original ``target_std``-style monitor honest; it is the regulariser's own unit
test at training time.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from masking import build_mask_generator  # noqa: E402
from models import build_model  # noqa: E402
from models.vicreg import (  # noqa: E402
    build_vicreg,
    collapse_report,
    covariance_loss,
    variance_loss,
    vicreg_loss,
)
from utils.config import Config  # noqa: E402


# --------------------------------------------------------------------------- #
# Variance term
# --------------------------------------------------------------------------- #
def test_variance_term_punishes_a_constant_embedding():
    """The textbook collapse: every sample identical.

    The loss approaches but never reaches ``gamma``: the epsilon inside the
    sqrt floors the measured std at ``sqrt(eps)``, so the hinge tops out at
    ``gamma - sqrt(eps)``.
    """
    collapsed = torch.ones(64, 32)
    expected = 1.0 - math.sqrt(1e-4)
    assert variance_loss(collapsed, gamma=1.0).item() == pytest.approx(expected, abs=1e-3)


def test_variance_term_is_zero_for_a_healthy_embedding():
    """Hinged, not quadratic: it must stop pulling once the floor is cleared."""
    healthy = torch.randn(4096, 32) * 3.0
    assert variance_loss(healthy, gamma=1.0).item() == pytest.approx(0.0, abs=1e-4)


def test_variance_term_increases_as_variance_shrinks():
    losses = [variance_loss(torch.randn(2048, 16) * s, gamma=1.0).item()
              for s in (1.5, 0.8, 0.4, 0.05)]
    assert all(b >= a - 1e-6 for a, b in zip(losses, losses[1:])), losses
    assert losses[-1] > losses[0]


def test_variance_term_only_penalises_the_dead_dimensions():
    """A single dead channel among healthy ones costs ~1/D, not the full hinge."""
    x = torch.randn(2048, 10) * 3.0
    x[:, 0] = 0.5                                  # one constant dimension
    loss = variance_loss(x, gamma=1.0).item()
    assert 0.02 < loss < 0.15


def test_variance_term_is_safe_on_a_single_sample():
    assert variance_loss(torch.randn(1, 8)).item() == 0.0


# --------------------------------------------------------------------------- #
# Covariance term
# --------------------------------------------------------------------------- #
def test_covariance_term_is_near_zero_for_independent_dimensions():
    x = torch.randn(20000, 8)
    assert covariance_loss(x).item() < 0.05


def test_covariance_term_punishes_duplicated_dimensions():
    """Variance can be satisfied by one direction copied across channels."""
    base = torch.randn(4096, 1)
    duplicated = base.repeat(1, 8)
    assert covariance_loss(duplicated).item() > covariance_loss(torch.randn(4096, 8)).item()


def test_covariance_term_ignores_the_diagonal():
    """Scaling a single dimension changes its variance, not its correlations."""
    x = torch.randn(8192, 6)
    x_scaled = x.clone()
    x_scaled[:, 0] *= 5.0
    assert covariance_loss(x).item() < 0.1
    assert covariance_loss(x_scaled).item() < 0.5


def test_covariance_term_is_safe_on_a_single_sample():
    assert covariance_loss(torch.randn(1, 8)).item() == 0.0


# --------------------------------------------------------------------------- #
# Combined
# --------------------------------------------------------------------------- #
def test_vicreg_reports_both_terms_separately():
    """A combined number hides which term is actually doing the work."""
    _, stats = vicreg_loss(torch.randn(512, 16))
    for key in ("vicreg_var", "vicreg_cov", "vicreg_total", "embed_std"):
        assert key in stats


def test_vicreg_accepts_token_shaped_input():
    loss_3d, _ = vicreg_loss(torch.randn(4, 32, 16))
    loss_2d, _ = vicreg_loss(torch.randn(128, 16))
    assert torch.isfinite(loss_3d) and torch.isfinite(loss_2d)


def test_vicreg_is_larger_for_a_collapsed_embedding():
    collapsed, _ = vicreg_loss(torch.ones(256, 16) + torch.randn(256, 16) * 1e-4)
    healthy, _ = vicreg_loss(torch.randn(256, 16) * 2.0)
    assert collapsed.item() > healthy.item()


def test_vicreg_weights_scale_their_terms():
    x = torch.ones(256, 16)                        # collapsed -> variance active
    low, _ = vicreg_loss(x, var_weight=1.0, cov_weight=0.0)
    high, _ = vicreg_loss(x, var_weight=10.0, cov_weight=0.0)
    assert high.item() == pytest.approx(10.0 * low.item(), rel=1e-4)


def test_vicreg_gradient_pushes_variance_up():
    """The term must be optimisable, not merely measurable."""
    x = (torch.full((256, 8), 0.5) + torch.randn(256, 8) * 1e-3).requires_grad_(True)
    loss, _ = vicreg_loss(x, var_weight=1.0, cov_weight=0.0)
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_variance_gradient_vanishes_at_exact_symmetric_collapse():
    """A real limitation of VICReg, recorded so it is not rediscovered later.

    When every sample is *exactly* identical the batch variance is 0 and
    d(var)/dx = 2(x - mean)/n is identically 0, so the variance term produces no
    gradient at all. VICReg cannot escape a perfectly symmetric collapse; it can
    only prevent drifting into one.

    In practice the embedding always carries some noise, so this is a boundary
    case rather than a live failure -- but it is the reason the EMA teacher, not
    VICReg, remains the primary anti-collapse mechanism.
    """
    x = torch.full((256, 8), 0.5, requires_grad=True)
    loss, _ = vicreg_loss(x, var_weight=1.0, cov_weight=0.0)
    loss.backward()
    assert x.grad.abs().sum() == 0.0


# --------------------------------------------------------------------------- #
# Collapse diagnostics -- the regulariser's own unit test
# --------------------------------------------------------------------------- #
def test_collapse_report_detects_a_constant_representation():
    """Constant embedding: caught by `embed_std` and `dead_dim_frac`.

    Note `effective_rank` does NOT fire here, and correctly so -- it is
    scale-invariant (eigenvalues are normalised), so isotropic noise of
    magnitude 1e-5 still has close to full rank. The two diagnostics detect
    different collapses, which is why both are reported.
    """
    report = collapse_report(torch.ones(256, 32) + torch.randn(256, 32) * 1e-5)
    assert report["embed_std"] < 0.01
    assert report["dead_dim_frac"] > 0.9


def test_collapse_report_detects_a_low_rank_representation():
    """Directional collapse: variance is healthy but lives in one direction.

    This is the failure `dead_dim_frac` misses and `effective_rank` catches --
    the case the covariance term exists to prevent.
    """
    factor = torch.randn(4096, 1)
    directional = factor @ torch.randn(1, 32)
    report = collapse_report(directional)

    assert report["embed_std"] > 0.1, "variance is not the problem here"
    # A random projection leaves a couple of near-zero coefficients, so a few
    # dimensions read as dead; the point is that this diagnostic stays mostly
    # quiet while the rank one fires loudly.
    assert report["dead_dim_frac"] < 0.2, "dead_dim_frac should barely react"
    assert report["effective_rank"] < 2.5, "rank collapse not detected"


def test_collapse_report_is_clean_for_a_healthy_representation():
    report = collapse_report(torch.randn(4096, 32) * 2.0)
    assert report["embed_std"] > 1.0
    assert report["dead_dim_frac"] == pytest.approx(0.0)
    assert report["effective_rank"] > 20.0


def test_effective_rank_tracks_the_true_rank():
    """A low-rank embedding must report a low effective rank."""
    basis = torch.randn(4096, 3)
    low_rank = basis @ torch.randn(3, 32)
    assert collapse_report(low_rank)["effective_rank"] < 6.0


def test_effective_rank_is_bounded_by_the_dimension():
    report = collapse_report(torch.randn(4096, 16))
    assert 0.0 <= report["effective_rank"] <= 16.0 + 1e-6


# --------------------------------------------------------------------------- #
# Config wiring
# --------------------------------------------------------------------------- #
def test_disabled_vicreg_builds_to_none():
    assert build_vicreg({"vicreg": {"enabled": False}}) is None
    assert build_vicreg({}) is None
    assert build_vicreg(None) is None


def test_enabled_vicreg_reads_its_weights():
    cfg = build_vicreg({"vicreg": {"enabled": True, "var_weight": 2.0,
                                   "cov_weight": 0.5, "gamma": 0.8}})
    assert cfg == {"var_weight": 2.0, "cov_weight": 0.5, "gamma": 0.8}


def _model(vicreg_enabled: bool):
    return build_model(Config({
        "data": {"img_size": 256, "patch_size": 16},
        "encoder": {"embed_dim": 64, "depth": 2, "num_heads": 4},
        "predictor": {"predictor_dim": 64, "depth": 2, "num_heads": 4},
        "loss": {"kind": "cosine"},
        "regularizer": {"vicreg": {"enabled": vicreg_enabled}},
    }))


def test_model_runs_with_vicreg_enabled():
    model = _model(True)
    assert model.vicreg is not None

    gen = build_mask_generator({"strategy": "small"}, model.grid_size, seed=0)
    loss, stats = model(torch.randn(2, 3, 256, 256), gen())

    assert torch.isfinite(loss)
    for key in ("vicreg_var", "vicreg_cov", "dead_dim_frac", "effective_rank"):
        assert key in stats, f"{key} missing from training stats"


def test_model_skips_vicreg_when_disabled():
    model = _model(False)
    assert model.vicreg is None

    gen = build_mask_generator({"strategy": "small"}, model.grid_size, seed=0)
    _, stats = model(torch.randn(2, 3, 256, 256), gen())
    assert "vicreg_var" not in stats


def test_vicreg_adds_to_the_loss_rather_than_replacing_it():
    """The predictive objective must survive; VICReg is additive and small."""
    torch.manual_seed(0)
    plain = _model(False)
    torch.manual_seed(0)
    regularised = _model(True)

    gen_a = build_mask_generator({"strategy": "small"}, plain.grid_size, seed=0)
    gen_b = build_mask_generator({"strategy": "small"}, regularised.grid_size, seed=0)
    images = torch.randn(2, 3, 256, 256)

    loss_plain, _ = plain(images, gen_a())
    loss_reg, stats = regularised(images, gen_b())

    assert loss_reg.item() >= loss_plain.item() - 1e-4
    assert loss_reg.item() == pytest.approx(
        loss_plain.item() + stats["vicreg_total"], abs=5e-3
    )


def test_gradients_still_flow_with_vicreg_active():
    model = _model(True)
    gen = build_mask_generator({"strategy": "small"}, model.grid_size, seed=0)

    model.zero_grad(set_to_none=True)
    loss, _ = model(torch.randn(2, 3, 256, 256), gen())
    loss.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.context_encoder.parameters())
    # The teacher must remain untouched even with an extra loss term.
    assert all(p.grad is None for p in model.target_encoder.parameters())
