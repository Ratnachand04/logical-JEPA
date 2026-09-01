"""Tests for decoupled detection / localization normalization (Phase 2c).

Study 4 measured that the two jobs want opposite normalizations:

===================  ==============  =============
normalization        image AUROC     pixel AUROC
===================  ==============  =============
``global``           0.762           0.615
``zscore``           0.461           0.876
===================  ==============  =============

Rather than compromise, the scorer computes both maps and each metric is read
off the map built for its purpose. The tests below make sure the two paths stay
genuinely distinct and that the detection map -- the one the verdict depends on
-- is never silently swapped for the localization map.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly.scoring import AnomalyScorer, build_scorer  # noqa: E402
from masking.sweep_mask import SweepMaskBank  # noqa: E402
from models import build_model  # noqa: E402
from utils.config import Config  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model():
    cfg = Config({
        "data": {"img_size": 256, "patch_size": 16},
        "encoder": {"embed_dim": 64, "depth": 2, "num_heads": 4},
        "predictor": {"predictor_dim": 64, "depth": 2, "num_heads": 4},
        "loss": {"kind": "cosine"},
    })
    return build_model(cfg).to(DEVICE)


@pytest.fixture(scope="module")
def bank(model):
    return SweepMaskBank(model.grid_size, windows=(4,), device=DEVICE)


def _scorer(model, bank, detection="global", localization=None, calibrated=False):
    """Build a scorer, optionally with per-position statistics already fitted.

    ``global`` deliberately falls back to ``zscore`` until it has been
    calibrated, so any test that needs the two modes to *differ* must supply
    statistics -- otherwise it would be comparing zscore against itself.
    """
    scorer = AnomalyScorer(
        model, bank, out_size=256, normalize=detection,
        normalize_localization=localization, chunk=16,
    )
    if calibrated:
        import numpy as np

        grid = model.grid_size
        scorer.calibration.scale_stats = {
            window: (np.full((grid, grid), 0.5, np.float32),
                     np.full((grid, grid), 0.1, np.float32))
            for window in bank.scales()
        }
    return scorer


# --------------------------------------------------------------------------- #
# The two paths exist and are distinct
# --------------------------------------------------------------------------- #
def test_score_batch_returns_both_maps(model, bank):
    out = _scorer(model, bank, "global", "zscore").score_batch(
        torch.randn(2, 3, 256, 256, device=DEVICE)
    )
    for key in ("scores", "maps", "detection_maps", "localization_maps"):
        assert key in out, f"missing {key}"
    assert out["detection_maps"].shape == out["localization_maps"].shape


def test_the_two_maps_differ_when_the_settings_differ(model, bank):
    out = _scorer(model, bank, "global", "zscore", calibrated=True).score_batch(
        torch.randn(2, 3, 256, 256, device=DEVICE)
    )
    assert not torch.allclose(out["detection_maps"], out["localization_maps"]), (
        "detection and localization maps are identical despite different modes"
    )


def test_matching_settings_reuse_one_map(model, bank):
    """Equal settings must not pay for the fusion twice."""
    out = _scorer(model, bank, "zscore", "zscore").score_batch(
        torch.randn(1, 3, 256, 256, device=DEVICE)
    )
    assert out["detection_maps"] is out["localization_maps"]


def test_scores_come_from_the_detection_map(model, bank):
    """The verdict must never be computed from the localization map."""
    from anomaly.scoring import aggregate_score

    scorer = _scorer(model, bank, "global", "zscore")
    out = scorer.score_batch(torch.randn(2, 3, 256, 256, device=DEVICE))

    expected = aggregate_score(out["detection_maps"], scorer.aggregation,
                               scorer.top_k_ratio)
    assert torch.allclose(out["scores"], expected)


def test_maps_alias_stays_the_detection_map(model, bank):
    """`maps` is the legacy key; older callers must keep getting detection."""
    out = _scorer(model, bank, "global", "zscore").score_batch(
        torch.randn(1, 3, 256, 256, device=DEVICE)
    )
    assert torch.equal(out["maps"], out["detection_maps"])


def test_localization_choice_does_not_change_the_image_score(model, bank):
    """Changing the heatmap setting must not move the verdict.

    This is the guarantee that makes the decoupling safe: a presentation-layer
    choice cannot alter a reported detection number.
    """
    images = torch.randn(2, 3, 256, 256, device=DEVICE)
    a = _scorer(model, bank, "global", "zscore", calibrated=True).score_batch(images)["scores"]
    b = _scorer(model, bank, "global", "median", calibrated=True).score_batch(images)["scores"]
    c = _scorer(model, bank, "global", "none", calibrated=True).score_batch(images)["scores"]

    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a, c, atol=1e-6)


def test_detection_choice_does_change_the_image_score(model, bank):
    """Sanity check the previous test is not passing vacuously."""
    images = torch.randn(2, 3, 256, 256, device=DEVICE)
    a = _scorer(model, bank, "global", "zscore", calibrated=True).score_batch(images)["scores"]
    b = _scorer(model, bank, "zscore", "zscore", calibrated=True).score_batch(images)["scores"]
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# Defaults and config wiring
# --------------------------------------------------------------------------- #
def test_localization_defaults_to_the_detection_setting(model, bank):
    scorer = _scorer(model, bank, "global", None)
    assert scorer.normalize_localization == "global"
    assert scorer.normalize_detection == "global"


def test_builder_reads_the_new_keys(model, bank):
    cfg = Config({
        "data": {"img_size": 256},
        "anomaly": {"normalize_detection": "global", "normalize_localization": "zscore"},
    })
    scorer = build_scorer(cfg, model, bank)
    assert scorer.normalize_detection == "global"
    assert scorer.normalize_localization == "zscore"


def test_builder_falls_back_to_the_legacy_key(model, bank):
    """Checkpoints saved before this change carry only `anomaly.normalize`."""
    cfg = Config({"data": {"img_size": 256}, "anomaly": {"normalize": "median"}})
    scorer = build_scorer(cfg, model, bank)
    assert scorer.normalize_detection == "median"
    assert scorer.normalize_localization == "median"


def test_explicit_detection_key_wins_over_the_legacy_one(model, bank):
    cfg = Config({
        "data": {"img_size": 256},
        "anomaly": {"normalize": "none", "normalize_detection": "global"},
    })
    assert build_scorer(cfg, model, bank).normalize_detection == "global"


def test_score_loader_carries_both_map_stacks(model, bank):
    """evaluate.py reads `localization_maps` for the pixel metrics."""
    from torch.utils.data import DataLoader, Dataset

    class _Tiny(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {
                "image": torch.randn(3, 256, 256),
                "label": torch.tensor(i % 2, dtype=torch.long),
                "defect_type": "good" if i % 2 == 0 else "logical_anomalies",
                "mask": torch.zeros(1, 256, 256),
            }

    scorer = _scorer(model, bank, "global", "zscore")
    out = scorer.score_loader(DataLoader(_Tiny(), batch_size=2), DEVICE,
                              collect_maps=True)

    assert "maps" in out and "localization_maps" in out
    assert out["maps"].shape == out["localization_maps"].shape
    assert out["scores"].shape == (2,)


def test_uncalibrated_global_falls_back_to_zscore(model, bank):
    """Documented behaviour: without statistics, `global` cannot be applied.

    Falling back keeps a freshly built scorer usable instead of crashing, and
    this test pins that it is a *fallback* rather than a silent no-op.
    """
    images = torch.randn(2, 3, 256, 256, device=DEVICE)
    uncalibrated = _scorer(model, bank, "global", "global").score_batch(images)
    zscore = _scorer(model, bank, "zscore", "zscore").score_batch(images)
    assert torch.allclose(uncalibrated["detection_maps"], zscore["detection_maps"],
                          atol=1e-5)
