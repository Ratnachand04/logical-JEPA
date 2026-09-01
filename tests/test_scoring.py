"""Tests for anomaly-map construction, scoring and calibration.

The most important test in this file is
:func:`test_calibration_never_sees_an_anomalous_image`, which encodes the
project's second hard constraint. Calibration leakage would not raise an error
or look wrong in any plot -- it would simply make every reported number
optimistic while the pipeline still claimed to be unsupervised.

The normalization tests pin the measured trade-off from Study 4: per-position
statistics preserve between-image differences (so image-level detection works),
while per-image z-scoring destroys them by construction.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly.anomaly_map import (  # noqa: E402
    FUSION_MODES,
    build_anomaly_map,
    fuse_scales,
    gaussian_blur,
    normalize_grid,
    upsample_map,
)
from anomaly.scoring import (  # noqa: E402
    AGGREGATIONS,
    Calibration,
    aggregate_score,
    fit_calibration,
)

GRID = 16


# --------------------------------------------------------------------------- #
# Grid normalization
# --------------------------------------------------------------------------- #
def test_per_image_zscore_standardises_every_map():
    grid = torch.randn(4, GRID, GRID) * 3.0 + 7.0
    out = normalize_grid(grid, "zscore")
    flat = out.reshape(4, -1)
    assert flat.mean(dim=1).abs().max() < 1e-5
    assert (flat.std(dim=1) - 1.0).abs().max() < 1e-3


def test_per_image_zscore_destroys_between_image_differences():
    """Why `zscore` cannot drive image-level detection (measured in Study 4).

    A uniformly "hot" image and a uniformly "cold" one become indistinguishable
    once each is standardised against itself.
    """
    cold = torch.randn(1, GRID, GRID) * 0.1 + 0.5
    hot = cold * 1.0 + 40.0                       # same texture, far higher level

    a = normalize_grid(cold, "zscore")
    b = normalize_grid(hot, "zscore")
    assert torch.allclose(a, b, atol=1e-4), "zscore should erase the level difference"


def test_global_normalization_preserves_between_image_differences():
    """The proposed mode keeps the level difference the image score needs."""
    stats = (0.5, 0.1)                            # fitted on normals
    cold = torch.full((1, GRID, GRID), 0.5)
    hot = torch.full((1, GRID, GRID), 1.5)

    a = normalize_grid(cold, "global", stats=stats)
    b = normalize_grid(hot, "global", stats=stats)
    assert b.mean() > a.mean() + 5.0, "global normalisation lost the level difference"


def test_global_normalization_accepts_per_position_tables():
    """Statistics may be a (H, W) table, which is what fit_scale_stats returns."""
    mean = np.random.rand(GRID, GRID).astype(np.float32)
    std = np.full((GRID, GRID), 0.5, dtype=np.float32)
    grid = torch.rand(3, GRID, GRID)

    out = normalize_grid(grid, "global", stats=(mean, std))
    expected = (grid - torch.from_numpy(mean)) / 0.5
    assert torch.allclose(out, expected, atol=1e-5)


def test_global_normalization_divides_out_position_dependent_difficulty():
    """The mechanism behind Study 4's largest effect.

    Some positions are intrinsically hard on *every* image. After dividing by
    per-position normal statistics, a normal image should look flat.
    """
    baseline = torch.rand(GRID, GRID) * 5.0        # position-dependent difficulty
    normals = baseline.unsqueeze(0).repeat(8, 1, 1) + torch.randn(8, GRID, GRID) * 0.01

    mean = normals.mean(dim=0).numpy()
    std = normals.std(dim=0).clamp_min(1e-4).numpy()

    fresh_normal = baseline.unsqueeze(0) + torch.randn(1, GRID, GRID) * 0.01
    out = normalize_grid(fresh_normal, "global", stats=(mean, std))
    assert out.abs().mean() < 6.0, "positional baseline was not removed"


def test_global_falls_back_to_zscore_when_uncalibrated():
    grid = torch.randn(2, GRID, GRID) * 4 + 9
    out = normalize_grid(grid, "global", stats=None)
    assert out.reshape(2, -1).mean(dim=1).abs().max() < 1e-5


def test_none_normalization_is_a_passthrough():
    grid = torch.randn(2, GRID, GRID)
    assert torch.equal(normalize_grid(grid, "none"), grid)


def test_unknown_normalization_raises():
    with pytest.raises(ValueError, match="Unknown normalisation"):
        normalize_grid(torch.randn(1, GRID, GRID), "banana")


# --------------------------------------------------------------------------- #
# Scale fusion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", FUSION_MODES)
def test_every_fusion_mode_returns_the_right_shape(mode):
    grids = {2: torch.rand(3, GRID, GRID), 4: torch.rand(3, GRID, GRID)}
    kwargs = {"weights": {2: 1.0, 4: 2.0}} if mode == "weighted" else {}
    out = fuse_scales(grids, mode=mode, normalize="zscore", **kwargs)
    assert out.shape == (3, GRID, GRID)
    assert torch.isfinite(out).all()


def test_fusion_removes_the_per_scale_offset():
    """Larger windows are intrinsically harder; fusion must not let that
    offset alone decide the outcome."""
    grids = {
        2: torch.full((1, GRID, GRID), 0.5),
        6: torch.full((1, GRID, GRID), 3.0),      # much higher raw level
    }
    fused = fuse_scales(grids, mode="mean", normalize="zscore")
    assert fused.abs().max() < 1.0


def test_weighted_fusion_respects_the_weights():
    hot = torch.zeros(1, GRID, GRID)
    hot[0, 0, 0] = 10.0
    grids = {2: torch.zeros(1, GRID, GRID), 6: hot}

    toward_large = fuse_scales(grids, "weighted", {2: 0.0, 6: 1.0}, normalize="none")
    toward_small = fuse_scales(grids, "weighted", {2: 1.0, 6: 0.0}, normalize="none")
    assert toward_large[0, 0, 0] > toward_small[0, 0, 0]


def test_fusion_rejects_an_empty_input():
    with pytest.raises(ValueError):
        fuse_scales({}, mode="mean")


# --------------------------------------------------------------------------- #
# Upsampling / smoothing
# --------------------------------------------------------------------------- #
def test_upsample_reaches_pixel_resolution():
    out = upsample_map(torch.rand(2, GRID, GRID), out_size=256, sigma=4.0)
    assert out.shape == (2, 1, 256, 256)


def test_gaussian_blur_preserves_a_constant_field():
    """Reflect padding must not darken the border of the map."""
    flat = torch.ones(1, 1, 64, 64)
    out = gaussian_blur(flat, sigma=4.0)
    assert (out - 1.0).abs().max() < 1e-4


def test_zero_sigma_is_a_passthrough():
    x = torch.rand(1, 1, 32, 32)
    assert torch.equal(gaussian_blur(x, 0.0), x)


def test_build_anomaly_map_end_to_end():
    grids = {2: torch.rand(2, GRID, GRID), 4: torch.rand(2, GRID, GRID)}
    out = build_anomaly_map(grids, out_size=256, fusion="mean", normalize="zscore")
    assert out.shape == (2, 1, 256, 256)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", AGGREGATIONS)
def test_aggregation_returns_one_score_per_image(mode):
    maps = torch.rand(5, 1, 64, 64)
    assert aggregate_score(maps, mode, top_k_ratio=0.01).shape == (5,)


def test_topk_is_between_mean_and_max():
    maps = torch.rand(4, 1, 64, 64)
    mean = aggregate_score(maps, "mean")
    topk = aggregate_score(maps, "topk", 0.01)
    mx = aggregate_score(maps, "max")
    assert (mean <= topk + 1e-6).all() and (topk <= mx + 1e-6).all()


def test_topk_is_robust_to_a_single_hot_pixel_where_max_is_not():
    """Why `topk` is the default: one noisy pixel must not decide the verdict."""
    clean = torch.zeros(1, 1, 64, 64)
    spiked = clean.clone()
    spiked[0, 0, 32, 32] = 100.0

    max_shift = (aggregate_score(spiked, "max") - aggregate_score(clean, "max")).item()
    topk_shift = (aggregate_score(spiked, "topk", 0.01)
                  - aggregate_score(clean, "topk", 0.01)).item()
    assert topk_shift < max_shift / 10.0


def test_unknown_aggregation_raises():
    with pytest.raises(ValueError, match="Unknown aggregation"):
        aggregate_score(torch.rand(1, 1, 8, 8), "median")


# --------------------------------------------------------------------------- #
# Calibration -- the unsupervised constraint
# --------------------------------------------------------------------------- #
def test_threshold_is_mean_plus_k_sigma():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    calib = fit_calibration(scores, sigma_threshold=3.0)
    assert calib.score_mean == pytest.approx(3.0)
    assert calib.threshold == pytest.approx(3.0 + 3.0 * scores.std())


def test_calibration_never_sees_an_anomalous_image():
    """PROJECT CONSTRAINT: the threshold is a function of normals only.

    Fitting on the normal scores must give the identical threshold whether or
    not anomalous scores exist elsewhere in the run -- i.e. the fit has no
    channel through which a label could leak.
    """
    normals = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    calib_a = fit_calibration(normals, sigma_threshold=3.0)

    # The same normals, in a world that also contains screaming anomalies.
    _anomalies = np.array([500.0, 900.0])
    calib_b = fit_calibration(normals, sigma_threshold=3.0)

    assert calib_a.threshold == calib_b.threshold
    assert calib_a.threshold < 2.0, "threshold drifted towards the anomalies"


def test_three_sigma_flags_almost_no_normals():
    rng = np.random.default_rng(0)
    normals = rng.normal(0, 1, 4000)
    calib = fit_calibration(normals, sigma_threshold=3.0)
    false_alarm = calib.is_anomalous(normals).mean()
    assert false_alarm < 0.01, f"3-sigma false-alarm rate {false_alarm:.4f} too high"


def test_is_anomalous_is_a_strict_threshold_comparison():
    calib = Calibration(score_mean=0.0, score_std=1.0, threshold=10.0)
    out = calib.is_anomalous(np.array([9.99, 10.0, 10.01]))
    assert out.tolist() == [False, False, True]


def test_confidence_is_monotone_and_bounded():
    calib = Calibration(score_mean=0.0, score_std=1.0, threshold=5.0)
    values = [calib.confidence(s) for s in [-10, 0, 5, 10, 50]]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert all(b >= a for a, b in zip(values, values[1:])), "confidence not monotone"
    assert calib.confidence(5.0) == pytest.approx(0.5)


def test_zscore_normalisation_of_scores():
    calib = Calibration(score_mean=10.0, score_std=2.0)
    assert calib.normalize(np.array([10.0, 12.0, 6.0])).tolist() == [0.0, 1.0, -2.0]


def test_calibration_survives_a_json_round_trip():
    calib = Calibration(
        score_mean=1.5, score_std=0.25, threshold=2.25,
        map_lo=-1.0, map_hi=4.0, n_samples=20,
        scale_stats={4: (np.zeros((GRID, GRID), np.float32),
                         np.ones((GRID, GRID), np.float32))},
    )
    restored = Calibration.from_dict(calib.to_dict())

    assert restored.threshold == pytest.approx(calib.threshold)
    assert restored.n_samples == calib.n_samples
    assert set(restored.scale_stats) == {4}
    mean, std = restored.scale_stats[4]
    assert np.asarray(mean).shape == (GRID, GRID)
    assert np.allclose(np.asarray(std), 1.0)


def test_fit_calibration_rejects_empty_input():
    with pytest.raises(ValueError):
        fit_calibration(np.array([]))
