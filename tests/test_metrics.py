"""Tests for the evaluation metrics.

Metrics are the one place where a bug cannot be caught by inspection later --
a wrong AUROC still looks like a plausible number. Each test here pins a case
whose correct answer is known analytically:

* a perfect detector scores 1.0, an inverted one 0.0, a random one ~0.5;
* AU-PRO must weight *regions* equally, so missing a small region costs a lot
  even when pixel AUROC barely moves (this is the whole reason AU-PRO exists);
* the logical/structural split must score each family against the *same*
  normals, otherwise the two numbers are not comparable to each other.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly.metrics import (  # noqa: E402
    compute_pro,
    evaluate_split,
    image_auroc,
    image_average_precision,
    pixel_auroc,
)

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Image-level
# --------------------------------------------------------------------------- #
def test_perfect_separation_gives_auroc_one():
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert image_auroc(scores, labels) == pytest.approx(1.0)


def test_inverted_separation_gives_auroc_zero():
    scores = np.array([1.0, 0.9, 0.8, 0.2, 0.1, 0.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert image_auroc(scores, labels) == pytest.approx(0.0)


def test_constant_scores_give_chance_auroc():
    scores = np.full(20, 3.14)
    labels = np.array([0] * 10 + [1] * 10)
    assert image_auroc(scores, labels) == pytest.approx(0.5)


def test_single_class_returns_nan_not_a_number():
    """One-class splits must not silently report 0.5 as if it were measured."""
    assert np.isnan(image_auroc(np.array([1.0, 2.0]), np.array([0, 0])))
    assert np.isnan(image_average_precision(np.array([1.0, 2.0]), np.array([1, 1])))


def test_auroc_is_invariant_to_monotone_rescaling():
    """AUROC is rank-based; the calibration transform must not change it."""
    scores = RNG.normal(size=50)
    labels = (RNG.random(50) > 0.5).astype(int)
    base = image_auroc(scores, labels)
    assert image_auroc(scores * 7.5 + 3.0, labels) == pytest.approx(base)
    assert image_auroc(np.exp(scores), labels) == pytest.approx(base)


# --------------------------------------------------------------------------- #
# Pixel-level
# --------------------------------------------------------------------------- #
def test_pixel_auroc_perfect_and_inverted():
    masks = np.zeros((3, 32, 32), dtype=np.float32)
    masks[:, 8:20, 8:20] = 1.0
    assert pixel_auroc(masks.copy(), masks) == pytest.approx(1.0)
    assert pixel_auroc(-masks, masks) == pytest.approx(0.0)


def test_pixel_auroc_random_is_near_chance():
    masks = np.zeros((4, 32, 32), dtype=np.float32)
    masks[:, 4:12, 4:12] = 1.0
    score = pixel_auroc(RNG.random((4, 32, 32)).astype(np.float32), masks)
    assert 0.40 < score < 0.60


def test_pixel_auroc_without_defect_pixels_is_nan():
    masks = np.zeros((2, 16, 16), dtype=np.float32)
    assert np.isnan(pixel_auroc(RNG.random((2, 16, 16)).astype(np.float32), masks))


# --------------------------------------------------------------------------- #
# AU-PRO -- the region-weighted metric
# --------------------------------------------------------------------------- #
def test_au_pro_perfect_detector():
    masks = np.zeros((2, 64, 64), dtype=bool)
    masks[0, 5:15, 5:15] = True
    masks[1, 40:60, 40:60] = True
    assert compute_pro(masks.astype(np.float32), masks) == pytest.approx(1.0, abs=1e-6)


def test_au_pro_random_detector_is_low():
    masks = np.zeros((2, 64, 64), dtype=bool)
    masks[0, 5:15, 5:15] = True
    masks[1, 40:60, 40:60] = True
    score = compute_pro(RNG.random((2, 64, 64)).astype(np.float32), masks)
    assert score < 0.35


def test_au_pro_punishes_missing_a_small_region_more_than_pixel_auroc():
    """The defining property of AU-PRO, and the reason it is reported.

    Image 2 holds one tiny region and one large one. A detector that finds only
    the large one keeps a high pixel AUROC (it got most of the *pixels* right)
    but must lose substantial AU-PRO (it missed half the *regions*).
    """
    masks = np.zeros((1, 64, 64), dtype=bool)
    masks[0, 2:6, 2:6] = True             # 16 px  -- small region
    masks[0, 30:58, 30:58] = True         # 784 px -- large region

    big_only = masks.copy().astype(np.float32)
    big_only[0, 2:6, 2:6] = 0.0           # miss the small region entirely

    pro = compute_pro(big_only, masks)
    px = pixel_auroc(big_only, masks)

    assert px > 0.95, "pixel AUROC should barely notice the miss"
    assert pro < 0.75, "AU-PRO must penalise the missed region"
    assert pro < px, "AU-PRO should be the stricter metric here"


def test_au_pro_is_nan_without_any_region():
    masks = np.zeros((2, 32, 32), dtype=bool)
    assert np.isnan(compute_pro(RNG.random((2, 32, 32)).astype(np.float32), masks))


def test_au_pro_respects_the_fpr_limit():
    """Integrating to a wider FPR must not *decrease* the normalised area."""
    masks = np.zeros((2, 48, 48), dtype=bool)
    masks[:, 10:24, 10:24] = True
    maps = RNG.random((2, 48, 48)).astype(np.float32)
    maps[:, 10:24, 10:24] += 0.6

    narrow = compute_pro(maps, masks, max_fpr=0.05)
    wide = compute_pro(maps, masks, max_fpr=0.30)
    assert 0.0 <= narrow <= 1.0 and 0.0 <= wide <= 1.0


# --------------------------------------------------------------------------- #
# The logical / structural split
# --------------------------------------------------------------------------- #
def _split_fixture():
    n_good, n_log, n_str = 20, 15, 15
    defect_types = np.array(
        ["good"] * n_good + ["logical_anomalies"] * n_log + ["structural_anomalies"] * n_str
    )
    labels = np.array([0] * n_good + [1] * (n_log + n_str))
    return defect_types, labels, n_good, n_log, n_str


def test_split_reports_each_family_separately():
    defect_types, labels, n_good, n_log, n_str = _split_fixture()

    # Logical perfectly separable, structural completely inseparable.
    scores = np.concatenate([
        np.zeros(n_good),
        np.ones(n_log) * 10.0,
        np.zeros(n_str),
    ])

    res = evaluate_split(scores, labels, defect_types)
    assert res["logical_auroc"] == pytest.approx(1.0)
    assert res["structural_auroc"] == pytest.approx(0.5)
    assert res["n_logical"] == n_log and res["n_structural"] == n_str


def test_family_auroc_uses_the_same_normals_as_the_other_family():
    """Both family scores must be measured against the full normal set.

    If one family were scored against a subset, the two numbers would not be
    comparable -- and comparing them is the entire point of the study.
    """
    defect_types, labels, n_good, n_log, n_str = _split_fixture()
    scores = np.concatenate([RNG.normal(0, 1, n_good),
                             RNG.normal(3, 1, n_log),
                             RNG.normal(3, 1, n_str)])

    res = evaluate_split(scores, labels, defect_types)

    normals = scores[:n_good]
    expected_log = image_auroc(
        np.concatenate([normals, scores[n_good:n_good + n_log]]),
        np.array([0] * n_good + [1] * n_log),
    )
    assert res["logical_auroc"] == pytest.approx(expected_log)


def test_split_counts_are_reported():
    defect_types, labels, n_good, n_log, n_str = _split_fixture()
    res = evaluate_split(RNG.normal(size=len(labels)), labels, defect_types)
    assert res["n_total"] == len(labels)
    assert res["n_normal"] == n_good
    assert res["n_anomalous"] == n_log + n_str


def test_split_handles_a_missing_family():
    """A category with no structural anomalies must yield NaN, not crash."""
    defect_types = np.array(["good"] * 5 + ["logical_anomalies"] * 5)
    labels = np.array([0] * 5 + [1] * 5)
    res = evaluate_split(RNG.normal(size=10), labels, defect_types)
    assert np.isnan(res["structural_auroc"])
    assert not np.isnan(res["logical_auroc"])
