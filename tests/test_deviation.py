"""Tests for signed vs absolute deviation scoring (Study 7).

This mode exists because of a measured failure rather than a hunch. On real
MVTec LOCO ``pushpins``, the signed score put logical anomalies at
**0.449 ± 0.003 image AUROC over three seeds** -- reliably *below* chance,
meaning anomalous images ranked as more normal than normal ones.

The mechanism: a missing pushpin leaves an empty compartment, and an empty
compartment is *easier* to predict than the object that belongs there. Signed
scoring only counts "harder than normal", so it reads that as extra-normal and
pushes the image down the ranking.

:func:`test_absolute_recovers_a_missing_object_signal` reproduces exactly that
situation on synthetic grids, so the fix is pinned to the mechanism it was
built for rather than to one dataset's numbers.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly.anomaly_map import build_anomaly_map, fuse_scales  # noqa: E402
from anomaly.scoring import aggregate_score, build_scorer  # noqa: E402
from utils.config import Config  # noqa: E402

GRID = 16


def _stats(mean=0.5, std=0.1):
    import numpy as np
    return {4: (np.full((GRID, GRID), mean, np.float32),
                np.full((GRID, GRID), std, np.float32))}


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #
def test_signed_keeps_the_sign():
    """A region easier than normal stays negative under `signed`."""
    grid = torch.full((1, GRID, GRID), 0.5)
    grid[0, 4, 4] = 0.1                      # much easier than the 0.5 baseline

    out = fuse_scales({4: grid}, normalize="global",
                      scale_stats=_stats(), deviation="signed")
    assert out[0, 4, 4] < 0, "easier-than-normal should read as negative"


def test_absolute_makes_an_easy_region_positive():
    grid = torch.full((1, GRID, GRID), 0.5)
    grid[0, 4, 4] = 0.1

    out = fuse_scales({4: grid}, normalize="global",
                      scale_stats=_stats(), deviation="absolute")
    assert out[0, 4, 4] > 0
    assert out[0, 4, 4] == pytest.approx(4.0, abs=1e-4)   # |(0.1-0.5)/0.1|


def test_absolute_leaves_a_hard_region_unchanged():
    """The classic case must not regress."""
    grid = torch.full((1, GRID, GRID), 0.5)
    grid[0, 8, 8] = 0.9                      # harder than normal

    signed = fuse_scales({4: grid}, normalize="global",
                         scale_stats=_stats(), deviation="signed")
    absolute = fuse_scales({4: grid}, normalize="global",
                           scale_stats=_stats(), deviation="absolute")
    assert signed[0, 8, 8] == pytest.approx(absolute[0, 8, 8], abs=1e-5)


def test_absolute_is_applied_per_scale_before_fusion():
    """One scale being easy must not cancel another being hard.

    Averaging -4 and +4 gives 0; averaging |−4| and |+4| gives 4. Taking the
    absolute value after fusion would silently lose the first case.
    """
    easy = torch.full((1, GRID, GRID), 0.5)
    easy[0, 2, 2] = 0.1                      # z = -4
    hard = torch.full((1, GRID, GRID), 0.5)
    hard[0, 2, 2] = 0.9                      # z = +4

    import numpy as np
    stats = {2: (np.full((GRID, GRID), 0.5, np.float32),
                 np.full((GRID, GRID), 0.1, np.float32)),
             4: (np.full((GRID, GRID), 0.5, np.float32),
                 np.full((GRID, GRID), 0.1, np.float32))}

    fused = fuse_scales({2: easy, 4: hard}, mode="mean", normalize="global",
                        scale_stats=stats, deviation="absolute")
    assert fused[0, 2, 2] == pytest.approx(4.0, abs=1e-4)


def test_unknown_deviation_is_rejected():
    with pytest.raises(ValueError, match="Unknown deviation mode"):
        fuse_scales({4: torch.rand(1, GRID, GRID)}, deviation="squared")


def test_default_is_signed():
    """Every existing measured number was produced under `signed`."""
    grid = torch.full((1, GRID, GRID), 0.5)
    grid[0, 4, 4] = 0.1

    default = fuse_scales({4: grid}, normalize="global", scale_stats=_stats())
    signed = fuse_scales({4: grid}, normalize="global",
                         scale_stats=_stats(), deviation="signed")
    assert torch.allclose(default, signed)


# --------------------------------------------------------------------------- #
# The failure this was built for
# --------------------------------------------------------------------------- #
def test_absolute_recovers_a_missing_object_signal():
    """Reproduces the real-LOCO pushpins failure and its fix.

    Normal images: every position at the baseline difficulty.
    Anomalous images: one region is *easier* than baseline, standing in for a
    missing object whose empty compartment predicts cleanly.

    Under `signed`, the anomalous images must rank BELOW the normal ones
    (AUROC < 0.5). Under `absolute`, they must rank above.
    """
    from anomaly.metrics import image_auroc
    import numpy as np

    torch.manual_seed(0)
    n = 24
    normal = torch.full((n, GRID, GRID), 0.5) + torch.randn(n, GRID, GRID) * 0.01
    anomalous = torch.full((n, GRID, GRID), 0.5) + torch.randn(n, GRID, GRID) * 0.01
    anomalous[:, 6:9, 6:9] = 0.20            # missing object -> easy to predict

    grids = {4: torch.cat([normal, anomalous], dim=0)}
    labels = np.array([0] * n + [1] * n)

    def score(deviation: str) -> float:
        maps = build_anomaly_map(grids, out_size=64, normalize="global",
                                 scale_stats=_stats(), deviation=deviation,
                                 sigma=0.0)
        return image_auroc(aggregate_score(maps, "topk", 0.05).numpy(), labels)

    signed_auroc = score("signed")
    absolute_auroc = score("absolute")

    # `signed` lands just *below* chance rather than near zero, and that is the
    # faithful reproduction: `topk` reads the high end of the map, so a strongly
    # negative region barely moves the score at all -- it simply fails to
    # contribute. The real measurement behaved the same way (0.449).
    assert signed_auroc < 0.5, (
        f"signed should rank missing-object anomalies at or below chance, "
        f"got {signed_auroc:.3f}"
    )
    assert absolute_auroc > 0.9, (
        f"absolute should recover the signal, got {absolute_auroc:.3f}"
    )
    assert absolute_auroc - signed_auroc > 0.4, (
        "absolute must be a large improvement, not a marginal one"
    )


def test_absolute_does_not_break_the_extra_object_case():
    """The opposite anomaly -- an *added* object -- must still be detected."""
    from anomaly.metrics import image_auroc
    import numpy as np

    torch.manual_seed(0)
    n = 24
    normal = torch.full((n, GRID, GRID), 0.5) + torch.randn(n, GRID, GRID) * 0.01
    anomalous = torch.full((n, GRID, GRID), 0.5) + torch.randn(n, GRID, GRID) * 0.01
    anomalous[:, 6:9, 6:9] = 0.85            # extra object -> hard to predict

    grids = {4: torch.cat([normal, anomalous], dim=0)}
    labels = np.array([0] * n + [1] * n)

    for deviation in ("signed", "absolute"):
        maps = build_anomaly_map(grids, out_size=64, normalize="global",
                                 scale_stats=_stats(), deviation=deviation,
                                 sigma=0.0)
        auroc = image_auroc(aggregate_score(maps, "topk", 0.05).numpy(), labels)
        assert auroc > 0.9, f"{deviation} lost the extra-object signal ({auroc:.3f})"


# --------------------------------------------------------------------------- #
# Config wiring
# --------------------------------------------------------------------------- #
def test_builder_reads_the_deviation_key():
    from masking.sweep_mask import SweepMaskBank
    from models import build_model

    model = build_model(Config({
        "data": {"img_size": 256, "patch_size": 16},
        "encoder": {"embed_dim": 64, "depth": 2, "num_heads": 4},
        "predictor": {"predictor_dim": 64, "depth": 2, "num_heads": 4},
    }))
    bank = SweepMaskBank(model.grid_size, windows=(4,))

    cfg = Config({"data": {"img_size": 256},
                  "anomaly": {"deviation": "absolute"}})
    assert build_scorer(cfg, model, bank).deviation == "absolute"

    cfg_default = Config({"data": {"img_size": 256}, "anomaly": {}})
    assert build_scorer(cfg_default, model, bank).deviation == "signed"
