"""Tests for the memory-bank scoring channel (Study 11).

What is pinned here, and why:

* ``memory: off`` -- the default -- leaves every score bit-for-bit unchanged,
  so no previously reported number moves;
* the bank holds exactly the images the positional statistics are fitted on
  (in a few-shot run, the K training images) and never a validation image --
  otherwise the few-shot budget would quietly grow;
* a bank image is never compared with itself while the statistics are fitted.
  Its distance would be exactly zero, the fitted spread far too small, and
  every new normal image would look anomalous;
* a bank saved beside the checkpoint reproduces the evaluated scores exactly,
  so the demo and the reported numbers cannot drift apart.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly.scoring import AnomalyScorer, Calibration, build_scorer  # noqa: E402
from masking.sweep_mask import SweepMaskBank  # noqa: E402
from models import build_model  # noqa: E402
from utils.config import Config  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID = 16


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(Config({
        "data": {"img_size": 256, "patch_size": 16},
        "encoder": {"embed_dim": 64, "depth": 2, "num_heads": 4},
        "predictor": {"predictor_dim": 64, "depth": 1, "num_heads": 4},
    })).to(DEVICE).eval()


@pytest.fixture(scope="module")
def bank():
    return SweepMaskBank(GRID, windows=(4,), device=DEVICE)


def _loader(name: str, n: int, seed: int, batch: int = 3):
    """A list of batches shaped like the MVTecLOCO loader's."""
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, 256, 256, generator=g)
    paths = [f"/{name}/{i:03d}.png" for i in range(n)]
    return [{"image": images[i:i + batch], "path": paths[i:i + batch],
             "label": torch.zeros(len(paths[i:i + batch]), dtype=torch.long),
             "defect_type": ["good"] * len(paths[i:i + batch])}
            for i in range(0, n, batch)]


def _scorer(model, bank, **kw):
    return AnomalyScorer(model, bank, **kw)


def _scores(scorer, loader):
    return scorer.score_loader(loader, DEVICE, collect_maps=False)["scores"]


@pytest.fixture(scope="module")
def data():
    return {"train": _loader("train", 6, 0), "val": _loader("val", 4, 1),
            "test": _loader("test", 4, 2)}


# --------------------------------------------------------------------------- #
# Default and config
# --------------------------------------------------------------------------- #
def test_off_changes_no_score(model, bank, data):
    plain = _scorer(model, bank)
    plain.calibrate(data["val"], DEVICE, stats_loader=data["train"])
    off = _scorer(model, bank, memory="off")
    off.calibrate(data["val"], DEVICE, stats_loader=data["train"])
    # Tolerance, not bit-equality: two GPU passes may reduce in different order.
    assert torch.allclose(torch.as_tensor(_scores(plain, data["test"])),
                          torch.as_tensor(_scores(off, data["test"])), rtol=0, atol=1e-5)
    assert off.memory_bank is None and not off.calibration.mem_stats


@pytest.mark.parametrize("value, expected", [(False, "off"), (None, "off"),
                                              ("add", "add"), ("only", "only")])
def test_yaml_off_is_read_as_off(model, bank, value, expected):
    """YAML turns a bare `off` into False; that must not mean anything else."""
    cfg = Config({"data": {"img_size": 256}, "anomaly": {"memory": value}})
    assert build_scorer(cfg, model, bank).memory == expected


def test_unknown_or_conflicting_modes_are_refused(model, bank):
    with pytest.raises(ValueError, match="memory mode"):
        _scorer(model, bank, memory="sometimes")
    with pytest.raises(ValueError, match="cardinality=off"):
        _scorer(model, bank, memory="only", cardinality="add")


# --------------------------------------------------------------------------- #
# The bank holds the budget, and nothing else
# --------------------------------------------------------------------------- #
def test_bank_holds_exactly_the_statistics_images(model, bank, data):
    scorer = _scorer(model, bank, memory="add")
    scorer.calibrate(data["val"], DEVICE, stats_loader=data["train"])
    train_paths = [p for b in data["train"] for p in b["path"]]
    assert scorer.memory_paths == train_paths
    assert scorer.memory_bank.size(0) == len(train_paths) * GRID * GRID
    assert not any(p.startswith("/val/") for p in scorer.memory_paths)


def test_a_bank_image_is_never_compared_with_itself(model, bank, data):
    scorer = _scorer(model, bank, memory="add")
    scorer.fit_memory_bank(data["train"], DEVICE)
    batch = data["train"][0]
    images = batch["image"].to(DEVICE)

    against_all = scorer._memory_grid(images)                     # no path: self included
    against_others = scorer._memory_grid(images, batch["path"])   # leave-one-out
    assert against_all.abs().max() < 1e-2                         # its own patches match
    assert (against_others > 0.05).all()


def test_the_fitted_spread_is_that_of_a_new_image(model, bank, data):
    """Leave-one-out keeps the statistics of a bank image honest."""
    scorer = _scorer(model, bank, memory="only")
    scorer.calibrate(data["val"], DEVICE, stats_loader=data["train"])
    mean, std = scorer.calibration.mem_stats[0]
    assert (mean > 0.05).all() and (std > 0).all()


def test_a_copy_of_a_bank_image_is_the_least_surprising(model, bank, data):
    scorer = _scorer(model, bank, memory="only")
    scorer.calibrate(data["val"], DEVICE, stats_loader=data["train"])
    copy = dict(data["train"][0], path=["/elsewhere/copy.png"] * 3)
    fresh = data["test"][0]
    assert _scores(scorer, [copy]).max() < _scores(scorer, [fresh]).min()


def test_only_mode_never_runs_the_sweep(model, bank, data, monkeypatch):
    scorer = _scorer(model, bank, memory="only")

    def _no_sweep(*args, **kwargs):
        raise AssertionError("memory=only must not run the masked sweep")

    monkeypatch.setattr(scorer, "_grids", _no_sweep)
    scorer.calibrate(data["val"], DEVICE, stats_loader=data["train"])
    assert len(_scores(scorer, data["test"])) == 4


def test_add_mode_combines_both_channels(model, bank, data):
    jepa = _scorer(model, bank)
    jepa.calibrate(data["val"], DEVICE, stats_loader=data["train"])
    both = _scorer(model, bank, memory="add")
    both.calibrate(data["val"], DEVICE, stats_loader=data["train"])
    assert not torch.allclose(torch.as_tensor(_scores(jepa, data["test"])),
                              torch.as_tensor(_scores(both, data["test"])))


def test_scoring_without_a_bank_fails_loudly(model, bank, data):
    scorer = _scorer(model, bank, memory="add")
    with pytest.raises(RuntimeError, match="no memory bank"):
        scorer.score_batch(data["test"][0]["image"].to(DEVICE))


# --------------------------------------------------------------------------- #
# What the demo loads reproduces what evaluation measured
# --------------------------------------------------------------------------- #
def test_a_saved_bank_reproduces_the_scores_exactly(model, bank, data, tmp_path):
    fitted = _scorer(model, bank, memory="add")
    fitted.calibrate(data["val"], DEVICE, stats_loader=data["train"])
    expected = _scores(fitted, data["test"])

    (tmp_path / "calibration.json").write_text(json.dumps(fitted.calibration.to_dict()))
    torch.save(fitted.memory_state(), tmp_path / "memory_bank.pt")

    loaded = _scorer(model, bank, memory="add")
    loaded.calibration = Calibration.from_dict(
        json.loads((tmp_path / "calibration.json").read_text()))
    loaded.load_memory_bank(torch.load(tmp_path / "memory_bank.pt", weights_only=True), DEVICE)

    assert loaded.calibration.mem_stats.keys() == {0}
    assert torch.allclose(torch.as_tensor(expected),
                          torch.as_tensor(_scores(loaded, data["test"])), atol=1e-5)
