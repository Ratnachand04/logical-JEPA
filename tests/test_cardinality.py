"""Tests for the cardinality head and its scoring channel (Phase 3c).

What is pinned here:

* the target, :func:`region_foreground_mass`, moves the right way -- 0 for a
  region that is all background, 1 for a region that is all component;
* the head's features are permutation-invariant over slots (Phase 3a: slot
  identity is not stable, so nothing may depend on slot order);
* **stop-gradient**: the cardinality loss trains the head and nothing else;
* scoring is reproducible and independent of an image's batch position;
* ``cardinality: off`` leaves every existing number bit-for-bit unchanged;
* the subtype breakdown that tests the removal mechanism is computed against
  the same normals as everything else.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly.metrics import load_subtype_manifest, subtype_breakdown  # noqa: E402
from anomaly.scoring import (  # noqa: E402
    AnomalyScorer,
    Calibration,
    _cardinality_mode,
    build_scorer,
)
from masking.base import complement_indices  # noqa: E402
from masking.sweep_mask import SweepMaskBank  # noqa: E402
from models import build_model  # noqa: E402
from models.cardinality import CardinalityHead  # noqa: E402
from models.slot_bottleneck import region_foreground_mass, sorted_usage  # noqa: E402
from utils.config import Config  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID = 16


def _cfg(slots: bool = True) -> Config:
    return Config({
        "data": {"img_size": 256, "patch_size": 16},
        "encoder": {"embed_dim": 64, "depth": 2, "num_heads": 4},
        "predictor": {"predictor_dim": 64, "depth": 2, "num_heads": 4},
        "loss": {"kind": "cosine"},
        "slots": {"enabled": slots, "num_slots": 4, "iters": 2,
                  "recon_weight": 0.0, "hierarchical_weight": 0.0},
    })


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(_cfg()).to(DEVICE).eval()


@pytest.fixture(scope="module")
def bank():
    return SweepMaskBank(GRID, windows=(4,), device=DEVICE)


# --------------------------------------------------------------------------- #
# The target and the features
# --------------------------------------------------------------------------- #
def test_foreground_mass_is_zero_on_pure_background():
    attn = torch.zeros(1, 3, 8)
    attn[:, 0] = 1.0                                   # slot 0 owns everything
    region = torch.tensor([[0, 1, 2]])
    mass = region_foreground_mass(attn, region, torch.tensor([0]))
    assert mass.item() == pytest.approx(0.0)


def test_foreground_mass_is_one_on_pure_component():
    attn = torch.zeros(1, 3, 8)
    attn[:, 0, 4:] = 1.0                               # background
    attn[:, 1, :4] = 1.0                               # a component
    mass = region_foreground_mass(attn, torch.tensor([[0, 1]]), torch.tensor([0]))
    assert mass.item() == pytest.approx(1.0)


def test_foreground_mass_falls_when_a_component_goes_missing():
    """The mechanism the head exists for: removal must lower the target."""
    full = torch.zeros(1, 2, 4)
    full[:, 0] = torch.tensor([0.5, 0.5, 1.0, 1.0])    # background
    full[:, 1] = torch.tensor([0.5, 0.5, 0.0, 0.0])    # a screw over patches 0-1
    emptied = full.clone()
    emptied[:, 0, :2], emptied[:, 1, :2] = 1.0, 0.0     # the screw removed
    region, bg = torch.tensor([[0, 1]]), torch.tensor([0])
    assert region_foreground_mass(emptied, region, bg) < region_foreground_mass(full, region, bg)


def test_sorted_usage_ignores_slot_order():
    attn = torch.rand(2, 5, 30)
    perm = torch.randperm(5)
    assert torch.allclose(sorted_usage(attn), sorted_usage(attn[:, perm]))
    assert torch.allclose(sorted_usage(attn).sum(-1), torch.ones(2))


def test_head_outputs_one_bounded_value_per_region():
    head = CardinalityHead(num_slots=4, grid_size=GRID)
    out = head(torch.softmax(torch.randn(3, 4), -1), torch.randint(0, GRID**2, (3, 9)))
    assert out.shape == (3,)
    assert ((out >= 0) & (out <= 1)).all()


# --------------------------------------------------------------------------- #
# Stop-gradient
# --------------------------------------------------------------------------- #
def test_cardinality_loss_trains_only_the_head():
    model = build_model(_cfg()).to(DEVICE)
    B, D = 2, model.embed_dim
    block = torch.tensor([0, 1, GRID, GRID + 1], device=DEVICE)
    ctx = complement_indices(GRID, [block.cpu()]).to(DEVICE).unsqueeze(0).expand(B, -1)
    preds = torch.randn(B, 4, D, device=DEVICE, requires_grad=True)

    loss, stats = model._slot_losses(torch.randn(B, GRID**2, D, device=DEVICE), ctx,
                                     [block.unsqueeze(0).expand(B, -1)], [preds])
    loss.backward()

    assert "loss_card" in stats
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.card_head.parameters())
    # recon_weight=0 and lambda=0: the only live term is cardinality, so any
    # gradient on the slots or the predictions would be a stop-gradient leak.
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.slots.parameters())
    assert preds.grad is None


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def test_cardinality_grids_have_one_value_per_patch(model, bank):
    grids, card = model.anomaly_grids(torch.randn(2, 3, 256, 256, device=DEVICE),
                                      bank, chunk=16, with_cardinality=True)
    assert set(card) == set(grids) == {4}
    assert card[4].shape == (2, GRID, GRID)
    assert (card[4] >= 0).all() and torch.isfinite(card[4]).all()


def test_cardinality_is_reproducible_and_batch_position_independent(model, bank):
    """Fixed slot initialisation: an image's score must not depend on its batchmates."""
    images = torch.randn(3, 3, 256, 256, device=DEVICE)
    _, together = model.anomaly_grids(images, bank, chunk=16, with_cardinality=True)
    _, alone = model.anomaly_grids(images[2:3], bank, chunk=16, with_cardinality=True)
    _, again = model.anomaly_grids(images, bank, chunk=16, with_cardinality=True)
    assert torch.allclose(together[4][2:3], alone[4], atol=1e-4)
    assert torch.allclose(together[4], again[4])


def test_requesting_cardinality_without_a_head_fails_loudly(bank):
    plain = build_model(_cfg(slots=False)).to(DEVICE)
    with pytest.raises(ValueError, match="cardinality"):
        plain.anomaly_grids(torch.randn(1, 3, 256, 256, device=DEVICE), bank,
                            with_cardinality=True)
    with pytest.raises(ValueError, match="cardinality"):
        AnomalyScorer(plain, bank, cardinality="add")


def test_cardinality_off_changes_nothing(model, bank):
    """Every number reported before Phase 3c must be reproducible unchanged."""
    images = torch.randn(2, 3, 256, 256, device=DEVICE)
    default = AnomalyScorer(model, bank).score_batch(images)["scores"]
    off = AnomalyScorer(model, bank, cardinality="off").score_batch(images)["scores"]
    assert torch.equal(default, off)


def test_add_with_zero_weight_equals_off(model, bank):
    images = torch.randn(2, 3, 256, 256, device=DEVICE)
    off = AnomalyScorer(model, bank, cardinality="off").score_batch(images)["scores"]
    zero = AnomalyScorer(model, bank, cardinality="add",
                         cardinality_weight=0.0).score_batch(images)["scores"]
    assert torch.allclose(off, zero, atol=1e-5)


def test_only_mode_ignores_the_jepa_channel(model, bank):
    """`only` isolates the channel: the JEPA distance must not move its score."""
    images = torch.randn(2, 3, 256, 256, device=DEVICE)
    a = AnomalyScorer(model, bank, cardinality="only", distance="cosine")
    b = AnomalyScorer(model, bank, cardinality="only", distance="l2")
    assert torch.allclose(a.score_batch(images)["scores"], b.score_batch(images)["scores"])


def test_calibration_fits_and_round_trips_card_stats(model, bank):
    images = torch.randn(4, 3, 256, 256, device=DEVICE)
    loader = [{"image": images[:2], "label": torch.zeros(2, dtype=torch.long),
               "defect_type": ["good"] * 2},
              {"image": images[2:], "label": torch.zeros(2, dtype=torch.long),
               "defect_type": ["good"] * 2}]
    scorer = AnomalyScorer(model, bank, cardinality="add")
    calib = scorer.calibrate(loader, DEVICE)

    assert set(calib.card_stats) == {4}
    restored = Calibration.from_dict(json.loads(json.dumps(calib.to_dict())))
    assert np.allclose(restored.card_stats[4][0], calib.card_stats[4][0])


def test_yaml_off_is_read_as_off():
    """A bare `off` in YAML is boolean False -- it must not become a crash."""
    assert _cardinality_mode(False) == "off"
    assert _cardinality_mode(None) == "off"
    assert _cardinality_mode("only") == "only"


def test_build_scorer_reads_the_config(model, bank):
    cfg = _cfg()
    cfg.set_path("anomaly.cardinality", "add")
    cfg.set_path("anomaly.cardinality_weight", 0.5)
    scorer = build_scorer(cfg, model, bank)
    assert scorer.cardinality == "add" and scorer.cardinality_weight == 0.5


def test_checkpoint_architecture_survives_an_inherited_override(tmp_path):
    """Study 9 crashed on this: the ablation config inherits slots.enabled=false
    and was merged over a slot-trained checkpoint. Inference overrides must
    still apply, but the architecture must come from the checkpoint."""
    from evaluate import load_model_from_checkpoint

    trained = build_model(_cfg())
    path = tmp_path / "ckpt.pt"
    torch.save({"model": trained.state_dict(), "config": _cfg().to_dict()}, path)

    override = Config({"slots": {"enabled": False},
                       "anomaly": {"cardinality": "add"}})
    model, cfg, _ = load_model_from_checkpoint(str(path), "cpu", override)

    assert model.has_cardinality
    assert cfg.get_path("anomaly.cardinality") == "add"


# --------------------------------------------------------------------------- #
# Subtype breakdown
# --------------------------------------------------------------------------- #
def test_subtype_breakdown_scores_each_direction_against_all_normals():
    paths = ["x/test/good/000.png", "x/test/good/001.png",
             "x/test/logical_anomalies/000.png", "x/test/logical_anomalies/001.png"]
    manifest = {
        "logical_anomalies/000.png": {"subtype": "missing_screw", "direction": "removal"},
        "logical_anomalies/001.png": {"subtype": "extra_screw", "direction": "addition"},
    }
    labels = np.array([0, 0, 1, 1])
    # removal scored BELOW both normals, addition above: the Phase 1 pattern.
    scores = np.array([0.5, 0.6, 0.1, 0.9])
    out = subtype_breakdown(scores, labels, paths, manifest)

    assert out["removal_auroc"] == pytest.approx(0.0)
    assert out["addition_auroc"] == pytest.approx(1.0)
    assert out["subtype_missing_screw_auroc"] == pytest.approx(0.0)
    assert "structural_auroc" not in out           # never shadows the family metric


def test_synthetic_generator_writes_a_consistent_manifest(tmp_path):
    from datasets.synthetic_loco import DEFECT_DIRECTION, generate_dataset

    base = generate_dataset(root=str(tmp_path), n_train=1, n_val=1, n_test_good=1,
                            n_test_logical=5, n_test_structural=3, verbose=False)
    manifest = load_subtype_manifest(base)

    assert manifest is not None and len(manifest) == 8
    for rel, entry in manifest.items():
        assert os.path.isfile(os.path.join(base, "test", rel))
        assert DEFECT_DIRECTION[entry["subtype"]] == entry["direction"]
