"""Tests for the reference-conditioned model, its data, alignment and scoring.

What is pinned, and why:

* the predictor really uses the reference -- change the reference and the
  prediction changes; otherwise "compare with a reference" would be a label;
* training touches the predictor only; the frozen encoder never moves;
* a pair is always two normal images of the *same* product, and held-out
  products never reach training -- the generalisation experiment depends on it;
* alignment undoes a known shift, never reports the empty border as content,
  and refuses implausible warps;
* the scorer ranks a current image with a foreign object above an identical
  copy of the reference.

Everything runs on a tiny randomly initialised ViT, on CPU or GPU.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly.alignment import align_to_reference, plausible, warp_mask  # noqa: E402
from anomaly.reference_scoring import ReferenceScorer, load_calibration  # noqa: E402
from datasets.pairs import PairSource, ReferencePairs, resolve_sources  # noqa: E402
from masking import build_mask_generator  # noqa: E402
from masking.sweep_mask import SweepMaskBank  # noqa: E402
from models import pretrained as P  # noqa: E402
from models.reference_jepa import ReferenceJEPA, ReferencePredictor  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TINY = P.PretrainedSpec(name="tiny_ref", url="http://invalid.local/x.pth", embed_dim=32, depth=2,
                        num_heads=4, patch_size=16, source_grid=4)
SIZE = 64          # 4x4 patch grid


@pytest.fixture(autouse=True)
def _tiny_backbone(monkeypatch):
    monkeypatch.setitem(P.PRETRAINED, "tiny_ref", TINY)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ReferenceJEPA(backbone="tiny_ref", img_size=SIZE,
                         predictor_cfg={"predictor_dim": 32, "depth": 1, "num_heads": 4},
                         load_pretrained=False).to(DEVICE)


def _image(seed: int, size: int = SIZE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((size, size, 3), np.uint8)
    base[8:56, 8:56] = rng.integers(60, 200, (48, 48, 3))
    base[20:44, 20:44] = (220, 40, 40)
    return base


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_predictor_accepts_one_or_several_references():
    pred = ReferencePredictor(embed_dim=32, predictor_dim=32, depth=1, num_heads=4, grid_size=4)
    ctx = torch.randn(2, 10, 32)
    ctx_idx = torch.arange(10).expand(2, -1)
    tgt = torch.arange(10, 14).expand(2, -1)
    for refs in (1, 3):
        out = pred(ctx, ctx_idx, torch.randn(2, refs * 16, 32), tgt)
        assert out.shape == (2, 4, 32)


def test_partial_reference_token_sets_are_refused():
    pred = ReferencePredictor(embed_dim=32, predictor_dim=32, depth=1, num_heads=4, grid_size=4)
    with pytest.raises(ValueError, match="whole images"):
        pred(torch.randn(1, 10, 32), torch.arange(10)[None], torch.randn(1, 15, 32),
             torch.arange(10, 12)[None])


def test_the_prediction_depends_on_the_reference():
    """If it did not, the model would not be comparing anything."""
    torch.manual_seed(0)
    pred = ReferencePredictor(embed_dim=32, predictor_dim=32, depth=2, num_heads=4, grid_size=4)
    ctx, ctx_idx, tgt = torch.randn(1, 10, 32), torch.arange(10)[None], torch.arange(10, 14)[None]
    a = pred(ctx, ctx_idx, torch.randn(1, 16, 32), tgt)
    b = pred(ctx, ctx_idx, torch.randn(1, 16, 32), tgt)
    assert not torch.allclose(a, b)


def test_training_updates_the_predictor_only(model):
    before = [p.clone() for p in model.encoder.parameters()]
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    gen = build_mask_generator({"strategy": "small", "num_small_only_targets": 2,
                                "small_sizes": [[2, 2]]}, model.grid_size, seed=0)
    cur = torch.randn(2, 3, SIZE, SIZE, device=DEVICE)
    refs = torch.randn(2, 1, 3, SIZE, SIZE, device=DEVICE)
    loss, stats = model(cur, refs, gen())
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert all(torch.equal(a, b) for a, b in zip(before, model.encoder.parameters()))
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert all(not p.requires_grad for p in model.encoder.parameters())


def test_error_and_memory_grids(model):
    bank = SweepMaskBank(model.grid_size, windows=(2,), device=DEVICE)
    x = torch.randn(1, 3, SIZE, SIZE, device=DEVICE)
    tokens = model.encode_references(x.unsqueeze(1))
    grids = model.error_grids(x, tokens, bank)
    assert grids[2].shape == (1, 4, 4) and (grids[2] >= 0).all()
    # A patch's nearest reference patch is itself when current == reference.
    assert model.memory_grid(x, tokens).abs().max() < 1e-3
    other = model.encode_references(torch.randn(1, 1, 3, SIZE, SIZE, device=DEVICE))
    assert model.memory_grid(x, other).min() > 1e-3


# --------------------------------------------------------------------------- #
# Pairs
# --------------------------------------------------------------------------- #
@pytest.fixture
def products(tmp_path):
    for prod, n in (("bottle", 5), ("board", 4), ("lonely", 1)):
        d = tmp_path / prod / "train" / "good"
        d.mkdir(parents=True)
        for i in range(n):
            Image.fromarray(_image(i * 7 + len(prod))).save(d / f"{prod}_{i}.png")
    return str(tmp_path)


def test_holdout_products_never_reach_training(products):
    names = [s.category for s in resolve_sources([{"root": products, "categories": "all"}],
                                                 exclude=["board"])]
    assert "board" not in names and "bottle" in names


def test_a_pair_is_always_from_one_product(products):
    ds = ReferencePairs(resolve_sources([{"root": products, "categories": "all"}]),
                        img_size=SIZE, samples_per_epoch=40, num_references=2, seed=1)
    assert {s.category for s in ds.sources} == {"bottle", "board"}   # 'lonely' has one image
    for i in range(len(ds)):
        item = ds[i]
        folder = os.path.dirname(item["current_path"])
        assert all(os.path.dirname(p) == folder for p in item["reference_paths"])
        assert item["references"].shape == (2, 3, SIZE, SIZE)


def test_same_image_pairs_follow_the_requested_share(products):
    src = resolve_sources([{"root": products, "categories": ["bottle"]}])
    always = ReferencePairs(src, img_size=SIZE, samples_per_epoch=20, same_image_prob=1.0)
    never = ReferencePairs(src, img_size=SIZE, samples_per_epoch=20, same_image_prob=0.0)
    assert all(always[i]["reference_paths"][0] == always[i]["current_path"] for i in range(20))
    assert all(never[i]["reference_paths"][0] != never[i]["current_path"] for i in range(20))


def test_samples_are_reproducible(products):
    src = resolve_sources([{"root": products, "categories": "all"}])
    a = ReferencePairs(src, img_size=SIZE, samples_per_epoch=8, seed=3)
    b = ReferencePairs(src, img_size=SIZE, samples_per_epoch=8, seed=3)
    assert [a[i]["current_path"] for i in range(8)] == [b[i]["current_path"] for i in range(8)]


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def test_alignment_undoes_a_known_shift():
    ref = np.array(Image.fromarray(_image(1, 128)).resize((128, 128)))
    cur = np.roll(ref, shift=(5, -4), axis=(0, 1))
    aligned, valid, info = align_to_reference(cur, ref)
    assert info["aligned"]
    assert abs(info["warp"][0, 2] - (-4)) < 1.0 and abs(info["warp"][1, 2] - 5) < 1.0
    inner = (slice(10, 118), slice(10, 118))
    assert np.abs(aligned[inner].astype(int) - ref[inner].astype(int)).mean() < 6


def test_the_empty_border_comes_from_the_reference():
    ref = _image(2, 128)
    cur = np.zeros_like(ref)
    cur[:, 6:] = ref[:, :-6]
    aligned, valid, info = align_to_reference(cur, ref)
    if info["aligned"]:
        assert (aligned[~valid] == ref[~valid]).all()


def test_implausible_warps_are_refused():
    assert plausible(np.array([[1.0, 0, 3], [0, 1.0, -2]], np.float32), 256)
    assert not plausible(np.array([[2.0, 0, 0], [0, 2.0, 0]], np.float32), 256)
    assert not plausible(np.array([[1.0, 0, 120], [0, 1.0, 0]], np.float32), 256)


def test_a_mask_follows_the_alignment():
    mask = np.zeros((64, 64), np.uint8)
    mask[10:20, 10:20] = 1
    info = {"aligned": True, "warp": np.array([[1, 0, 0], [0, 1, 0]], np.float32)}
    assert (warp_mask(mask, info) == mask).all()
    assert warp_mask(mask, {"aligned": False}) is mask


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_a_foreign_object_scores_above_an_identical_copy(model):
    scorer = ReferenceScorer(model, DEVICE, windows=(2,), out_size=SIZE, n_pseudo=2, align=False)
    ref = _image(5)
    state = scorer.prepare([ref])
    damaged = ref.copy()
    damaged[30:50, 30:50] = (10, 220, 10)
    same = scorer.compare(ref, state)
    bad = scorer.compare(damaged, state)
    assert bad["score"] > same["score"]
    assert same["map"].shape == (SIZE, SIZE)


def test_several_references_are_supported(model):
    scorer = ReferenceScorer(model, DEVICE, windows=(2,), out_size=SIZE, n_pseudo=3, align=False)
    state = scorer.prepare([_image(5), _image(6)])
    assert state.tokens.shape[1] == 2 * model.num_patches
    assert len(state.pseudo_jepa) == 3


def test_missing_calibration_means_no_verdict(tmp_path):
    calib = load_calibration(str(tmp_path / "final.pt"))
    assert calib["threshold"] == float("inf") and calib["jepa_scale"] == 1.0


def test_verdicts_are_relative_to_the_products_normal_level(model):
    """Products differ in how much two normal instances differ; the ratio rule
    divides that out, and the level is computed once per reference set."""
    scorer = ReferenceScorer(model, DEVICE, windows=(2,), out_size=SIZE, n_pseudo=3, align=False,
                             calibration={"threshold": 2.0, "rule": "ratio"})
    state = scorer.prepare([_image(5)])
    level = scorer.normal_level(state)
    assert level > 0 and scorer.normal_level(state) == level
    out = scorer.compare(_image(5), state)
    assert out["decision_value"] == pytest.approx(out["score"] / level)
    assert out["is_anomalous"] == (out["decision_value"] > 2.0)


def test_higher_sensitivity_lowers_the_threshold(model):
    scorer = ReferenceScorer(model, DEVICE, windows=(2,), out_size=SIZE, n_pseudo=2, align=False,
                             calibration={"threshold": 2.0})
    state = scorer.prepare([_image(5)])
    assert scorer.compare(_image(6), state, sensitivity=2.0)["threshold"] == pytest.approx(1.0)
    assert scorer.compare(_image(6), state, sensitivity=0.5)["threshold"] == pytest.approx(4.0)


def test_the_absolute_rule_uses_the_raw_score(model):
    scorer = ReferenceScorer(model, DEVICE, windows=(2,), out_size=SIZE, n_pseudo=2, align=False,
                             calibration={"threshold": 1.0, "rule": "absolute"})
    state = scorer.prepare([_image(5)])
    out = scorer.compare(_image(6), state)
    assert out["decision_value"] == pytest.approx(out["score"])


def test_each_reference_count_uses_its_own_threshold(model):
    calib = {"threshold": 9.0, "rule": "ratio",
             "by_references": {"1": {"ratio": 3.0, "absolute": 5.0},
                               "3": {"ratio": 2.0, "absolute": 4.0}}}
    scorer = ReferenceScorer(model, DEVICE, windows=(2,), out_size=SIZE, align=False,
                             calibration=calib)
    assert scorer.threshold_for(1) == 3.0
    assert scorer.threshold_for(2) == 3.0       # largest calibrated count not above 2
    assert scorer.threshold_for(4) == 2.0
    scorer.calibration["rule"] = "absolute"
    assert scorer.threshold_for(3) == 4.0


def test_several_references_set_the_normal_level_from_real_instances(model):
    """Perturbed copies of one photo understate how much two real parts differ."""
    scorer = ReferenceScorer(model, DEVICE, windows=(2,), out_size=SIZE, n_pseudo=4, align=False)
    one = scorer.prepare([_image(5)])
    three = scorer.prepare([_image(5), _image(6), _image(7)])
    assert one.n_real == 0 and three.n_real == 3
    assert scorer.normal_level(three) > 0
