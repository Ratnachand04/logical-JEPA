"""Tests for the few-shot training budget.

A few-shot claim is only as good as the accounting behind it. These tests pin
the places where extra normal images could leak in without any error:

* the training subset must be reproducible and independent of the order the
  operating system lists files in;
* the per-position calibration statistics must be fitted on the *same* few
  images, not on the full training split;
* evaluating a few-shot checkpoint with a config that merely inherits
  ``num_train_images: null`` must not silently widen the budget;
* validation and test must never be thinned, so a few-shot model is always
  measured on the full test set.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.mvtec_loco import (  # noqa: E402
    DEFECT_LOGICAL,
    LocoSample,
    MVTecLOCO,
    build_dataloaders,
    build_normal_loader,
    select_few_shot,
)
from utils.config import Config  # noqa: E402

CATEGORY = "fewshot_cat"
N_TRAIN = 24


def _img(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.full((40, 40, 3), value, np.uint8)).save(path)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    base = tmp_path_factory.mktemp("fs") / CATEGORY
    for i in range(N_TRAIN):
        _img(str(base / "train" / "good" / f"{i:03d}.png"), i)
    for i in range(5):
        _img(str(base / "validation" / "good" / f"{i:03d}.png"), 100 + i)
        _img(str(base / "test" / "good" / f"{i:03d}.png"), 150 + i)
        _img(str(base / "test" / DEFECT_LOGICAL / f"{i:03d}.png"), 200 + i)
        mask = np.zeros((40, 40), np.uint8)
        mask[5:15, 5:15] = 255
        os.makedirs(base / "ground_truth" / DEFECT_LOGICAL / f"{i:03d}", exist_ok=True)
        Image.fromarray(mask, mode="L").save(
            str(base / "ground_truth" / DEFECT_LOGICAL / f"{i:03d}" / "000.png"))
    return str(base.parent)


def _samples(n):
    return [LocoSample(f"/x/{i:03d}.png", 0, "good") for i in range(n)]


def _cfg(root, k, seed=0, samples=64):
    return Config({
        "data": {"root": root, "img_size": 32, "num_train_images": k,
                 "subset_seed": seed, "samples_per_epoch": samples, "num_workers": 0},
        "train": {"batch_size": 16},
        "eval": {"batch_size": 4},
    })


# --------------------------------------------------------------------------- #
# Subset selection
# --------------------------------------------------------------------------- #
def test_selects_exactly_k_images():
    assert len(select_few_shot(_samples(50), 8, seed=0)) == 8


def test_same_seed_gives_the_same_images():
    a = [s.image_path for s in select_few_shot(_samples(50), 8, seed=3)]
    b = [s.image_path for s in select_few_shot(_samples(50), 8, seed=3)]
    assert a == b


def test_different_seeds_give_different_images():
    a = {s.image_path for s in select_few_shot(_samples(50), 8, seed=0)}
    b = {s.image_path for s in select_few_shot(_samples(50), 8, seed=1)}
    assert a != b


def test_choice_does_not_depend_on_listing_order():
    """Directory listing order differs between machines; the subset must not."""
    forward = _samples(50)
    backward = list(reversed(forward))
    a = [s.image_path for s in select_few_shot(forward, 8, seed=5)]
    b = [s.image_path for s in select_few_shot(backward, 8, seed=5)]
    assert a == b


def test_a_budget_above_the_split_size_keeps_every_image():
    assert len(select_few_shot(_samples(10), 99)) == 10


def test_a_zero_budget_is_rejected():
    with pytest.raises(ValueError):
        select_few_shot(_samples(10), 0)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def test_the_train_split_is_thinned(root):
    ds = MVTecLOCO(root, CATEGORY, "train", img_size=32, max_images=5)
    assert len(ds) == 5
    assert ds.available_images == N_TRAIN


def test_validation_and_test_are_never_thinned(root):
    """A few-shot model is still measured on the full test set."""
    assert len(MVTecLOCO(root, CATEGORY, "validation", 32, max_images=2)) == 5
    assert len(MVTecLOCO(root, CATEGORY, "test", 32, max_images=2)) == 10


def test_no_budget_means_every_image(root):
    assert len(MVTecLOCO(root, CATEGORY, "train", img_size=32)) == N_TRAIN


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def test_few_images_still_give_full_epochs(root):
    """4 images at batch 16 with drop_last would otherwise yield zero steps."""
    train_loader, _, _ = build_dataloaders(_cfg(root, k=4, samples=64), CATEGORY)
    assert len(train_loader) == 64 // 16
    batch = next(iter(train_loader))
    assert batch["image"].shape[0] == 16


def test_by_default_a_few_shot_epoch_matches_a_full_data_epoch(root):
    """The budget must change which images are seen, never the compute."""
    cfg = _cfg(root, k=4)
    cfg.set_path("data.samples_per_epoch", None)
    few, _, _ = build_dataloaders(cfg, CATEGORY)
    cfg.set_path("data.num_train_images", None)
    full, _, _ = build_dataloaders(cfg, CATEGORY)
    assert len(few) == len(full) == N_TRAIN // 16


def test_an_epoch_is_never_empty(root):
    cfg = _cfg(root, k=2, samples=3)
    train_loader, _, _ = build_dataloaders(cfg, CATEGORY)
    assert len(train_loader) == 1


def test_steps_only_ever_draw_from_the_subset(root):
    train_loader, _, _ = build_dataloaders(_cfg(root, k=3, samples=96), CATEGORY)
    allowed = {s.image_path for s in train_loader.dataset.samples}
    seen = set()
    for batch in train_loader:
        seen.update(batch["path"])
    assert seen <= allowed and len(allowed) == 3


def test_calibration_statistics_use_the_same_few_images(root):
    """Positional stats must be fitted on the training subset -- nothing more."""
    cfg = _cfg(root, k=6, seed=2)
    train_loader, _, _ = build_dataloaders(cfg, CATEGORY)
    stats_loader = build_normal_loader(cfg, CATEGORY, "train")
    assert ([s.image_path for s in stats_loader.dataset.samples]
            == [s.image_path for s in train_loader.dataset.samples])


# --------------------------------------------------------------------------- #
# The budget travels with the checkpoint
# --------------------------------------------------------------------------- #
def test_evaluation_restores_the_trained_budget(tmp_path):
    """An inherited `num_train_images: null` must not widen a few-shot budget."""
    from evaluate import load_model_from_checkpoint
    from models import build_model

    trained_cfg = Config({
        "data": {"img_size": 256, "patch_size": 16,
                 "num_train_images": 8, "subset_seed": 4},
        "encoder": {"embed_dim": 32, "depth": 1, "num_heads": 4},
        "predictor": {"predictor_dim": 32, "depth": 1, "num_heads": 4},
    })
    model = build_model(trained_cfg)
    path = tmp_path / "fs.pt"
    torch.save({"model": model.state_dict(), "config": trained_cfg.to_dict()}, path)

    override = Config({"data": {"num_train_images": None, "subset_seed": 0}})
    _, cfg, _ = load_model_from_checkpoint(str(path), "cpu", override)

    assert cfg.get_path("data.num_train_images") == 8
    assert cfg.get_path("data.subset_seed") == 4


# --------------------------------------------------------------------------- #
# Study 10 runner
# --------------------------------------------------------------------------- #
def _study_cfg():
    return Config({"experiment": {"name": "fewshot", "seed": 0}, "data": {}})


def test_each_seed_draws_its_own_subset():
    """Otherwise the reported spread hides how much the image choice matters."""
    from run_ablations import _seeded_cfg

    arm = {"name": "k16", "overrides": {"data.num_train_images": 16}}
    picks = [_seeded_cfg(_study_cfg(), arm, None, s, [0, 1, 2]).get_path("data.subset_seed")
             for s in (0, 1, 2)]
    assert picks == [0, 1, 2]


def test_an_arm_can_pin_its_subset():
    from run_ablations import _seeded_cfg

    arm = {"name": "k16", "overrides": {"data.num_train_images": 16, "data.subset_seed": 7}}
    assert _seeded_cfg(_study_cfg(), arm, None, 2, [0, 1, 2]).get_path("data.subset_seed") == 7


def test_full_data_arms_are_not_given_a_subset():
    from run_ablations import _seeded_cfg

    arm = {"name": "all", "overrides": {"encoder.pretrained": None}}
    cfg = _seeded_cfg(_study_cfg(), arm, None, 1, [0, 1])
    assert cfg.get_path("data.subset_seed", None) is None


def test_arm_filter_keeps_order_and_rejects_typos():
    from run_ablations import select_arms

    arms = [{"name": n} for n in ("a", "b", "c")]
    assert [a["name"] for a in select_arms(arms, "c,a")] == ["a", "c"]
    assert select_arms(arms, None) == arms
    with pytest.raises(SystemExit):
        select_arms(arms, "a,zz")


def test_study_10_is_defined_and_its_default_arm_is_the_shipped_config():
    """configs/fewshot.yaml must be exactly the arm the study measures."""
    from utils.config import load_config

    study = load_config(os.path.join(ROOT, "configs", "ablations.yaml"), [])
    arms = {a["name"]: a["overrides"] for a in study.get_path("ablation.fewshot")}
    assert {"fs_scratch_all", "fs_scratch_k16", "fs_dino_all", "fs_dino_k16"} <= set(arms)

    shipped = load_config(os.path.join(ROOT, "configs", "fewshot.yaml"), [])
    for key, value in arms["fs_dino_k16"].items():
        assert shipped.get_path(key) == value, key
    for key in ("train.epochs", "train.warmup_epochs", "train.lr", "train.batch_size",
                "masking.strategy", "predictor.predictor_dim", "predictor.depth"):
        assert shipped.get_path(key) == study.get_path(key), key

    # ... and scores it the way Study 11's mem_add arm measured.
    scoring = {a["name"]: a["overrides"] for a in study.get_path("ablation.memory")}
    for key, value in scoring["mem_add"].items():
        assert shipped.get_path(key) == value, key
