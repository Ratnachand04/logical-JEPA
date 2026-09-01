"""Tests for the MVTec LOCO dataset loader.

The split semantics are a correctness constraint, not a convenience: `train`
and `validation` must contain **only** normal images. If an anomaly ever
appeared in either, it would enter both the training objective and the
calibration fit, and the unsupervised claim would be false while every metric
still looked plausible.

These run against a synthetic fixture with the exact LOCO directory layout, so
they execute without the 6 GB download; a real installation is additionally
checked by ``scripts/verify_dataset.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.mvtec_loco import (  # noqa: E402
    DEFECT_GOOD,
    DEFECT_LOGICAL,
    DEFECT_STRUCTURAL,
    MVTecLOCO,
    available_categories,
)
from datasets.transforms import (  # noqa: E402
    build_eval_transform,
    build_mask_transform,
    build_train_transform,
    denormalize,
)

CATEGORY = "unit_cat"


def _img(path: str, size=(80, 60), value=120) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _mask(path: str, size=(80, 60), box=(10, 10, 30, 25)) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    x0, y0, x1, y1 = box
    arr[y0:y1, x0:x1] = 255
    Image.fromarray(arr, mode="L").save(path)


@pytest.fixture(scope="module")
def loco_root(tmp_path_factory):
    """A miniature dataset with the genuine LOCO layout."""
    root = tmp_path_factory.mktemp("loco")
    base = root / CATEGORY

    for i in range(6):
        _img(str(base / "train" / "good" / f"{i:03d}.png"))
    for i in range(3):
        _img(str(base / "validation" / "good" / f"{i:03d}.png"))
    for i in range(2):
        _img(str(base / "test" / "good" / f"{i:03d}.png"))

    # Anomalies, each with a *directory* of region masks (the LOCO convention).
    for i in range(3):
        _img(str(base / "test" / DEFECT_LOGICAL / f"{i:03d}.png"))
        _mask(str(base / "ground_truth" / DEFECT_LOGICAL / f"{i:03d}" / "000.png"))
        if i == 0:                      # one image with two annotated regions
            _mask(str(base / "ground_truth" / DEFECT_LOGICAL / f"{i:03d}" / "001.png"),
                  box=(50, 35, 70, 55))

    for i in range(2):
        _img(str(base / "test" / DEFECT_STRUCTURAL / f"{i:03d}.png"))
        _mask(str(base / "ground_truth" / DEFECT_STRUCTURAL / f"{i:03d}" / "000.png"))

    return str(root)


# --------------------------------------------------------------------------- #
# Split semantics -- the unsupervised constraint
# --------------------------------------------------------------------------- #
def test_train_split_contains_only_normal_images(loco_root):
    ds = MVTecLOCO(loco_root, CATEGORY, "train", img_size=64)
    assert len(ds) == 6
    assert set(ds.defect_types()) == {DEFECT_GOOD}
    assert (ds.labels() == 0).all(), "an anomaly reached the training split"


def test_validation_split_contains_only_normal_images(loco_root):
    ds = MVTecLOCO(loco_root, CATEGORY, "validation", img_size=64)
    assert len(ds) == 3
    assert (ds.labels() == 0).all(), "an anomaly reached the calibration split"


def test_test_split_carries_all_three_defect_families(loco_root):
    ds = MVTecLOCO(loco_root, CATEGORY, "test", img_size=64)
    assert ds.counts() == {DEFECT_GOOD: 2, DEFECT_LOGICAL: 3, DEFECT_STRUCTURAL: 2}
    assert len(ds) == 7


def test_labels_follow_the_directory_name(loco_root):
    ds = MVTecLOCO(loco_root, CATEGORY, "test", img_size=64)
    for sample in ds.samples:
        expected = 0 if sample.defect_type == DEFECT_GOOD else 1
        assert sample.label == expected


def test_defect_types_can_be_filtered(loco_root):
    ds = MVTecLOCO(loco_root, CATEGORY, "test", img_size=64,
                   defect_types=[DEFECT_GOOD, DEFECT_LOGICAL])
    assert set(ds.defect_types()) == {DEFECT_GOOD, DEFECT_LOGICAL}
    assert len(ds) == 5


# --------------------------------------------------------------------------- #
# Item contents
# --------------------------------------------------------------------------- #
def test_item_has_the_expected_keys_and_shapes(loco_root):
    ds = MVTecLOCO(loco_root, CATEGORY, "test", img_size=64)
    item = ds[0]

    assert item["image"].shape == (3, 64, 64)
    assert item["image"].dtype == torch.float32
    assert item["mask"].shape == (1, 64, 64)
    assert item["label"].dtype == torch.long
    assert item["defect_type"] in (DEFECT_GOOD, DEFECT_LOGICAL, DEFECT_STRUCTURAL)
    assert os.path.isfile(item["path"])


def test_normal_test_images_get_an_empty_mask(loco_root):
    ds = MVTecLOCO(loco_root, CATEGORY, "test", img_size=64)
    for i, sample in enumerate(ds.samples):
        if sample.defect_type == DEFECT_GOOD:
            assert ds[i]["mask"].sum() == 0
            return
    pytest.fail("fixture had no normal test image")


def test_anomalous_images_get_a_non_empty_mask(loco_root):
    ds = MVTecLOCO(loco_root, CATEGORY, "test", img_size=64)
    for i, sample in enumerate(ds.samples):
        if sample.defect_type != DEFECT_GOOD:
            assert ds[i]["mask"].sum() > 0, f"{sample.image_path} has an empty mask"


def test_multi_region_ground_truth_is_unioned_through_the_dataset(loco_root):
    """End-to-end version of the union test, exercised via __getitem__."""
    ds = MVTecLOCO(loco_root, CATEGORY, "test", img_size=64)

    areas = {}
    for i, sample in enumerate(ds.samples):
        if sample.defect_type == DEFECT_LOGICAL:
            areas[os.path.basename(sample.image_path)] = float(ds[i]["mask"].sum())

    # 000.png has two annotated regions; the others have one.
    assert areas["000.png"] > areas["001.png"], "second region was dropped"


def test_masks_stay_binary_after_resizing(loco_root):
    ds = MVTecLOCO(loco_root, CATEGORY, "test", img_size=128)
    for i in range(len(ds)):
        values = torch.unique(ds[i]["mask"])
        assert set(values.tolist()).issubset({0.0, 1.0})


def test_return_mask_false_omits_the_mask(loco_root):
    ds = MVTecLOCO(loco_root, CATEGORY, "test", img_size=64, return_mask=False)
    assert "mask" not in ds[0]


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #
def test_missing_category_raises_a_helpful_error(loco_root):
    with pytest.raises(FileNotFoundError, match="Category directory not found"):
        MVTecLOCO(loco_root, "no_such_category", "train")


def test_invalid_split_is_rejected(loco_root):
    with pytest.raises(ValueError, match="split must be"):
        MVTecLOCO(loco_root, CATEGORY, "holdout")


def test_available_categories_lists_the_fixture(loco_root):
    assert available_categories(loco_root) == [CATEGORY]


def test_available_categories_on_a_missing_root_is_empty():
    assert available_categories("/definitely/not/here") == []


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def test_eval_transform_is_deterministic():
    img = Image.fromarray((np.random.rand(70, 90, 3) * 255).astype(np.uint8))
    tf = build_eval_transform(64)
    assert torch.equal(tf(img), tf(img)), "evaluation preprocessing is stochastic"


def test_train_transform_is_stochastic_when_jitter_is_on():
    img = Image.fromarray((np.random.rand(70, 90, 3) * 255).astype(np.uint8))
    tf = build_train_transform(64, color_jitter=0.4)
    assert not torch.equal(tf(img), tf(img))


def test_train_augmentation_does_not_move_content_by_default():
    """Geometry-preserving by design.

    Translation/flips would teach the model that a component moving is normal,
    destroying exactly the logical signal being detected. The default policy
    must therefore leave a constant image constant.
    """
    flat = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    out = build_train_transform(64, color_jitter=0.0, translate=0.0, hflip=False)(flat)
    assert out.std() < 1e-5, "default augmentation introduced spatial variation"


def test_mask_transform_uses_nearest_neighbour():
    arr = np.zeros((40, 40), dtype=np.uint8)
    arr[10:20, 10:20] = 255
    out = torch.as_tensor(build_mask_transform(160)(Image.fromarray(arr, mode="L")))
    assert set(torch.unique(out).tolist()).issubset({0, 255}), \
        "mask resize introduced grey values"


def test_denormalize_inverts_the_normalisation():
    original = torch.rand(3, 32, 32)
    normalised = (original - 0.5) / 0.5
    assert torch.allclose(denormalize(normalised), original, atol=1e-6)


@pytest.mark.parametrize("img_size", [256, 384, 512])
def test_transforms_support_the_larger_resolutions(img_size):
    img = Image.fromarray((np.random.rand(300, 400, 3) * 255).astype(np.uint8))
    assert build_eval_transform(img_size)(img).shape == (3, img_size, img_size)
