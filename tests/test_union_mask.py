"""Tests for MVTec LOCO ground-truth mask loading.

This is the highest-risk piece of the data pipeline and the most likely place
for a *silent* bug. LOCO does not ship one mask per anomalous image; it ships a
**directory** of region masks:

    ground_truth/logical_anomalies/042/000.png
    ground_truth/logical_anomalies/042/001.png
    ...

A logical anomaly such as "two pushpins in one compartment" is annotated as
several disjoint regions. If only the first file is read, or if the union is
computed with the wrong dtype/threshold, pixel-level metrics silently drop
without ever raising an error -- the maps are still the right shape, they just
cover less than the true defect. Every assertion here exists to catch a way
that can happen quietly.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.mvtec_loco import load_union_mask  # noqa: E402


def _write_mask(path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


# --------------------------------------------------------------------------- #
# Core behaviour
# --------------------------------------------------------------------------- #
def test_single_region_is_preserved(tmp_path):
    """One region mask round-trips to a binary {0,255} image of the same shape."""
    mask_dir = tmp_path / "000"
    region = np.zeros((40, 60), dtype=np.uint8)
    region[5:15, 10:25] = 255
    _write_mask(str(mask_dir / "000.png"), region)

    out = np.array(load_union_mask(str(mask_dir), size=(60, 40)))

    assert out.shape == (40, 60)
    assert set(np.unique(out)).issubset({0, 255})
    assert out[5:15, 10:25].all()
    assert out.sum() == region.sum()


def test_two_disjoint_regions_are_unioned(tmp_path):
    """THE critical case: several region files must be ORed, not overwritten.

    If the loader returned only the last (or first) file, this test fails --
    which is exactly the silent failure mode this module exists to prevent.
    """
    mask_dir = tmp_path / "000"

    a = np.zeros((40, 60), dtype=np.uint8)
    a[2:8, 2:10] = 255                       # top-left region
    b = np.zeros((40, 60), dtype=np.uint8)
    b[30:38, 45:58] = 255                    # bottom-right region

    _write_mask(str(mask_dir / "000.png"), a)
    _write_mask(str(mask_dir / "001.png"), b)

    out = np.array(load_union_mask(str(mask_dir), size=(60, 40)))

    assert out[2:8, 2:10].all(), "first region missing from the union"
    assert out[30:38, 45:58].all(), "second region missing from the union"
    assert out.sum() == a.sum() + b.sum(), "union lost or double-counted pixels"


def test_many_regions_are_all_included(tmp_path):
    """LOCO images can carry a handful of regions; none may be dropped."""
    mask_dir = tmp_path / "000"
    expected = np.zeros((64, 64), dtype=bool)

    for i in range(7):
        region = np.zeros((64, 64), dtype=np.uint8)
        region[i * 8 : i * 8 + 5, i * 8 : i * 8 + 5] = 255
        expected[i * 8 : i * 8 + 5, i * 8 : i * 8 + 5] = True
        _write_mask(str(mask_dir / f"{i:03d}.png"), region)

    out = np.array(load_union_mask(str(mask_dir), size=(64, 64))) > 0
    assert np.array_equal(out, expected)
    assert out.sum() == 7 * 25


def test_overlapping_regions_do_not_double_count(tmp_path):
    """Overlap must stay binary -- 255 | 255 == 255, never 510 or a wraparound."""
    mask_dir = tmp_path / "000"

    a = np.zeros((32, 32), dtype=np.uint8)
    a[4:20, 4:20] = 255
    b = np.zeros((32, 32), dtype=np.uint8)
    b[12:28, 12:28] = 255                    # overlaps `a` on [12:20, 12:20]

    _write_mask(str(mask_dir / "000.png"), a)
    _write_mask(str(mask_dir / "001.png"), b)

    out = np.array(load_union_mask(str(mask_dir), size=(32, 32)))

    assert set(np.unique(out)).issubset({0, 255}), "overlap broke binarity"
    union_area = ((a > 0) | (b > 0)).sum()
    assert (out > 0).sum() == union_area


# --------------------------------------------------------------------------- #
# Degenerate / defensive cases
# --------------------------------------------------------------------------- #
def test_none_mask_dir_returns_all_zeros(tmp_path):
    """Normal images have no ground-truth directory and must yield an empty mask."""
    out = np.array(load_union_mask(None, size=(48, 32)))
    assert out.shape == (32, 48)
    assert out.sum() == 0


def test_missing_directory_returns_all_zeros(tmp_path):
    """A path that does not exist must not raise -- it means 'no annotation'."""
    out = np.array(load_union_mask(str(tmp_path / "nope"), size=(20, 20)))
    assert out.shape == (20, 20)
    assert out.sum() == 0


def test_empty_directory_returns_all_zeros(tmp_path):
    mask_dir = tmp_path / "000"
    mask_dir.mkdir()
    out = np.array(load_union_mask(str(mask_dir), size=(20, 20)))
    assert out.sum() == 0


def test_mismatched_region_size_is_resized_not_dropped(tmp_path):
    """A region stored at a different resolution must be resized, not skipped.

    LOCO's masks match their source image, but a resized copy of the dataset
    (or a hand-made mask) can disagree. Dropping the region silently would
    under-report the defect area.
    """
    mask_dir = tmp_path / "000"
    small = np.zeros((20, 30), dtype=np.uint8)
    small[5:15, 10:20] = 255
    _write_mask(str(mask_dir / "000.png"), small)

    out = np.array(load_union_mask(str(mask_dir), size=(60, 40)))   # 2x target

    assert out.shape == (40, 60)
    assert out.sum() > 0, "resized region was dropped entirely"
    assert set(np.unique(out)).issubset({0, 255}), "resize introduced grey values"


def test_nearest_resize_keeps_mask_binary(tmp_path):
    """Bilinear resizing would invent fractional labels along defect edges."""
    mask_dir = tmp_path / "000"
    region = np.zeros((17, 23), dtype=np.uint8)    # deliberately odd size
    region[3:9, 4:11] = 255
    _write_mask(str(mask_dir / "000.png"), region)

    out = np.array(load_union_mask(str(mask_dir), size=(64, 64)))
    assert set(np.unique(out)).issubset({0, 255})


def test_non_binary_input_is_thresholded(tmp_path):
    """Anti-aliased or greyscale masks must be binarised, not passed through."""
    mask_dir = tmp_path / "000"
    region = np.zeros((32, 32), dtype=np.uint8)
    region[10:20, 10:20] = 128                    # mid-grey annotation
    region[20:25, 20:25] = 7                      # very dark but non-zero
    _write_mask(str(mask_dir / "000.png"), region)

    out = np.array(load_union_mask(str(mask_dir), size=(32, 32)))

    assert set(np.unique(out)).issubset({0, 255})
    assert out[10:20, 10:20].all(), "mid-grey region lost"
    assert out[20:25, 20:25].all(), "faint non-zero region lost"


def test_files_are_read_in_sorted_order_and_all_used(tmp_path):
    """Ordering must not change the union; every file must contribute."""
    mask_dir = tmp_path / "000"
    total = np.zeros((40, 40), dtype=bool)

    # Written out of order on purpose.
    for name, (r, c) in [("002.png", (24, 4)), ("000.png", (4, 4)), ("001.png", (14, 14))]:
        region = np.zeros((40, 40), dtype=np.uint8)
        region[r : r + 6, c : c + 6] = 255
        total[r : r + 6, c : c + 6] = True
        _write_mask(str(mask_dir / name), region)

    out = np.array(load_union_mask(str(mask_dir), size=(40, 40))) > 0
    assert np.array_equal(out, total)


def test_non_image_files_are_ignored(tmp_path):
    """Stray files (e.g. .DS_Store, notes.txt) must not break loading."""
    mask_dir = tmp_path / "000"
    region = np.zeros((24, 24), dtype=np.uint8)
    region[4:12, 4:12] = 255
    _write_mask(str(mask_dir / "000.png"), region)
    (mask_dir / "notes.txt").write_text("ignore me")

    out = np.array(load_union_mask(str(mask_dir), size=(24, 24)))
    assert (out > 0).sum() == 64


@pytest.mark.parametrize("size", [(16, 16), (64, 48), (256, 256), (300, 200)])
def test_output_shape_always_matches_requested_size(tmp_path, size):
    """(width, height) in -> (height, width) array out, for any aspect ratio."""
    mask_dir = tmp_path / "000"
    region = np.zeros((50, 50), dtype=np.uint8)
    region[10:20, 10:20] = 255
    _write_mask(str(mask_dir / "000.png"), region)

    out = np.array(load_union_mask(str(mask_dir), size=size))
    assert out.shape == (size[1], size[0])
