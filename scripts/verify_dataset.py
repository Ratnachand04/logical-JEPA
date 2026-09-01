"""Strict structural verification of an MVTec LOCO AD installation.

``scripts/download_data.py --verify`` gives a friendly summary. This script is
the opposite: it is designed to **fail loudly**, and it checks the things that
would otherwise go wrong silently and only show up as quietly-wrong metrics:

* a category missing a split, or a split that is empty;
* an anomalous image with **no ground-truth directory**, which would be scored
  against an all-zero mask and drag pixel AUROC down for no visible reason;
* ground truth stored as a flat ``042.png`` instead of a directory ``042/``,
  which the union loader would skip entirely;
* an *empty* ground-truth directory, or one whose masks are all zero;
* a mask whose resolution disagrees with its source image;
* **any anomalous image inside train/ or validation/**, which would break the
  unsupervised guarantee.

Exit code is 0 only when every check passes.

Usage::

    py -3.12 scripts/verify_dataset.py
    py -3.12 scripts/verify_dataset.py --root data/mvtec_loco --strict-loco
    py -3.12 scripts/verify_dataset.py --sample-masks 25
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.mvtec_loco import (  # noqa: E402
    DEFECT_GOOD,
    DEFECT_LOGICAL,
    DEFECT_STRUCTURAL,
    IMAGE_EXTS,
    LOCO_CATEGORIES,
    available_categories,
    load_union_mask,
)

ANOMALY_FAMILIES = (DEFECT_LOGICAL, DEFECT_STRUCTURAL)


class Report:
    """Collects problems so the whole dataset is checked in one pass."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"  [FAIL] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"  [WARN] {message}")

    def ok(self, message: str) -> None:
        self.info.append(message)
        print(f"  [ ok ] {message}")

    @property
    def failed(self) -> bool:
        return bool(self.errors)


def _images(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, n)
        for n in os.listdir(directory)
        if n.lower().endswith(IMAGE_EXTS)
    )


# --------------------------------------------------------------------------- #
def check_category(root: str, category: str, report: Report, sample_masks: int) -> dict:
    """Verify one category end to end. Returns its split counts."""
    print(f"\n--- {category} ---")
    base = os.path.join(root, category)
    counts: dict[str, int] = {}

    # ---- splits exist and are populated ------------------------------- #
    for split, sub in (
        ("train", ("train", DEFECT_GOOD)),
        ("validation", ("validation", DEFECT_GOOD)),
        ("test/good", ("test", DEFECT_GOOD)),
        ("test/logical", ("test", DEFECT_LOGICAL)),
        ("test/structural", ("test", DEFECT_STRUCTURAL)),
    ):
        path = os.path.join(base, *sub)
        found = _images(path)
        counts[split] = len(found)

        if split == "train" and not found:
            report.error(f"{category}: train/good is empty or missing ({path})")
        elif split == "validation" and not found:
            report.warn(f"{category}: no validation/good -- calibration will fall back "
                        f"to the training split")
        elif not found:
            report.warn(f"{category}: {split} is empty ({path})")

    if counts["test/logical"] == 0 and counts["test/structural"] == 0:
        report.error(f"{category}: test set contains no anomalies at all")

    # ---- train / validation must be anomaly-free ---------------------- #
    for split in ("train", "validation"):
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            continue
        stray = [
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d)) and d != DEFECT_GOOD
        ]
        if stray:
            report.error(
                f"{category}: {split}/ contains non-'good' subfolders {stray} -- "
                f"anomalies must never reach training or calibration"
            )

    # ---- ground truth ------------------------------------------------- #
    gt_root = os.path.join(base, "ground_truth")
    if not os.path.isdir(gt_root):
        if counts["test/logical"] or counts["test/structural"]:
            report.error(f"{category}: anomalies present but ground_truth/ is missing")
        return counts

    checked, empty_dirs, missing_dirs, flat_files, size_mismatch = 0, 0, 0, 0, 0

    for family in ANOMALY_FAMILIES:
        test_dir = os.path.join(base, "test", family)
        gt_dir = os.path.join(gt_root, family)

        for image_path in _images(test_dir):
            stem = os.path.splitext(os.path.basename(image_path))[0]
            region_dir = os.path.join(gt_dir, stem)

            # A flat file where a directory belongs is silently skipped by the
            # union loader, so the image would score against an empty mask.
            for ext in IMAGE_EXTS:
                if os.path.isfile(os.path.join(gt_dir, stem + ext)):
                    flat_files += 1
                    report.error(
                        f"{category}/{family}/{stem}: ground truth is a flat file, "
                        f"expected a directory of region masks"
                    )
                    break

            if not os.path.isdir(region_dir):
                missing_dirs += 1
                report.error(
                    f"{category}/{family}/{stem}: no ground-truth directory "
                    f"({region_dir})"
                )
                continue

            regions = _images(region_dir)
            if not regions:
                empty_dirs += 1
                report.error(f"{category}/{family}/{stem}: ground-truth directory is empty")
                continue

            # Spot-check a subset: decoding every mask of 5 categories is slow.
            if checked < sample_masks:
                checked += 1
                with Image.open(image_path) as im:
                    width, height = im.size

                union = np.array(load_union_mask(region_dir, (width, height)))
                if union.sum() == 0:
                    report.error(
                        f"{category}/{family}/{stem}: union of {len(regions)} region "
                        f"mask(s) is entirely zero"
                    )

                for region_path in regions:
                    with Image.open(region_path) as rm:
                        if rm.size != (width, height):
                            size_mismatch += 1
                            report.warn(
                                f"{category}/{family}/{stem}: mask "
                                f"{os.path.basename(region_path)} is {rm.size}, image is "
                                f"{(width, height)} -- it will be resized"
                            )
                            break

    if not (flat_files or missing_dirs or empty_dirs):
        report.ok(
            f"{category}: ground truth well-formed "
            f"({checked} image(s) mask-verified, {size_mismatch} resized)"
        )

    report.ok(
        f"{category}: train={counts['train']} val={counts['validation']} "
        f"test(good/logical/structural)="
        f"{counts['test/good']}/{counts['test/logical']}/{counts['test/structural']}"
    )
    return counts


# --------------------------------------------------------------------------- #
def verify(root: str, strict_loco: bool, sample_masks: int) -> Report:
    report = Report()
    print(f"Verifying dataset at: {os.path.abspath(root)}")

    if not os.path.isdir(root):
        report.error(f"root directory does not exist: {root}")
        return report

    categories = available_categories(root)
    if not categories:
        report.error(f"no category with a train/good folder under {root}")
        return report

    print(f"Categories found ({len(categories)}): {', '.join(categories)}")

    if strict_loco:
        missing = [c for c in LOCO_CATEGORIES if c not in categories]
        if missing:
            report.error(
                f"--strict-loco: missing official categories {missing}. "
                f"All five are required for a benchmark run."
            )
        else:
            report.ok("all five official LOCO categories present")
    else:
        official = [c for c in categories if c in LOCO_CATEGORIES]
        if not official:
            report.warn(
                "no official LOCO category present -- this looks like the synthetic "
                "stand-in. Fine for development, NOT for reported results."
            )

    totals = {"train": 0, "validation": 0, "test/good": 0,
              "test/logical": 0, "test/structural": 0}
    for category in categories:
        counts = check_category(root, category, report, sample_masks)
        for key in totals:
            totals[key] += counts.get(key, 0)

    print("\n--- totals ---")
    for key, value in totals.items():
        print(f"  {key:<18s} {value}")
    print(f"  {'TOTAL images':<18s} {sum(totals.values())}")

    print("\n=== result ===")
    if report.failed:
        print(f"  FAILED: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    else:
        print(f"  PASSED: 0 errors, {len(report.warnings)} warning(s)")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly verify an MVTec LOCO install")
    parser.add_argument("--root", default="data/mvtec_loco")
    parser.add_argument("--strict-loco", action="store_true",
                        help="require all five official categories to be present")
    parser.add_argument("--sample-masks", type=int, default=20,
                        help="how many ground-truth unions to decode per category")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify(args.root, args.strict_loco, args.sample_masks)
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
