"""Dataset setup helper.

MVTec LOCO AD is released for non-commercial research use and is served behind a
registration form, so it cannot be fetched non-interactively. This script

* prints exactly where to download it and how the archive must be unpacked;
* verifies an existing installation (categories, splits, ground-truth format);
* generates the synthetic stand-in dataset, so the whole pipeline can be run
  before the real download arrives.

Usage::

    py -3.12 scripts/download_data.py --synthetic     # generate the stand-in
    py -3.12 scripts/download_data.py --verify        # check a real install
    py -3.12 scripts/download_data.py                 # print instructions
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.mvtec_loco import LOCO_CATEGORIES, available_categories  # noqa: E402

DOWNLOAD_URL = "https://www.mvtec.com/company/research/datasets/mvtec-loco"

INSTRUCTIONS = f"""
================================================================================
MVTec LOCO AD -- manual download required
================================================================================

The dataset is free for non-commercial research but is served behind a
registration form, so it cannot be downloaded automatically.

  1. Open {DOWNLOAD_URL}
  2. Fill in the form and download 'mvtec_loco_anomaly_detection.tar.xz'
     (about 6.4 GB, 3644 images across 5 categories)
  3. Unpack it so the layout is:

        {{root}}/
          breakfast_box/
            train/good/000.png
            validation/good/000.png
            test/good/000.png
            test/logical_anomalies/000.png
            test/structural_anomalies/000.png
            ground_truth/logical_anomalies/000/000.png
            ground_truth/structural_anomalies/000/000.png
          juice_bottle/ ...
          pushpins/ ...
          screw_bag/ ...
          splicing_connectors/ ...

  4. Verify:   py -3.12 scripts/download_data.py --verify --root {{root}}

--------------------------------------------------------------------------------
No dataset yet? Generate the synthetic stand-in instead:

    py -3.12 scripts/download_data.py --synthetic

It has the same directory layout, the same anomaly taxonomy (logical vs
structural) and the same ground-truth format, so every script in this repository
runs against it unchanged. Use it to validate the pipeline -- but report results
only from the real benchmark.
================================================================================
"""


def verify(root: str) -> bool:
    """Check an installation and report per-category split sizes."""
    print(f"Verifying dataset at: {os.path.abspath(root)}\n")

    if not os.path.isdir(root):
        print(f"  MISSING: '{root}' does not exist.")
        return False

    found = available_categories(root)
    if not found:
        print(f"  MISSING: no category has a 'train/good' folder under '{root}'.")
        return False

    print(f"  Categories found: {len(found)} -> {', '.join(found)}")

    official = [c for c in found if c in LOCO_CATEGORIES]
    if official:
        missing = [c for c in LOCO_CATEGORIES if c not in found]
        print(f"  Official LOCO categories: {len(official)}/5"
              + (f"  (missing: {', '.join(missing)})" if missing else "  -- complete"))
    else:
        print("  NOTE: no official LOCO category present -- this looks like the "
              "synthetic stand-in. Fine for development, not for reported results.")

    ok = True
    print()
    for category in found:
        base = os.path.join(root, category)
        counts = {}
        for split, sub in (
            ("train", "train/good"),
            ("validation", "validation/good"),
            ("test/good", "test/good"),
            ("test/logical", "test/logical_anomalies"),
            ("test/structural", "test/structural_anomalies"),
        ):
            path = os.path.join(base, *sub.split("/"))
            counts[split] = len(os.listdir(path)) if os.path.isdir(path) else 0

        gt_dir = os.path.join(base, "ground_truth")
        gt_ok = os.path.isdir(gt_dir)

        # Ground truth must be a *directory* of region masks per image.
        gt_note = ""
        if gt_ok:
            for family in ("logical_anomalies", "structural_anomalies"):
                fam = os.path.join(gt_dir, family)
                if os.path.isdir(fam):
                    entries = sorted(os.listdir(fam))
                    if entries and not os.path.isdir(os.path.join(fam, entries[0])):
                        gt_note = "  WARNING: ground truth should be one DIRECTORY per image"
                        ok = False

        summary = "  ".join(f"{k}={v}" for k, v in counts.items())
        status = "OK " if (counts["train"] > 0 and gt_ok) else "BAD"
        if counts["train"] == 0 or not gt_ok:
            ok = False

        print(f"  [{status}] {category:<22s} {summary}{gt_note}")

    print("\n  Result:", "dataset looks usable" if ok else "problems found -- see above")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVTec LOCO AD setup helper")
    parser.add_argument("--root", default="data/mvtec_loco", help="dataset root")
    parser.add_argument("--verify", action="store_true", help="check an existing install")
    parser.add_argument("--synthetic", action="store_true",
                        help="generate the synthetic stand-in dataset")
    parser.add_argument("--n-train", type=int, default=200,
                        help="synthetic: number of normal training images")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.synthetic:
        from datasets.synthetic_loco import generate_dataset

        generate_dataset(root=args.root, n_train=args.n_train, seed=args.seed)
        print()
        verify(args.root)
        print("\nNext:  py -3.12 train.py --config configs/loco.yaml")
        return

    if args.verify:
        sys.exit(0 if verify(args.root) else 1)

    print(INSTRUCTIONS.replace("{root}", args.root))
    if os.path.isdir(args.root):
        print()
        verify(args.root)


if __name__ == "__main__":
    main()
