"""Pre-resize MVTec LOCO to the training resolution.

LOCO ships images at up to 1700x1000. The loader resizes every one to 256x256
on every epoch, which measures **84 ms per image** on this machine -- so a
372-image category spends ~31 s of CPU per epoch on decoding alone, and that,
not the GPU, sets the training speed.

This script does the resize once. It is not an approximation: the loader would
have produced the same 256x256 tensor anyway, so the only difference is that
the work happens once instead of once per epoch.

Two details that would otherwise corrupt the labels:

* **Masks are resized with NEAREST**, never bilinear. Interpolating a mask
  invents fractional labels along every defect boundary, which quietly inflates
  pixel-level metrics. Note LOCO's masks are *not* {0, 255}: they carry values
  in the 234-255 range, so "binary" is the wrong mental model -- the loader
  thresholds at ``> 0``. NEAREST preserves the original value set exactly
  (verified: the resize introduces no new values), which is what keeps that
  threshold meaning the same thing before and after caching.
* **Ground truth keeps its directory-per-image layout**, so the union of region
  masks still happens downstream exactly as it does on the original data.

Usage::

    py -3.12 scripts/prepare_loco.py --src data/mvtec_loco_real --dst data/mvtec_loco_256
    py -3.12 scripts/prepare_loco.py --src <...> --dst <...> --size 384
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.mvtec_loco import IMAGE_EXTS, available_categories  # noqa: E402

# Splits carrying photographs (resampled smoothly) versus annotation masks
# (resampled by nearest neighbour).
IMAGE_SPLITS = ("train", "validation", "test")
MASK_ROOT = "ground_truth"


def _resize_image(src: str, dst: str, size: int, is_mask: bool) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with Image.open(src) as im:
        if is_mask:
            # NEAREST reproduces the original label values exactly; anything
            # else blends neighbouring labels and invents edge values.
            out = im.convert("L").resize((size, size), Image.NEAREST)
        else:
            out = im.convert("RGB").resize((size, size), Image.BILINEAR)
        out.save(dst, optimize=False, compress_level=1)


def _walk_images(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(IMAGE_EXTS):
                yield os.path.join(dirpath, name)


def prepare_category(src_root: str, dst_root: str, category: str, size: int) -> dict:
    """Resize one category, preserving the LOCO directory layout exactly."""
    src_cat = os.path.join(src_root, category)
    dst_cat = os.path.join(dst_root, category)

    counts = {"images": 0, "masks": 0}

    for split in IMAGE_SPLITS:
        split_dir = os.path.join(src_cat, split)
        if not os.path.isdir(split_dir):
            continue
        for path in _walk_images(split_dir):
            rel = os.path.relpath(path, src_cat)
            _resize_image(path, os.path.join(dst_cat, rel), size, is_mask=False)
            counts["images"] += 1

    mask_dir = os.path.join(src_cat, MASK_ROOT)
    if os.path.isdir(mask_dir):
        for path in _walk_images(mask_dir):
            rel = os.path.relpath(path, src_cat)
            _resize_image(path, os.path.join(dst_cat, rel), size, is_mask=True)
            counts["masks"] += 1

    # Carry the licence and readme across; this data stays MVTec's.
    for extra in ("readme.txt", "license.txt", "defects_config.json"):
        src_extra = os.path.join(src_cat, extra)
        if os.path.isfile(src_extra):
            shutil.copy2(src_extra, os.path.join(dst_cat, extra))

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-resize MVTec LOCO for training")
    parser.add_argument("--src", default="data/mvtec_loco_real")
    parser.add_argument("--dst", default="data/mvtec_loco_256")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--categories", default=None,
                        help="comma-separated subset; default is every category found")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    categories = available_categories(args.src)
    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",")}
        categories = [c for c in categories if c in wanted]

    if not categories:
        raise SystemExit(f"no categories found under {args.src}")

    print(f"Pre-resizing {len(categories)} categories to {args.size}x{args.size}")
    print(f"  {args.src} -> {args.dst}\n")

    totals = {"images": 0, "masks": 0}
    started = time.time()

    for category in categories:
        t0 = time.time()
        counts = prepare_category(args.src, args.dst, category, args.size)
        totals["images"] += counts["images"]
        totals["masks"] += counts["masks"]
        print(f"  {category:<24s} {counts['images']:5d} images  "
              f"{counts['masks']:5d} masks   ({time.time()-t0:.0f}s)")

    # Copy the top-level licence so the cache is not stripped of its terms.
    for extra in ("readme.txt", "license.txt"):
        src_extra = os.path.join(args.src, extra)
        if os.path.isfile(src_extra):
            os.makedirs(args.dst, exist_ok=True)
            shutil.copy2(src_extra, os.path.join(args.dst, extra))

    print(f"\nDone in {time.time()-started:.0f}s: "
          f"{totals['images']} images, {totals['masks']} masks")
    print(f"\nVerify with:\n  py -3.12 scripts/verify_dataset.py "
          f"--root {args.dst} --strict-loco")


if __name__ == "__main__":
    main()
