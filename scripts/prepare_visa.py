"""Convert VisA into the MVTec-style layout the loaders read, resized once.

VisA (Zou et al., ECCV 2022; CC BY 4.0) ships 12 products as
``<product>/Data/Images/{Normal,Anomaly}/*.JPG`` with masks in
``<product>/Data/Masks/Anomaly/*.png``. This writes

    <dst>/<product>/train/good/*.png
    <dst>/<product>/validation/good/*.png
    <dst>/<product>/test/{good,anomaly}/*.png
    <dst>/<product>/ground_truth/anomaly/<stem>/000.png

If the official ``split_csv/1cls.csv`` is supplied it is followed; otherwise a
deterministic split is used: every 10th normal image (sorted by name) goes to
test, every 20th of the remainder to validation, the rest to training, and
every anomalous image to test. VisA does not label anomalies as logical or
structural, so they are kept in one ``anomaly`` folder.

    py -3.12 scripts/prepare_visa.py --src data/_download/visa/extracted --dst data/visa_256
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from PIL import Image

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _images(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, f) for f in os.listdir(directory)
                  if f.lower().endswith(IMAGE_EXT))


def _save(src: str, dst: str, size: int, mask: bool = False) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with Image.open(src) as im:
        if mask:
            arr = np.asarray(im.convert("L").resize((size, size), Image.NEAREST)) > 0
            Image.fromarray((arr * 255).astype(np.uint8)).save(dst)
        else:
            im.convert("RGB").resize((size, size), Image.BICUBIC).save(dst)


def deterministic_split(normals: list[str]) -> dict[str, list[str]]:
    test = normals[::10]
    rest = [p for i, p in enumerate(normals) if i % 10]
    val = rest[::20]
    train = [p for i, p in enumerate(rest) if i % 20]
    return {"train": train, "validation": val, "test": test}


def official_split(csv_path: str, src: str, product: str) -> dict[str, list[str]] | None:
    if not csv_path or not os.path.isfile(csv_path):
        return None
    split = {"train": [], "validation": [], "test": []}
    with open(csv_path, newline="", encoding="utf-8") as h:
        for row in csv.DictReader(h):
            if row["object"] == product and row["label"] == "normal":
                split["train" if row["split"] == "train" else "test"].append(
                    os.path.join(src, row["image"]))
    if not split["train"]:
        return None
    train = sorted(split["train"])
    split["validation"] = train[::20]
    split["train"] = [p for i, p in enumerate(train) if i % 20]
    return split


def convert(src: str, dst: str, size: int, split_csv: str | None) -> None:
    products = sorted(p for p in os.listdir(src)
                      if os.path.isdir(os.path.join(src, p, "Data", "Images", "Normal")))
    if not products:
        raise SystemExit(f"no VisA products found under {src}")
    for product in products:
        base = os.path.join(src, product, "Data")
        normals = _images(os.path.join(base, "Images", "Normal"))
        split = official_split(split_csv, src, product) or deterministic_split(normals)
        for part, paths in split.items():
            folder = "good"
            for p in paths:
                stem = os.path.splitext(os.path.basename(p))[0]
                _save(p, os.path.join(dst, product, part, folder, f"{stem}.png"), size)
        anomalies = _images(os.path.join(base, "Images", "Anomaly"))
        for p in anomalies:
            stem = os.path.splitext(os.path.basename(p))[0]
            _save(p, os.path.join(dst, product, "test", "anomaly", f"{stem}.png"), size)
            mask = os.path.join(base, "Masks", "Anomaly", f"{stem}.png")
            if os.path.isfile(mask):
                _save(mask, os.path.join(dst, product, "ground_truth", "anomaly", stem, "000.png"),
                      size, mask=True)
        print(f"{product:12s} train {len(split['train']):4d}  val {len(split['validation']):3d}  "
              f"test good {len(split['test']):3d}  anomaly {len(anomalies):3d}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True, help="folder holding the extracted VisA products")
    parser.add_argument("--dst", default="data/visa_256")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--split-csv", default=None, help="official split_csv/1cls.csv, if available")
    args = parser.parse_args()
    convert(args.src, args.dst, args.size, args.split_csv)


if __name__ == "__main__":
    main()
