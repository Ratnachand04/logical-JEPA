"""(current, reference) pairs of normal images for the reference-conditioned model.

Every sample is two or more *normal* images of the same product: one plays the
current image, the others the references. They are usually different physical
instances, so the model learns which differences between instances are normal
(fill level, glare, the exact position of a part) instead of learning to copy.
With probability ``same_image_prob`` the reference is the current image itself,
which covers the single-reference case a user is most likely to provide.

Each image gets its own small affine jitter and photometric jitter, so the pair
is never pixel-aligned and the model has to find corresponding content rather
than compare fixed positions. The jitter is kept modest because at test time
the current image is first aligned to the reference.

Products are sampled uniformly, so a dataset with 900 images of one product and
200 of another still teaches both equally.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset

from .mvtec_loco import DEFECT_GOOD, _list_images
from .transforms import DEFAULT_MEAN, DEFAULT_STD


@dataclass(frozen=True)
class PairSource:
    """One product: a dataset root and a category folder inside it."""

    root: str
    category: str

    @property
    def name(self) -> str:
        return f"{os.path.basename(os.path.normpath(self.root))}/{self.category}"


def resolve_sources(entries, exclude=()) -> list[PairSource]:
    """Expand config entries ``{root, categories}`` into :class:`PairSource` s.

    ``categories: all`` takes every category folder that has ``train/good``.
    Names in ``exclude`` (plain category names) are left out, which is how the
    held-out products of the generalisation experiment are kept unseen.
    """
    sources = []
    for entry in entries:
        root = entry["root"]
        cats = entry.get("categories", "all")
        if cats == "all":
            cats = sorted(c for c in os.listdir(root)
                          if os.path.isdir(os.path.join(root, c, "train", DEFECT_GOOD)))
        for c in cats:
            if c not in exclude:
                sources.append(PairSource(root, c))
    return sources


def pair_augmentation(img_size: int = 256, translate: float = 0.04, degrees: float = 3.0,
                      scale: tuple = (0.95, 1.05), color_jitter: float = 0.2) -> T.Compose:
    """Independent jitter applied to each image of a pair."""
    ops = [T.ToImage(), T.Resize((img_size, img_size), antialias=True)]
    if translate or degrees or scale != (1.0, 1.0):
        ops.append(T.RandomAffine(degrees=degrees, translate=(translate, translate), scale=scale))
    if color_jitter:
        ops.append(T.ColorJitter(brightness=color_jitter, contrast=color_jitter,
                                 saturation=color_jitter * 0.5, hue=0.0))
    ops += [T.ToDtype(torch.float32, scale=True), T.Normalize(list(DEFAULT_MEAN), list(DEFAULT_STD))]
    return T.Compose(ops)


class ReferencePairs(Dataset):
    """Random (current, references) pairs of normal images from many products.

    Args:
        sources: products to draw from.
        img_size: output resolution.
        samples_per_epoch: length of one epoch (pairs are drawn at random).
        num_references: references per sample.
        same_image_prob: probability that the first reference is the current image.
        split: which normal split to read (``train`` for training, ``validation``
            for calibration).
        augment: kwargs for :func:`pair_augmentation`; ``None`` disables jitter.
        seed: base seed; each index draws from its own generator, so a sample is
            reproducible regardless of worker scheduling.
    """

    def __init__(self, sources: list[PairSource], img_size: int = 256,
                 samples_per_epoch: int = 4096, num_references: int = 1,
                 same_image_prob: float = 0.2, split: str = "train",
                 augment: dict | None = None, seed: int = 0):
        self.sources = []
        self.images: dict[str, list[str]] = {}
        for src in sources:
            paths = _list_images(os.path.join(src.root, src.category, split, DEFECT_GOOD))
            if len(paths) >= 2:
                self.sources.append(src)
                self.images[src.name] = paths
        if not self.sources:
            raise RuntimeError(f"no product with at least two '{split}/good' images")
        self.samples_per_epoch = samples_per_epoch
        self.num_references = num_references
        self.same_image_prob = same_image_prob
        self.seed = seed
        self.epoch = 0
        self.transform = (pair_augmentation(img_size, **augment) if augment is not None
                          else pair_augmentation(img_size, translate=0.0, degrees=0.0,
                                                 scale=(1.0, 1.0), color_jitter=0.0))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load(self, path: str) -> torch.Tensor:
        with Image.open(path) as im:
            return self.transform(im.convert("RGB"))

    def __getitem__(self, index: int) -> dict:
        rng = random.Random((self.seed * 1_000_003 + self.epoch) * 1_000_003 + index)
        src = rng.choice(self.sources)
        paths = self.images[src.name]
        current = rng.choice(paths)
        others = [p for p in paths if p != current]
        refs = []
        for r in range(self.num_references):
            if r == 0 and rng.random() < self.same_image_prob:
                refs.append(current)
            else:
                refs.append(rng.choice(others))
        return {
            "current": self._load(current),
            "references": torch.stack([self._load(p) for p in refs]),
            "product": src.name,
            "current_path": current,
            "reference_paths": refs,
        }

    def describe(self) -> str:
        n = sum(len(v) for v in self.images.values())
        return f"{len(self.sources)} products, {n} normal images"
