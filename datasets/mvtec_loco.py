"""MVTec LOCO AD dataset.

Expected layout (exactly as the official archive unpacks)::

    data/mvtec_loco/
      breakfast_box/
        train/good/000.png                  <- normal images ONLY
        validation/good/000.png             <- normal, held out for calibration
        test/good/000.png
        test/logical_anomalies/000.png
        test/structural_anomalies/000.png
        ground_truth/logical_anomalies/000/000.png
        ground_truth/structural_anomalies/000/000.png
      juice_bottle/ ...
      pushpins/ ...
      screw_bag/ ...
      splicing_connectors/ ...

Two details of this benchmark drive the implementation:

1. **Anomaly type is a directory name**, so every test sample carries a
   ``defect_type`` of ``good`` / ``logical_anomalies`` / ``structural_anomalies``.
   The evaluation reports the two anomaly families separately -- collapsing them
   into one AUROC would hide the entire finding this project is about.
2. **Ground truth is a *directory* of masks per image**, one PNG per annotated
   region, not a single file. They are unioned into one binary mask.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_eval_transform, build_mask_transform, build_train_transform

LOCO_CATEGORIES = (
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
)

DEFECT_GOOD = "good"
DEFECT_LOGICAL = "logical_anomalies"
DEFECT_STRUCTURAL = "structural_anomalies"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class LocoSample:
    """One image plus its labels and (for anomalies) its mask directory."""

    image_path: str
    label: int                 # 0 = normal, 1 = anomalous
    defect_type: str           # good / logical_anomalies / structural_anomalies
    mask_dir: str | None = None
    category: str = ""


def _list_images(directory: str) -> list[str]:
    """Sorted image paths in a directory; empty list if it does not exist."""
    if not os.path.isdir(directory):
        return []
    names = [n for n in os.listdir(directory) if n.lower().endswith(IMAGE_EXTS)]
    return [os.path.join(directory, n) for n in sorted(names)]


def load_union_mask(mask_dir: str | None, size: tuple[int, int]) -> Image.Image:
    """Union every mask PNG in ``mask_dir`` into one binary PIL image.

    LOCO stores one mask per annotated region (``000/000.png``, ``000/001.png``,
    ...). A logical anomaly such as "two pushpins in one compartment" is split
    across several files, so they must be ORed together.

    Args:
        mask_dir: directory of region masks, or ``None`` for a normal image.
        size: ``(width, height)`` of the source image.

    Returns:
        Mode-``L`` image with values in {0, 255}.
    """
    width, height = size
    canvas = np.zeros((height, width), dtype=np.uint8)

    if mask_dir and os.path.isdir(mask_dir):
        for path in _list_images(mask_dir):
            region = np.array(Image.open(path).convert("L"))
            if region.shape != canvas.shape:
                region = np.array(
                    Image.fromarray(region).resize((width, height), Image.NEAREST)
                )
            canvas = np.maximum(canvas, (region > 0).astype(np.uint8) * 255)

    return Image.fromarray(canvas, mode="L")


class MVTecLOCO(Dataset):
    """MVTec LOCO AD for one category and one split.

    Args:
        root: dataset root containing the category folders.
        category: one of :data:`LOCO_CATEGORIES`.
        split: ``train`` (normal only), ``validation`` (normal only) or ``test``.
        img_size: resolution images and masks are resized to.
        transform: override the default preprocessing.
        return_mask: load ground-truth masks (test split only).
        defect_types: restrict the test split to these subfolders, e.g.
            ``["good", "logical_anomalies"]`` to evaluate one anomaly family.
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        img_size: int = 256,
        transform=None,
        return_mask: bool = True,
        defect_types: list[str] | None = None,
        train_augment: bool = True,
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"split must be train/validation/test, got '{split}'")

        self.root = root
        self.category = category
        self.split = split
        self.img_size = img_size
        self.return_mask = return_mask and split == "test"

        self.category_dir = os.path.join(root, category)
        if not os.path.isdir(self.category_dir):
            raise FileNotFoundError(
                f"Category directory not found: {self.category_dir}\n"
                f"Download MVTec LOCO AD and unpack it under '{root}', or run "
                f"'py -3.12 scripts/download_data.py --synthetic' to generate a "
                f"stand-in dataset for a pipeline dry-run."
            )

        if transform is not None:
            self.transform = transform
        elif split == "train" and train_augment:
            self.transform = build_train_transform(img_size)
        else:
            self.transform = build_eval_transform(img_size)

        self.mask_transform = build_mask_transform(img_size)
        self.samples = self._build_index(defect_types)

        if not self.samples:
            raise RuntimeError(
                f"No images found for {category}/{split} under {self.category_dir}"
            )

    # ------------------------------------------------------------------ #
    def _build_index(self, defect_types: list[str] | None) -> list[LocoSample]:
        samples: list[LocoSample] = []

        if self.split in ("train", "validation"):
            # Training and validation contain normal images only -- this is what
            # makes the setup unsupervised.
            for path in _list_images(os.path.join(self.category_dir, self.split, DEFECT_GOOD)):
                samples.append(
                    LocoSample(path, 0, DEFECT_GOOD, None, self.category)
                )
            return samples

        test_dir = os.path.join(self.category_dir, "test")
        gt_dir = os.path.join(self.category_dir, "ground_truth")

        available = sorted(
            d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))
        ) if os.path.isdir(test_dir) else []
        wanted = defect_types or available

        for defect in wanted:
            if defect not in available:
                continue
            for path in _list_images(os.path.join(test_dir, defect)):
                stem = os.path.splitext(os.path.basename(path))[0]
                mask_dir = None if defect == DEFECT_GOOD else os.path.join(gt_dir, defect, stem)
                samples.append(
                    LocoSample(
                        image_path=path,
                        label=0 if defect == DEFECT_GOOD else 1,
                        defect_type=defect,
                        mask_dir=mask_dir,
                        category=self.category,
                    )
                )

        return samples

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        original_size = image.size

        item = {
            "image": self.transform(image),
            "label": torch.tensor(sample.label, dtype=torch.long),
            "defect_type": sample.defect_type,
            "path": sample.image_path,
            "category": sample.category,
            "index": index,
        }

        if self.return_mask:
            mask_img = load_union_mask(sample.mask_dir, original_size)
            mask = self.mask_transform(mask_img)
            item["mask"] = (torch.as_tensor(np.array(mask)) > 0).float().reshape(
                1, self.img_size, self.img_size
            )

        return item

    # ------------------------------------------------------------------ #
    def counts(self) -> dict[str, int]:
        """Number of samples per defect type -- printed in evaluation logs."""
        out: dict[str, int] = {}
        for sample in self.samples:
            out[sample.defect_type] = out.get(sample.defect_type, 0) + 1
        return out

    def labels(self) -> np.ndarray:
        return np.array([s.label for s in self.samples], dtype=np.int64)

    def defect_types(self) -> list[str]:
        return [s.defect_type for s in self.samples]


def available_categories(root: str) -> list[str]:
    """Categories actually present under ``root`` (real or synthetic)."""
    if not os.path.isdir(root):
        return []
    found = []
    for name in sorted(os.listdir(root)):
        if os.path.isdir(os.path.join(root, name, "train", DEFECT_GOOD)):
            found.append(name)
    return found


def build_normal_loader(cfg, category: str, split: str = "train",
                        num_workers: int | None = None):
    """Deterministic, un-augmented loader over a normal-only split.

    Calibration statistics must describe the model's behaviour on *clean* normal
    data. The training loader shuffles and applies colour jitter, both of which
    would add noise to a per-position mean and standard deviation, so statistics
    are fitted through this loader instead.
    """
    from torch.utils.data import DataLoader

    dataset = MVTecLOCO(
        cfg.get_path("data.root", "data/mvtec_loco"),
        category,
        split,
        cfg.get_path("data.img_size", 256),
        return_mask=False,
        train_augment=False,
    )
    workers = cfg.get_path("data.num_workers", 4) if num_workers is None else num_workers

    return DataLoader(
        dataset,
        batch_size=cfg.get_path("eval.batch_size", 8),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_dataloaders(cfg, category: str, num_workers: int | None = None):
    """Train / validation / test loaders for one category.

    Returns:
        ``(train_loader, val_loader, test_loader)``; the validation loader is
        ``None`` when the category has no ``validation/good`` folder.
    """
    from torch.utils.data import DataLoader

    from utils.seed import worker_init_fn

    root = cfg.get_path("data.root", "data/mvtec_loco")
    img_size = cfg.get_path("data.img_size", 256)
    batch_size = cfg.get_path("train.batch_size", 16)
    eval_batch = cfg.get_path("eval.batch_size", 8)
    workers = cfg.get_path("data.num_workers", 4) if num_workers is None else num_workers

    train_set = MVTecLOCO(
        root, category, "train", img_size,
        train_augment=cfg.get_path("data.augment", True),
    )
    test_set = MVTecLOCO(root, category, "test", img_size, return_mask=True)

    try:
        val_set = MVTecLOCO(root, category, "validation", img_size, train_augment=False)
    except RuntimeError:
        val_set = None

    common = dict(num_workers=workers, worker_init_fn=worker_init_fn,
                  pin_memory=torch.cuda.is_available(),
                  persistent_workers=workers > 0)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, **common
    )
    test_loader = DataLoader(test_set, batch_size=eval_batch, shuffle=False, **common)
    val_loader = (
        DataLoader(val_set, batch_size=eval_batch, shuffle=False, **common)
        if val_set is not None else None
    )

    return train_loader, val_loader, test_loader
