"""Image preprocessing and augmentation.

Augmentation policy is deliberately conservative. MVTec LOCO logical anomalies
are defined by *where things are and how many there are*, so any augmentation
that moves, duplicates or removes content would teach the model that those
variations are normal -- destroying exactly the signal we want to detect.

That rules out random cropping, flips on asymmetric categories, rotation and
cutout. What remains is mild photometric jitter (lighting varies between
acquisitions; component layout does not) and, optionally, small translations
that reflect real fixture play.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.v2 as T

# Per-channel statistics. ImageNet statistics are deliberately *not* used: this
# project trains from scratch, so the normalisation should describe this data.
# 0.5/0.5 maps the input to roughly [-1, 1] and is category-agnostic.
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)


def build_train_transform(
    img_size: int = 256,
    mean: tuple = DEFAULT_MEAN,
    std: tuple = DEFAULT_STD,
    color_jitter: float = 0.1,
    translate: float = 0.0,
    hflip: bool = False,
) -> T.Compose:
    """Training-time preprocessing for normal images.

    Args:
        img_size: output resolution.
        mean, std: normalisation statistics.
        color_jitter: brightness/contrast/saturation jitter strength. 0 disables.
        translate: max random shift as a fraction of the image size. Keep small
            (<=0.02) and only for categories with genuine fixture play -- larger
            shifts blur the notion of a component being in the wrong place.
        hflip: horizontal flip. Off by default; only safe for categories whose
            normal state is genuinely mirror-symmetric.
    """
    ops: list = [
        T.ToImage(),
        T.Resize((img_size, img_size), antialias=True),
    ]

    if color_jitter and color_jitter > 0:
        ops.append(
            T.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter * 0.5,
                hue=0.0,   # hue shifts change component identity (colour-coded parts)
            )
        )

    if hflip:
        ops.append(T.RandomHorizontalFlip(p=0.5))

    if translate and translate > 0:
        ops.append(T.RandomAffine(degrees=0, translate=(translate, translate)))

    ops += [
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=list(mean), std=list(std)),
    ]
    return T.Compose(ops)


def build_eval_transform(
    img_size: int = 256,
    mean: tuple = DEFAULT_MEAN,
    std: tuple = DEFAULT_STD,
) -> T.Compose:
    """Deterministic preprocessing for validation and test images."""
    return T.Compose(
        [
            T.ToImage(),
            T.Resize((img_size, img_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_mask_transform(img_size: int = 256) -> T.Compose:
    """Ground-truth mask resizing.

    Nearest-neighbour interpolation keeps the mask binary; bilinear would
    invent fractional labels along defect boundaries and quietly inflate
    pixel-level AUROC.
    """
    return T.Compose(
        [
            T.ToImage(),
            T.Resize(
                (img_size, img_size),
                interpolation=T.InterpolationMode.NEAREST,
                antialias=False,
            ),
        ]
    )


def denormalize(
    tensor: torch.Tensor,
    mean: tuple = DEFAULT_MEAN,
    std: tuple = DEFAULT_STD,
) -> torch.Tensor:
    """Invert :func:`build_eval_transform` normalisation, clamped to [0, 1].

    Accepts (C, H, W) or (B, C, H, W).
    """
    mean_t = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    if tensor.dim() == 4:
        mean_t, std_t = mean_t.unsqueeze(0), std_t.unsqueeze(0)
    return (tensor * std_t + mean_t).clamp(0.0, 1.0)


def to_numpy_image(tensor: torch.Tensor, mean=DEFAULT_MEAN, std=DEFAULT_STD) -> np.ndarray:
    """(C, H, W) normalised tensor -> (H, W, C) uint8 array for display."""
    img = denormalize(tensor.detach().cpu(), mean, std)
    return (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
