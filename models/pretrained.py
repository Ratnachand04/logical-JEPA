"""Pretrained encoder initialisation for few-shot Logical-JEPA.

The from-scratch model needs hundreds of normal images because its encoder
starts from noise. Initialising the encoder from a self-supervised checkpoint
removes that cost: the JEPA predictor then only has to learn *this part's*
layout, which a handful of images is enough for.

Why DINO and not I-JEPA
-----------------------
Meta released I-JEPA only at ViT-H/14 size: a 10.4 GB checkpoint and ~632M
parameters. The anomaly sweep runs ~299 forward passes per image, so that
encoder would take minutes per image on a laptop GPU, if it fit at all.

DINO ViT-S/16 is the closest practical substitute:

* same family -- self-supervised, no labels, an EMA teacher distilling into a
  student, which is the mechanism :mod:`models.target_encoder` already uses;
* same 16-px patch, so it lands on our 16x16 grid exactly (positions are
  interpolated 14x14 -> 16x16, the patch projection is reused as-is);
* 86.7 MB, ~22M parameters.

Everything downstream -- masking, the JEPA predictor, the EMA teacher, the
exhaustive sweep, scoring, calibration -- is unchanged. Only the encoder's
starting weights differ, so the model is still a JEPA.

Finetune modes
--------------
``frozen``   encoder fixed; only the predictor trains. The right default for
             very few images: ~1.4M trainable parameters cannot memorise much.
``last_k``   the last ``k`` blocks plus the final norm also adapt.
``full``     every encoder weight trains (the from-scratch behaviour).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FINETUNE_MODES = ("frozen", "last_k", "full")


@dataclass(frozen=True)
class PretrainedSpec:
    """Everything needed to rebuild and load one pretrained encoder."""

    name: str
    url: str
    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int
    source_grid: int                 # grid the checkpoint was trained at
    mlp_ratio: float = 4.0
    use_cls: bool = True
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    size_bytes: int | None = None
    sha256: str | None = None        # pinned after first verified download
    description: str = ""
    licence: str = ""
    extra: dict = field(default_factory=dict)


PRETRAINED: dict[str, PretrainedSpec] = {
    "dino_vits16": PretrainedSpec(
        name="dino_vits16",
        url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/"
            "dino_deitsmall16_pretrain.pth",
        embed_dim=384, depth=12, num_heads=6, patch_size=16, source_grid=14,
        size_bytes=86710517,
        sha256="1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7",
        description="DINO ViT-S/16, self-supervised on ImageNet-1k (no labels)",
        licence="Apache-2.0 (facebookresearch/dino)",
    ),
    "dino_vitb16": PretrainedSpec(
        name="dino_vitb16",
        url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/"
            "dino_vitbase16_pretrain.pth",
        embed_dim=768, depth=12, num_heads=12, patch_size=16, source_grid=14,
        size_bytes=343242485,
        description="DINO ViT-B/16, self-supervised on ImageNet-1k (no labels)",
        licence="Apache-2.0 (facebookresearch/dino)",
    ),
}


def get_spec(name: str) -> PretrainedSpec:
    if name not in PRETRAINED:
        raise ValueError(
            f"Unknown pretrained encoder '{name}'. Available: {sorted(PRETRAINED)}"
        )
    return PRETRAINED[name]


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def checkpoint_path(name: str) -> str:
    """Where :func:`fetch_state_dict` caches a checkpoint."""
    spec = get_spec(name)
    hub = torch.hub.get_dir()
    return os.path.join(hub, "checkpoints", os.path.basename(spec.url))


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(name: str, path: str | None = None) -> None:
    """Fail loudly on a truncated or substituted download.

    A truncated file usually still unpickles into *something*, so size and
    hash are checked before any weight is trusted.
    """
    spec = get_spec(name)
    path = path or checkpoint_path(name)
    if spec.size_bytes is not None:
        size = os.path.getsize(path)
        if size != spec.size_bytes:
            raise RuntimeError(
                f"{path} is {size} bytes, expected {spec.size_bytes}: the download is "
                f"incomplete or not the published checkpoint. Delete it and retry."
            )
    if spec.sha256 is not None:
        actual = _sha256(path)
        if actual != spec.sha256:
            raise RuntimeError(
                f"{path} has sha256 {actual}, expected {spec.sha256}. Refusing to load."
            )


def fetch_state_dict(name: str, progress: bool = True) -> dict[str, torch.Tensor]:
    """Download (once, cached) and return the raw encoder state dict."""
    spec = get_spec(name)
    try:
        state = torch.hub.load_state_dict_from_url(
            spec.url, map_location="cpu", progress=progress, weights_only=True,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with a useful message
        raise RuntimeError(
            f"Could not obtain pretrained weights '{name}' from {spec.url} ({exc}). "
            f"Run `py -3.12 scripts/fetch_pretrained.py --name {name}` on a machine "
            f"with internet access, or set encoder.pretrained: null to train from "
            f"scratch."
        ) from exc

    verify_checkpoint(name)

    # Some releases wrap the weights ({'model': ...}, {'teacher': ...}).
    for wrapper in ("teacher", "model", "state_dict"):
        if isinstance(state, dict) and wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]

    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


# --------------------------------------------------------------------------- #
# Weight transfer
# --------------------------------------------------------------------------- #
def interpolate_pos_embed(pos: torch.Tensor, src_grid: int, dst_grid: int) -> torch.Tensor:
    """Resize a (1, src^2, D) patch-position table to (1, dst^2, D), bicubic.

    The standard ViT resolution transfer: positions are a smooth 2-D field, so
    resampling them preserves "which patch is where" at the new grid.
    """
    if src_grid == dst_grid:
        return pos.clone()
    dim = pos.size(-1)
    grid = pos.reshape(1, src_grid, src_grid, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(dst_grid, dst_grid), mode="bicubic",
                         align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(1, dst_grid * dst_grid, dim).to(pos.dtype)


@torch.no_grad()
def load_pretrained_into(encoder, name: str,
                         state_dict: dict[str, torch.Tensor] | None = None) -> dict:
    """Copy a pretrained ViT into a :class:`models.ContextEncoder`.

    The encoder's own block layout (``norm1 / attn.qkv / attn.proj / norm2 /
    mlp.fc1 / mlp.fc2``) is the standard ViT one, so weights transfer
    one-to-one. Every encoder weight must be covered and every shape must
    match -- a partial load would leave random layers inside a "pretrained"
    model and nothing would flag it, so this refuses instead.

    Returns:
        A small report (counts and the interpolation applied), for the logs.
    """
    spec = get_spec(name)
    state = state_dict if state_dict is not None else fetch_state_dict(name)

    if encoder.embed_dim != spec.embed_dim or len(encoder.blocks) != spec.depth:
        raise ValueError(
            f"encoder is dim={encoder.embed_dim} depth={len(encoder.blocks)}, but "
            f"'{name}' is dim={spec.embed_dim} depth={spec.depth}"
        )

    own = dict(encoder.named_parameters())
    loaded, skipped = set(), []

    for key, value in state.items():
        if key in ("cls_token", "pos_embed") or key.startswith("head"):
            continue
        if key not in own:
            skipped.append(key)
            continue
        if own[key].shape != value.shape:
            raise ValueError(f"shape mismatch for {key}: {tuple(own[key].shape)} "
                             f"vs checkpoint {tuple(value.shape)}")
        own[key].copy_(value.to(own[key].dtype))
        loaded.add(key)

    # Positions: drop the CLS slot, resample patch positions to our grid.
    pos = state["pos_embed"]
    has_cls = pos.size(1) == spec.source_grid**2 + 1
    patch_pos = pos[:, 1:] if has_cls else pos
    encoder.pos_embed.copy_(
        interpolate_pos_embed(patch_pos, spec.source_grid, encoder.grid_size)
    )

    if getattr(encoder, "cls_embed", None) is not None:
        if "cls_token" not in state or not has_cls:
            raise ValueError(f"'{name}' has no CLS token, but the encoder expects one")
        # The CLS token never moves, so its position is folded in once here.
        encoder.cls_embed.copy_(state["cls_token"] + pos[:, :1])
        loaded.add("cls_embed")

    missing = sorted(set(own) - loaded)
    if missing:
        raise RuntimeError(
            f"pretrained '{name}' left {len(missing)} encoder weights uninitialised, "
            f"e.g. {missing[:4]}. Refusing a partial load."
        )

    return {
        "name": name,
        "tensors_loaded": len(loaded),
        "tensors_skipped": skipped,
        "pos_grid": f"{spec.source_grid}x{spec.source_grid} -> "
                    f"{encoder.grid_size}x{encoder.grid_size}",
    }


def apply_finetune_mode(encoder, mode: str = "frozen", last_k: int = 2) -> int:
    """Set ``requires_grad`` on the encoder. Returns the trainable count."""
    if mode not in FINETUNE_MODES:
        raise ValueError(f"Unknown finetune mode '{mode}'. Use one of {FINETUNE_MODES}")

    for param in encoder.parameters():
        param.requires_grad = mode == "full"

    if mode == "last_k":
        if not 1 <= last_k <= len(encoder.blocks):
            raise ValueError(f"finetune_last_k={last_k} must be in 1..{len(encoder.blocks)}")
        for block in list(encoder.blocks)[-last_k:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in encoder.norm.parameters():
            param.requires_grad = True

    return sum(p.numel() for p in encoder.parameters() if p.requires_grad)


def describe_spec(name: str) -> str:
    spec = get_spec(name)
    return (f"{spec.description} [{spec.embed_dim}d x {spec.depth} blocks, "
            f"{(spec.size_bytes or 0) / 1e6:.1f} MB, {spec.licence}]")


__all__ = [
    "PRETRAINED", "PretrainedSpec", "FINETUNE_MODES", "IMAGENET_MEAN", "IMAGENET_STD",
    "get_spec", "checkpoint_path", "fetch_state_dict", "verify_checkpoint",
    "interpolate_pos_embed", "load_pretrained_into", "apply_finetune_mode",
    "describe_spec",
]
