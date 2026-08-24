"""Patch embedding and 2-D sine-cosine positional encodings.

Everything here is written from scratch: no timm, no pretrained weights. A
256x256 image with a 16x16 patch size yields a 16x16 = 256 token grid, which is
the grid the whole masking / anomaly-map pipeline operates on.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split an image into non-overlapping patches and linearly project them.

    The projection is implemented as a strided convolution, which is exactly a
    per-patch linear layer but far faster on GPU.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size {img_size} must be divisible by patch_size {patch_size}"
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size          # 16
        self.num_patches = self.grid_size**2            # 256
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_patches, embed_dim) in row-major patch order."""
        B, C, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            raise ValueError(
                f"Expected {self.img_size}x{self.img_size} input, got {H}x{W}"
            )
        x = self.proj(x)                 # (B, D, gh, gw)
        x = x.flatten(2).transpose(1, 2)  # (B, gh*gw, D)
        return x


def build_1d_sincos(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    """Classic transformer sine-cosine embedding for a 1-D coordinate array."""
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sin-cos embeddings")

    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000**omega)                      # (D/2,)

    positions = positions.reshape(-1)                 # (M,)
    angles = np.einsum("m,d->md", positions, omega)   # (M, D/2)

    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)  # (M, D)


def build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 2-D sin-cos position table of shape (1, grid_size**2, embed_dim).

    Half the channels encode the row index, half encode the column index. Fixed
    (non-learned) positions matter here: at inference we sweep target masks over
    positions that were never targets during a given training step, and a fixed
    table extrapolates consistently.
    """
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2-D sin-cos embeddings")

    coords = np.arange(grid_size, dtype=np.float32)
    grid_h, grid_w = np.meshgrid(coords, coords, indexing="ij")

    emb_h = build_1d_sincos(embed_dim // 2, grid_h)  # (N, D/2)
    emb_w = build_1d_sincos(embed_dim // 2, grid_w)  # (N, D/2)
    pos = np.concatenate([emb_h, emb_w], axis=1)     # (N, D)

    return torch.from_numpy(pos).float().unsqueeze(0)


def gather_tokens(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Select tokens by index.

    Args:
        tokens: (B, N, D) full token sequence.
        index: (B, K) or (K,) long tensor of patch indices.

    Returns:
        (B, K, D) gathered tokens.
    """
    if index.dim() == 1:
        index = index.unsqueeze(0).expand(tokens.size(0), -1)
    index = index.to(tokens.device)
    expanded = index.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
    return torch.gather(tokens, dim=1, index=expanded)


def gather_pos_embed(pos_embed: torch.Tensor, index: torch.Tensor, batch: int) -> torch.Tensor:
    """Select rows of a (1, N, D) position table for a batch of index tensors."""
    if index.dim() == 1:
        index = index.unsqueeze(0).expand(batch, -1)
    pos = pos_embed.expand(batch, -1, -1)
    return gather_tokens(pos, index)
