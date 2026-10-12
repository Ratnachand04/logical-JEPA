"""Context encoder: the trainable (student) tiny Vision Transformer.

The context encoder sees only the *visible* patches of an image. Masked target
patches are removed from the sequence entirely rather than replaced with a mask
token -- this is the I-JEPA choice, and it is what forces the predictor (not the
encoder) to do the imagining.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .patch_embed import PatchEmbed, build_2d_sincos_pos_embed, gather_tokens
from .transformer import TransformerBlock, init_vit_weights, rescale_block_depth


class ContextEncoder(nn.Module):
    """Tiny ViT encoder over a subset of patch tokens.

    Args:
        img_size: input resolution (square).
        patch_size: patch side in pixels.
        in_chans: image channels.
        embed_dim: token width.
        depth: number of transformer blocks.
        num_heads: attention heads.
        mlp_ratio: FFN expansion factor.
        drop_path_rate: maximum stochastic-depth rate, linearly ramped by depth.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_cls: bool = False,
        persistent_pos: bool = False,
        input_mean: tuple | None = None,
        input_std: tuple | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        # From scratch: fixed 2-D sin-cos positions, recomputed on build (see
        # patch_embed for why). With a pretrained encoder the positions are the
        # checkpoint's own, resampled to our grid -- they cannot be recomputed,
        # so they are made persistent and travel inside every checkpoint.
        self.register_buffer(
            "pos_embed",
            build_2d_sincos_pos_embed(embed_dim, self.grid_size),
            persistent=persistent_pos,
        )

        # Pretrained ViTs were trained with a CLS token in the sequence. It is
        # prepended internally and dropped from the output, so every caller
        # still sees exactly one token per patch. Holds CLS + its position.
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls else None

        # The data pipeline normalises to mean 0.5 / std 0.5. A pretrained
        # encoder expects its own statistics (ImageNet), so the conversion is
        # done here as a fixed affine map -- the data pipeline, the heatmap
        # rendering and every saved image stay exactly as they were.
        if input_mean is not None and input_std is not None:
            mean = torch.tensor(input_mean, dtype=torch.float32).view(1, -1, 1, 1)
            std = torch.tensor(input_std, dtype=torch.float32).view(1, -1, 1, 1)
            self.register_buffer("in_scale", 0.5 / std, persistent=False)
            self.register_buffer("in_shift", (0.5 - mean) / std, persistent=False)
        else:
            self.in_scale = None
            self.in_shift = None

        # Linearly increasing stochastic depth across blocks.
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.apply(init_vit_weights)
        rescale_block_depth(self.blocks)

    def embed_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Patchify and add positions, without masking. (B, N, D)."""
        if self.in_scale is not None:
            images = images * self.in_scale + self.in_shift
        tokens = self.patch_embed(images)
        return tokens + self.pos_embed

    def forward(
        self,
        images: torch.Tensor,
        context_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the visible patches of a batch of images.

        Args:
            images: (B, C, H, W).
            context_idx: (B, K) or (K,) indices of visible patches. ``None``
                encodes the full 256-token grid, which is what the target
                encoder and the anomaly sweep use.

        Returns:
            (B, K, D) encoded context tokens, in the order given by
            ``context_idx``.
        """
        tokens = self.embed_patches(images)

        if context_idx is not None:
            tokens = gather_tokens(tokens, context_idx)

        return self.forward_tokens(tokens)

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the block stack on already-embedded tokens.

        Lets the anomaly sweep patch-embed an image once and then re-encode many
        different context subsets without redoing the convolution.
        """
        if self.cls_embed is not None:
            cls = self.cls_embed.expand(tokens.size(0), -1, -1).to(tokens.dtype)
            tokens = torch.cat([cls, tokens], dim=1)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        return tokens[:, 1:] if self.cls_embed is not None else tokens

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_context_encoder(cfg) -> ContextEncoder:
    """Construct a :class:`ContextEncoder` from a config node.

    With ``pretrained`` set, the architecture comes from the checkpoint's spec
    and any ``embed_dim`` / ``depth`` / ``num_heads`` in the config are ignored
    -- a pretrained encoder only loads into the shape it was trained in. The
    weights themselves are loaded separately (see :class:`models.LogicalJEPA`),
    so rebuilding from a saved checkpoint never needs the network.
    """
    name = cfg.get("pretrained", None)
    if not name:
        return ContextEncoder(
            img_size=cfg.get("img_size", 256),
            patch_size=cfg.get("patch_size", 16),
            in_chans=cfg.get("in_chans", 3),
            embed_dim=cfg.get("embed_dim", 256),
            depth=cfg.get("depth", 6),
            num_heads=cfg.get("num_heads", 8),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            qkv_bias=cfg.get("qkv_bias", True),
            drop_rate=cfg.get("drop_rate", 0.0),
            attn_drop_rate=cfg.get("attn_drop_rate", 0.0),
            drop_path_rate=cfg.get("drop_path_rate", 0.0),
        )

    from .pretrained import get_spec

    spec = get_spec(name)
    img_size = cfg.get("img_size", 256)
    if img_size % spec.patch_size:
        raise ValueError(f"img_size {img_size} is not a multiple of {name}'s "
                         f"patch size {spec.patch_size}")
    return ContextEncoder(
        img_size=img_size,
        patch_size=spec.patch_size,
        in_chans=3,
        embed_dim=spec.embed_dim,
        depth=spec.depth,
        num_heads=spec.num_heads,
        mlp_ratio=spec.mlp_ratio,
        qkv_bias=True,
        drop_rate=cfg.get("drop_rate", 0.0),
        attn_drop_rate=cfg.get("attn_drop_rate", 0.0),
        drop_path_rate=cfg.get("drop_path_rate", 0.0),
        use_cls=spec.use_cls,
        persistent_pos=True,
        input_mean=spec.mean,
        input_std=spec.std,
    )
