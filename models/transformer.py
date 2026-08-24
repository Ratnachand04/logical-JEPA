"""From-scratch Vision-Transformer building blocks.

Multi-head self-attention, the MLP block, stochastic depth and the pre-norm
transformer block are all implemented here rather than imported from timm, in
line with the project constraint of no pretrained and no black-box backbone.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth: randomly zero whole residual branches per sample."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Broadcast over every dim except batch so the whole sample is dropped.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x * mask / keep_prob


class DropPath(nn.Module):
    """Module wrapper around :func:`drop_path`."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention.

    Uses ``F.scaled_dot_product_attention`` when available (flash / memory
    efficient kernels on CUDA) and otherwise falls back to an explicit
    implementation, so the maths stays visible and CPU-only machines still run.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_fused: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_drop_p = attn_drop
        self.use_fused = use_fused and hasattr(F, "scaled_dot_product_attention")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B, N, D) -> (B, N, D).

        ``attn_mask`` is an optional additive or boolean mask broadcastable to
        (B, heads, N, N). The encoders here attend over visible tokens only, so
        it is normally ``None``.
        """
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_fused:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop_p if self.training else 0.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~attn_mask, float("-inf"))
                else:
                    attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    """Two-layer feed-forward network with GELU, the ViT default."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_scale_init: float | None = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path_rate)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        # LayerScale stabilises deeper stacks; disabled by default for the tiny
        # 6-block encoder used in the reference configuration.
        if layer_scale_init is not None:
            self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim))
            self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim))
        else:
            self.gamma1 = None
            self.gamma2 = None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), attn_mask=attn_mask)
        if self.gamma1 is not None:
            attn_out = self.gamma1 * attn_out
        x = x + self.drop_path(attn_out)

        mlp_out = self.mlp(self.norm2(x))
        if self.gamma2 is not None:
            mlp_out = self.gamma2 * mlp_out
        return x + self.drop_path(mlp_out)


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02, a: float = -2.0, b: float = 2.0):
    """In-place truncated normal init (the ViT weight initialisation)."""
    with torch.no_grad():
        tensor.normal_(0.0, std)
        # Resample the tail instead of clamping, which would pile mass on the bounds.
        lo, hi = a * std, b * std
        for _ in range(8):
            bad = (tensor < lo) | (tensor > hi)
            if not bool(bad.any()):
                break
            tensor[bad] = torch.empty(int(bad.sum()), device=tensor.device).normal_(0.0, std)
        tensor.clamp_(lo, hi)
    return tensor


def init_vit_weights(module: nn.Module) -> None:
    """Apply the standard ViT initialisation to Linear / LayerNorm / Conv2d."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        # Patch-embed conv: fan-in init over the flattened patch.
        fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
        trunc_normal_(module.weight, std=math.sqrt(1.0 / fan_in))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def rescale_block_depth(blocks: nn.ModuleList) -> None:
    """Scale residual-projection weights by 1/sqrt(2*layer_id).

    This is the depth-dependent rescaling used by I-JEPA and BEiT; it keeps the
    residual stream variance from growing with depth.
    """
    for layer_id, block in enumerate(blocks, start=1):
        denom = math.sqrt(2.0 * layer_id)
        block.attn.proj.weight.data.div_(denom)
        block.mlp.fc2.weight.data.div_(denom)
