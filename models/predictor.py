"""The JEPA predictor: a narrow transformer that imagines masked regions.

Given the encoded *context* tokens plus positional queries for the hidden
region, the predictor outputs the latent representation the target encoder is
expected to produce there. It is deliberately smaller than the encoders (3
blocks, optionally a narrower width) so that the semantic work happens in the
encoder and the predictor cannot simply memorise the dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .patch_embed import build_2d_sincos_pos_embed, gather_pos_embed
from .transformer import TransformerBlock, init_vit_weights, rescale_block_depth, trunc_normal_


class JEPAPredictor(nn.Module):
    """Predict target-region embeddings from context embeddings.

    The forward pass is:

    1. project context tokens ``embed_dim -> predictor_dim``;
    2. build one learned *mask token* per target patch, each stamped with the
       2-D sin-cos position of the patch it must predict;
    3. run context tokens and mask tokens jointly through the block stack, so
       the mask tokens can attend to the visible structure;
    4. keep only the mask-token outputs and project back to ``embed_dim``.

    Targets are predicted one *block* at a time. With ``M`` target blocks of
    ``K`` patches each, the blocks are folded into the batch dimension so all of
    them are predicted in a single pass.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        predictor_dim: int = 256,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        grid_size: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim
        self.grid_size = grid_size
        self.num_patches = grid_size**2

        self.embed_in = nn.Linear(embed_dim, predictor_dim, bias=True)

        # One learned mask token, made position-specific by the added sin-cos
        # embedding. A single shared token keeps the parameter count tiny and
        # forces positional information to come from the position table.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        trunc_normal_(self.mask_token, std=0.02)

        self.register_buffer(
            "pos_embed",
            build_2d_sincos_pos_embed(predictor_dim, grid_size),
            persistent=False,
        )

        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=predictor_dim,
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
        self.norm = nn.LayerNorm(predictor_dim, eps=1e-6)
        self.embed_out = nn.Linear(predictor_dim, embed_dim, bias=True)

        self.apply(init_vit_weights)
        rescale_block_depth(self.blocks)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        context: torch.Tensor,
        context_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the embeddings of one target block per sample.

        Args:
            context: (B, Kc, D) context-encoder output.
            context_idx: (B, Kc) or (Kc,) patch indices of those context tokens.
            target_idx: (B, Kt) or (Kt,) patch indices to predict.

        Returns:
            (B, Kt, D) predicted target embeddings.
        """
        B = context.size(0)

        # Context tokens: project and re-stamp with predictor-space positions.
        x = self.embed_in(context)
        x = x + gather_pos_embed(self.pos_embed, context_idx, B)
        n_context = x.size(1)

        # Query tokens: shared mask token + the position of each target patch.
        n_target = target_idx.size(-1)
        queries = self.mask_token.expand(B, n_target, -1)
        queries = queries + gather_pos_embed(self.pos_embed, target_idx, B)

        x = torch.cat([x, queries], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Drop the context half; only the imagined region is a prediction.
        x = x[:, n_context:, :]
        return self.embed_out(x)

    def forward_multi(
        self,
        context: torch.Tensor,
        context_idx: torch.Tensor,
        target_idx_blocks: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Predict several target blocks that may differ in size.

        Blocks of equal length are batched together; unequal lengths fall back
        to separate passes. Returns one (B, Kt_m, D) tensor per block, in the
        input order.
        """
        if not target_idx_blocks:
            return []

        sizes = [t.size(-1) for t in target_idx_blocks]
        B = context.size(0)

        if len(set(sizes)) == 1:
            # Fast path: fold the M blocks into the batch dimension.
            M, K = len(target_idx_blocks), sizes[0]
            stacked = torch.stack(
                [t if t.dim() == 2 else t.unsqueeze(0).expand(B, -1) for t in target_idx_blocks],
                dim=0,
            )                                             # (M, B, K)
            stacked = stacked.reshape(M * B, K)

            ctx_rep = context.repeat(M, 1, 1)             # (M*B, Kc, D)
            cidx = context_idx if context_idx.dim() == 2 else context_idx.unsqueeze(0).expand(B, -1)
            cidx_rep = cidx.repeat(M, 1)

            out = self.forward(ctx_rep, cidx_rep, stacked)  # (M*B, K, D)
            return list(out.reshape(M, B, K, -1).unbind(0))

        return [self.forward(context, context_idx, t) for t in target_idx_blocks]

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_predictor(cfg, embed_dim: int, grid_size: int) -> JEPAPredictor:
    """Construct a :class:`JEPAPredictor` from a config node."""
    return JEPAPredictor(
        embed_dim=embed_dim,
        predictor_dim=cfg.get("predictor_dim", embed_dim),
        depth=cfg.get("depth", 3),
        num_heads=cfg.get("num_heads", 8),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        qkv_bias=cfg.get("qkv_bias", True),
        drop_rate=cfg.get("drop_rate", 0.0),
        attn_drop_rate=cfg.get("attn_drop_rate", 0.0),
        drop_path_rate=cfg.get("drop_path_rate", 0.0),
        grid_size=grid_size,
    )
