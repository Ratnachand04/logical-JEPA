"""Reference-conditioned Logical-JEPA: compare a *current* image with a reference.

The category models answer "is this region what the context of *this* image
predicts?" and need normal images of one product to learn what normal is. This
model answers a question that needs no per-product training at all:

    given the visible rest of the current image **and** a reference image of a
    correct part, what should the hidden region of the current image contain?

The predictor receives three groups of tokens:

* the current image's visible context (as in the category model),
* every token of one or more reference images,
* one mask query per hidden patch of the current image,

each tagged with a learned segment embedding so attention can tell them apart.
It is trained on pairs of *different* normal images of the same product, drawn
from many products and geometrically jittered independently, so it learns to
find the corresponding content in the reference, to tolerate normal variation
between instances, and to work on products it has never seen. At test time a
region whose content neither the reference nor the context explains -- a
missing label, a swapped part, a scratch -- gets a large prediction error.

The encoder is a frozen self-supervised ViT (DINO ViT-S/16 by default). Only
the predictor is trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from anomaly.embedding_error import jepa_loss, patch_distance

from .context_encoder import build_context_encoder
from .patch_embed import build_2d_sincos_pos_embed, gather_pos_embed, gather_tokens
from .pretrained import apply_finetune_mode, load_pretrained_into
from .target_encoder import normalize_targets
from .transformer import TransformerBlock, init_vit_weights, rescale_block_depth, trunc_normal_

SEG_CONTEXT, SEG_REFERENCE, SEG_QUERY = 0, 1, 2


class ReferencePredictor(nn.Module):
    """Predict hidden-patch embeddings of the current image from its context
    and from the full token sets of one or more reference images.

    Args:
        embed_dim: encoder width ``D``.
        predictor_dim: predictor width ``d``.
        depth / num_heads / mlp_ratio: transformer shape.
        grid_size: patch-grid side ``G``; references are ``G*G`` tokens each.
    """

    def __init__(self, embed_dim: int, predictor_dim: int = 256, depth: int = 4,
                 num_heads: int = 8, mlp_ratio: float = 4.0, grid_size: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim
        self.grid_size = grid_size
        self.num_patches = grid_size ** 2

        self.context_in = nn.Linear(embed_dim, predictor_dim)
        self.reference_in = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        # One learned vector per token group: context, reference, query.
        self.segment = nn.Parameter(torch.zeros(3, predictor_dim))
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.segment, std=0.02)

        self.register_buffer("pos_embed", build_2d_sincos_pos_embed(predictor_dim, grid_size),
                             persistent=False)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=predictor_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim, eps=1e-6)
        self.embed_out = nn.Linear(predictor_dim, embed_dim)

        self.apply(init_vit_weights)
        rescale_block_depth(self.blocks)

    def forward(self, context: torch.Tensor, context_idx: torch.Tensor,
                reference: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, Kc, D) encoded visible tokens of the current image.
            context_idx: (B, Kc) their patch indices.
            reference: (B, R*N, D) tokens of R reference images, each in grid order.
            target_idx: (B, Kt) hidden patch indices of the current image.

        Returns:
            (B, Kt, D) predicted embeddings.
        """
        B = context.size(0)
        n_ref = reference.size(1) // self.num_patches
        if reference.size(1) != n_ref * self.num_patches:
            raise ValueError("reference must hold whole images of grid_size**2 tokens")

        ctx = self.context_in(context) + gather_pos_embed(self.pos_embed, context_idx, B)
        ctx = ctx + self.segment[SEG_CONTEXT]
        ref = self.reference_in(reference) + self.pos_embed.repeat(1, n_ref, 1)
        ref = ref + self.segment[SEG_REFERENCE]
        n_target = target_idx.size(-1)
        queries = self.mask_token.expand(B, n_target, -1) + gather_pos_embed(
            self.pos_embed, target_idx, B) + self.segment[SEG_QUERY]

        x = torch.cat([ctx, ref, queries], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, -n_target:])
        return self.embed_out(x)

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ReferenceJEPA(nn.Module):
    """Frozen pretrained encoder + :class:`ReferencePredictor`."""

    def __init__(self, backbone: str = "dino_vits16", img_size: int = 256,
                 predictor_cfg: dict | None = None, loss_kind: str = "cosine",
                 target_norm: str = "layernorm", load_pretrained: bool = True):
        super().__init__()
        self.backbone = backbone
        self.encoder = build_context_encoder({"pretrained": backbone, "img_size": img_size})
        if load_pretrained:
            load_pretrained_into(self.encoder, backbone)
        apply_finetune_mode(self.encoder, "frozen")
        self.encoder.eval()

        cfg = dict(predictor_cfg or {})
        self.grid_size = self.encoder.grid_size
        self.num_patches = self.grid_size ** 2
        self.embed_dim = self.encoder.embed_dim
        self.predictor = ReferencePredictor(
            embed_dim=self.embed_dim,
            predictor_dim=cfg.get("predictor_dim", 256),
            depth=cfg.get("depth", 4),
            num_heads=cfg.get("num_heads", 8),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            grid_size=self.grid_size,
        )
        self.loss_kind = loss_kind
        self.target_norm = target_norm

    def train(self, mode: bool = True) -> "ReferenceJEPA":
        super().train(mode)
        self.encoder.eval()             # frozen: no dropout / stochastic depth, ever
        return self

    def trainable_parameters(self):
        return [p for p in self.predictor.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """LayerNorm-ed embedding of every patch. (B, N, D)."""
        return normalize_targets(self.encoder(images), self.target_norm)

    @torch.no_grad()
    def encode_references(self, references: torch.Tensor) -> torch.Tensor:
        """(B, R, C, H, W) or (B, C, H, W) -> (B, R*N, D)."""
        if references.dim() == 4:
            references = references.unsqueeze(1)
        B, R = references.shape[:2]
        tokens = self.encode(references.flatten(0, 1))                  # (B*R, N, D)
        return tokens.reshape(B, R * self.num_patches, -1)

    # ------------------------------------------------------------------ #
    def forward(self, current: torch.Tensor, references: torch.Tensor,
                mask_spec) -> tuple[torch.Tensor, dict]:
        """One training step on a batch of (current, reference) normal pairs."""
        device = current.device
        spec = mask_spec.to(device)
        B = current.size(0)
        with torch.no_grad():
            targets_full = self.encode(current)
            ref_tokens = self.encode_references(references)
            context = self.encoder(current, spec.context_idx)           # frozen: no grad needed
        ctx_idx = spec.context_idx.unsqueeze(0).expand(B, -1)

        preds, targets = [], []
        for block in spec.target_blocks:
            tgt_idx = block.unsqueeze(0).expand(B, -1)
            preds.append(self.predictor(context, ctx_idx, ref_tokens, tgt_idx))
            targets.append(gather_tokens(targets_full, tgt_idx))
        return jepa_loss(preds, targets, kind=self.loss_kind)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def error_grids(self, current: torch.Tensor, ref_tokens: torch.Tensor, mask_bank,
                    chunk: int = 16) -> dict[int, torch.Tensor]:
        """Per-scale prediction-error grids of the current image. {w: (B, G, G)}.

        ``ref_tokens`` is (B, R*N, D) from :meth:`encode_references`.
        """
        self.eval()
        B = current.size(0)
        device = current.device
        full = self.encode(current)                                     # (B, N, D)
        tokens = self.encoder.embed_patches(current)                    # (B, N, D)
        D = tokens.size(-1)

        grids: dict[int, torch.Tensor] = {}
        for window in mask_bank.scales():
            accum = torch.zeros(B, self.num_patches, device=device)
            count = torch.zeros(B, self.num_patches, device=device)
            for ctx_idx, tgt_idx, _ in mask_bank.batched(window, chunk=chunk):
                ctx_idx, tgt_idx = ctx_idx.to(device), tgt_idx.to(device)
                M = ctx_idx.size(0)
                ctx_b = ctx_idx.unsqueeze(0).expand(B, -1, -1).reshape(B * M, -1)
                tgt_b = tgt_idx.unsqueeze(0).expand(B, -1, -1).reshape(B * M, -1)
                tok_b = tokens.unsqueeze(1).expand(B, M, -1, -1).reshape(B * M, -1, D)
                encoded = self.encoder.forward_tokens(
                    torch.gather(tok_b, 1, ctx_b.unsqueeze(-1).expand(-1, -1, D)))
                ref_b = ref_tokens.unsqueeze(1).expand(B, M, -1, -1).reshape(
                    B * M, ref_tokens.size(1), D)
                pred = self.predictor(encoded, ctx_b, ref_b, tgt_b)
                full_b = full.unsqueeze(1).expand(B, M, -1, -1).reshape(B * M, -1, D)
                actual = torch.gather(full_b, 1, tgt_b.unsqueeze(-1).expand(-1, -1, D))
                err = patch_distance(pred.float(), actual.float(), kind=self.loss_kind)
                flat = tgt_idx.reshape(1, -1).expand(B, -1)
                accum.scatter_add_(1, flat, err.reshape(B, -1))
                count.scatter_add_(1, flat, torch.ones_like(err.reshape(B, -1)))
            grids[window] = (accum / count.clamp_min(1.0)).view(B, self.grid_size, self.grid_size)
        return grids

    @torch.no_grad()
    def memory_grid(self, current: torch.Tensor, ref_tokens: torch.Tensor) -> torch.Tensor:
        """Distance from each current patch to its nearest reference patch. (B, G, G).

        For unit vectors ``||u - v||^2 = 2 - 2 cos(u, v)``, so this is the
        reference patch most similar in direction.
        """
        cur = F.normalize(self.encode(current).float(), dim=-1)          # (B, N, D)
        ref = F.normalize(ref_tokens.float(), dim=-1)                    # (B, R*N, D)
        dist = torch.cdist(cur, ref).min(dim=-1).values                  # (B, N)
        return dist.view(-1, self.grid_size, self.grid_size)

    def describe(self) -> str:
        return (f"ReferenceJEPA(backbone={self.backbone} frozen, grid={self.grid_size}x"
                f"{self.grid_size}, predictor={self.predictor.num_parameters() / 1e6:.2f}M "
                f"trainable)")


def build_reference_model(cfg, load_pretrained: bool = True) -> ReferenceJEPA:
    """Build a :class:`ReferenceJEPA` from a config with a ``reference`` node."""
    node = cfg.get("model", {})
    return ReferenceJEPA(
        backbone=node.get("backbone", "dino_vits16"),
        img_size=cfg.get_path("data.img_size", 256),
        predictor_cfg=node.get("predictor", {}),
        loss_kind=node.get("loss", "cosine"),
        load_pretrained=load_pretrained,
    )
