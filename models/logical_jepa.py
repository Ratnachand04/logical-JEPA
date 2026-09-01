"""Logical-JEPA: the full model.

Training (one step)::

    image --> patch embed --> [context patches] --> ContextEncoder --> Zc
                          \\
                           -> [full image]       --> TargetEncoder  --> Zt  (EMA, no grad)

    Zc + target positions --> Predictor --> Zt_hat
    loss = d(Zt_hat, Zt)

Inference (anomaly detection)::

    for every sliding window r over the patch grid:
        context = all patches except r
        A_r = d(Predictor(Encoder(context), r), TargetEncoder(image)[r])

    accumulate A_r over overlapping windows and over window scales
    --> 16x16 anomaly grid --> upsample --> 256x256 heatmap
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anomaly.embedding_error import jepa_loss, patch_distance

from .context_encoder import ContextEncoder, build_context_encoder
from .patch_embed import gather_tokens
from .predictor import JEPAPredictor, build_predictor
from .target_encoder import TargetEncoder, normalize_targets
from .vicreg import build_vicreg, collapse_report, vicreg_loss


class LogicalJEPA(nn.Module):
    """Context encoder + EMA target encoder + latent predictor.

    Args:
        encoder_cfg: config node for the context/target encoders.
        predictor_cfg: config node for the predictor.
        loss_cfg: distance settings (``kind``, ``alpha``, ``beta``) and
            ``target_norm`` (how teacher outputs are normalised).
        ema_cfg: ``base_momentum`` / ``final_momentum`` for the EMA schedule.
    """

    def __init__(
        self,
        encoder_cfg: dict | None = None,
        predictor_cfg: dict | None = None,
        loss_cfg: dict | None = None,
        ema_cfg: dict | None = None,
        regularizer_cfg: dict | None = None,
    ):
        super().__init__()
        encoder_cfg = dict(encoder_cfg or {})
        predictor_cfg = dict(predictor_cfg or {})
        loss_cfg = dict(loss_cfg or {})
        ema_cfg = dict(ema_cfg or {})

        self.context_encoder: ContextEncoder = build_context_encoder(encoder_cfg)
        self.grid_size = self.context_encoder.grid_size
        self.num_patches = self.context_encoder.num_patches
        self.embed_dim = self.context_encoder.embed_dim
        self.img_size = self.context_encoder.patch_embed.img_size
        self.patch_size = self.context_encoder.patch_embed.patch_size

        self.predictor: JEPAPredictor = build_predictor(
            predictor_cfg, embed_dim=self.embed_dim, grid_size=self.grid_size
        )
        self.target_encoder = TargetEncoder(
            self.context_encoder,
            base_momentum=ema_cfg.get("base_momentum", 0.996),
            final_momentum=ema_cfg.get("final_momentum", 1.0),
        )

        self.loss_kind = loss_cfg.get("kind", "cosine")
        self.loss_alpha = loss_cfg.get("alpha", 0.7)
        self.loss_beta = loss_cfg.get("beta", 1.0)
        self.target_norm = loss_cfg.get("target_norm", "layernorm")
        # per_patch (default) weights blocks by area; per_block gives every
        # block equal say regardless of size -- see anomaly.embedding_error.
        self.loss_reduction = loss_cfg.get("reduction", "per_patch")

        # Optional collapse regulariser. None when disabled, so the term costs
        # nothing rather than being multiplied by a zero weight.
        self.vicreg = build_vicreg(regularizer_cfg)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def forward(self, images: torch.Tensor, mask_spec) -> tuple[torch.Tensor, dict]:
        """One JEPA training step.

        Args:
            images: (B, C, H, W) batch of normal images.
            mask_spec: a :class:`masking.MaskSpec` (masks are shared across the
                batch, following the I-JEPA collator).

        Returns:
            ``(loss, stats)``.
        """
        device = images.device
        spec = mask_spec.to(device)
        B = images.size(0)

        # 1. Teacher sees the whole image; the targets are read off at the
        #    masked positions. No gradient flows through this path.
        with torch.no_grad():
            full_target = self.target_encoder(images)                  # (B, N, D)
            full_target = normalize_targets(full_target, self.target_norm)
            targets = [gather_tokens(full_target, blk) for blk in spec.target_blocks]

        # 2. Student sees only the context patches.
        context = self.context_encoder(images, spec.context_idx)       # (B, Kc, D)

        # 3. Predictor imagines each hidden block from that context.
        ctx_idx = spec.context_idx
        if ctx_idx.dim() == 1:
            ctx_idx = ctx_idx.unsqueeze(0).expand(B, -1)
        blocks = [
            blk if blk.dim() == 2 else blk.unsqueeze(0).expand(B, -1)
            for blk in spec.target_blocks
        ]
        preds = self.predictor.forward_multi(context, ctx_idx, blocks)

        loss, stats = jepa_loss(
            preds, targets,
            kind=self.loss_kind, alpha=self.loss_alpha, beta=self.loss_beta,
            reduction=self.loss_reduction,
        )
        # 4. Optional VICReg safety net, applied to the *predicted* embeddings.
        #    Applying it to the teacher would be pointless: the teacher receives
        #    no gradient, so nothing there can be regularised.
        if self.vicreg is not None:
            flat = torch.cat([p.reshape(-1, p.size(-1)) for p in preds], dim=0)
            reg_loss, reg_stats = vicreg_loss(flat, **self.vicreg)
            loss = loss + reg_loss
            stats.update(reg_stats)
            stats.update(collapse_report(flat, gamma=self.vicreg["gamma"]))

        stats["masked_ratio"] = spec.num_target_patches / self.num_patches
        stats["num_blocks"] = spec.num_targets
        return loss, stats

    @torch.no_grad()
    def update_target_encoder(self, step: int, total_steps: int) -> float:
        """Advance the EMA teacher. Call once per optimiser step."""
        momentum = self.target_encoder.momentum_at(step, total_steps)
        return self.target_encoder.update(self.context_encoder, momentum)

    def trainable_parameters(self):
        """Student + predictor parameters (the teacher is excluded by design)."""
        return list(self.context_encoder.parameters()) + list(self.predictor.parameters())

    # ------------------------------------------------------------------ #
    # Inference: masked sweep
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_targets(self, images: torch.Tensor) -> torch.Tensor:
        """Teacher embedding of every patch, normalised. (B, N, D)."""
        out = self.target_encoder(images)
        return normalize_targets(out, self.target_norm)

    @torch.no_grad()
    def sweep_scale(
        self,
        images: torch.Tensor,
        context_idx: torch.Tensor,
        target_idx: torch.Tensor,
        full_target: torch.Tensor | None = None,
        distance: str | None = None,
        alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a chunk of sweep positions for a batch of images.

        Every image is paired with every mask position, so ``B`` images and
        ``M`` positions produce ``B*M`` predictions in one batched pass.

        Args:
            images: (B, C, H, W).
            context_idx: (M, Kc) visible indices per position.
            target_idx: (M, Kt) hidden indices per position.
            full_target: precomputed teacher output (B, N, D); recomputed if
                omitted. Passing it in avoids re-encoding the image once per
                chunk, which is the single biggest saving in the sweep.
            distance: override the configured distance.
            alpha: override the ``combined`` mixing weight.

        Returns:
            ``(errors, flat_target_idx)`` where ``errors`` is (B, M, Kt) and
            ``flat_target_idx`` is (M, Kt), for scattering into the grid.
        """
        B = images.size(0)
        M, Kt = target_idx.shape
        device = images.device

        if full_target is None:
            full_target = self.encode_targets(images)

        # Patch-embed once, then replicate across mask positions.
        tokens = self.context_encoder.embed_patches(images)             # (B, N, D)
        D = tokens.size(-1)

        ctx_idx = context_idx.to(device)
        tgt_idx = target_idx.to(device)

        # (B, M, K) index grids -> flatten to a (B*M, K) batch.
        ctx_b = ctx_idx.unsqueeze(0).expand(B, -1, -1).reshape(B * M, -1)
        tgt_b = tgt_idx.unsqueeze(0).expand(B, -1, -1).reshape(B * M, -1)

        tokens_b = tokens.unsqueeze(1).expand(B, M, -1, -1).reshape(B * M, -1, D)
        ctx_tokens = torch.gather(
            tokens_b, 1, ctx_b.unsqueeze(-1).expand(-1, -1, D)
        )

        encoded = self.context_encoder.forward_tokens(ctx_tokens)       # (B*M, Kc, D)
        pred = self.predictor(encoded, ctx_b, tgt_b)                    # (B*M, Kt, D)

        target_b = full_target.unsqueeze(1).expand(B, M, -1, -1).reshape(B * M, -1, D)
        actual = torch.gather(target_b, 1, tgt_b.unsqueeze(-1).expand(-1, -1, D))

        err = patch_distance(
            pred, actual,
            kind=distance or self.loss_kind,
            alpha=self.loss_alpha if alpha is None else alpha,
            beta=self.loss_beta,
        )                                                               # (B*M, Kt)

        return err.reshape(B, M, Kt), tgt_idx

    @torch.no_grad()
    def anomaly_grids(
        self,
        images: torch.Tensor,
        mask_bank,
        chunk: int = 16,
        distance: str | None = None,
        alpha: float | None = None,
    ) -> dict[int, torch.Tensor]:
        """Per-scale anomaly grids for a batch of images.

        For each window size in ``mask_bank`` the sweep errors are scattered
        back onto the patch grid and averaged by how many windows covered each
        patch.

        Returns:
            ``{window_size: (B, grid, grid) tensor}``.
        """
        self.eval()
        B = images.size(0)
        device = images.device
        full_target = self.encode_targets(images)

        grids: dict[int, torch.Tensor] = {}

        for window in mask_bank.scales():
            accum = torch.zeros(B, self.num_patches, device=device)
            count = torch.zeros(B, self.num_patches, device=device)

            for ctx_idx, tgt_idx, _boxes in mask_bank.batched(window, chunk=chunk):
                err, tgt = self.sweep_scale(
                    images, ctx_idx, tgt_idx,
                    full_target=full_target, distance=distance, alpha=alpha,
                )                                                        # (B, M, Kt)

                flat_idx = tgt.reshape(1, -1).expand(B, -1)              # (B, M*Kt)
                accum.scatter_add_(1, flat_idx, err.reshape(B, -1))
                count.scatter_add_(1, flat_idx, torch.ones_like(err.reshape(B, -1)))

            grid = accum / count.clamp_min(1.0)
            grids[window] = grid.view(B, self.grid_size, self.grid_size)

        return grids

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        ctx = self.context_encoder.num_parameters()
        pred = self.predictor.num_parameters()
        return (
            f"LogicalJEPA(grid={self.grid_size}x{self.grid_size}, dim={self.embed_dim}, "
            f"encoder={ctx/1e6:.2f}M, predictor={pred/1e6:.2f}M, "
            f"trainable={(ctx+pred)/1e6:.2f}M, teacher={self.target_encoder.num_parameters()/1e6:.2f}M frozen)"
        )


def build_model(cfg) -> LogicalJEPA:
    """Build a :class:`LogicalJEPA` from the top-level config."""
    encoder_cfg = dict(cfg.get("encoder", {}))
    # The encoders need to know the image geometry, which lives under `data`.
    encoder_cfg.setdefault("img_size", cfg.get_path("data.img_size", 256))
    encoder_cfg.setdefault("patch_size", cfg.get_path("data.patch_size", 16))

    return LogicalJEPA(
        encoder_cfg=encoder_cfg,
        predictor_cfg=cfg.get("predictor", {}),
        loss_cfg=cfg.get("loss", {}),
        ema_cfg=cfg.get("ema", {}),
        regularizer_cfg=cfg.get("regularizer", {}),
    )
