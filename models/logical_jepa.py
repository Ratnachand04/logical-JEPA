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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from anomaly.embedding_error import jepa_loss, patch_distance

from .cardinality import CardinalityHead
from .context_encoder import ContextEncoder, build_context_encoder
from .patch_embed import gather_tokens
from .predictor import JEPAPredictor, build_predictor
from .slot_bottleneck import (
    background_slot,
    build_slot_bottleneck,
    chamfer_slot_distance,
    region_foreground_mass,
    slot_attention_entropy,
    sorted_usage,
)
from .target_encoder import TargetEncoder, normalize_targets
from .vicreg import build_vicreg, collapse_report, vicreg_loss

# Fixed seed for the slot initialisation at inference, so an image's slot
# grouping -- and therefore its cardinality score -- does not depend on the run.
_INFERENCE_SLOT_SEED = 0


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
        slots_cfg: dict | None = None,
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

        # Optional slot stage (Phases 3b/3c). The slot module learns its grouping
        # from detached teacher tokens by reconstruction alone; the JEPA only
        # sees it through the hierarchical term, with its parameters frozen.
        slots_cfg = dict(slots_cfg or {})
        self.slots = build_slot_bottleneck(slots_cfg, token_dim=self.embed_dim)
        self.card_head: CardinalityHead | None = None
        self.slot_recon_weight = float(slots_cfg.get("recon_weight", 1.0))
        self.hier_weight = float(slots_cfg.get("hierarchical_weight", 0.0))
        self.hier_start = float(slots_cfg.get("hierarchical_start", 0.3))
        self.hier_ramp = float(slots_cfg.get("hierarchical_ramp", 0.1))
        self.card_weight = 0.0
        if self.slots is not None:
            card_cfg = dict(slots_cfg.get("cardinality", {}) or {})
            if card_cfg.get("enabled", True):
                self.card_head = CardinalityHead(
                    num_slots=self.slots.num_slots, grid_size=self.grid_size,
                    pos_dim=card_cfg.get("pos_dim", 64),
                    hidden_dim=card_cfg.get("hidden_dim", 128),
                )
                self.card_weight = float(card_cfg.get("weight", 1.0))
        self.progress = 0.0

    def set_progress(self, progress: float) -> None:
        """Training progress in [0, 1]; drives the hierarchical-term ramp."""
        self.progress = float(progress)

    def hierarchical_lambda(self) -> float:
        """Current slot-term weight: 0 until ``hierarchical_start``, then a linear ramp.

        The delay is not optional. Early in training the teacher is near-random
        and the slot module has not yet grouped anything (Phase 3a: an
        unconverged bottleneck is indistinguishable from a collapsed one), so a
        slot term applied from step 0 would pull the predictor towards noise.
        """
        if self.hier_weight <= 0:
            return 0.0
        ramp = (self.progress - self.hier_start) / max(self.hier_ramp, 1e-6)
        return self.hier_weight * min(max(ramp, 0.0), 1.0)

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

        if self.slots is not None:
            aux_loss, aux_stats = self._slot_losses(full_target, ctx_idx, blocks, preds)
            loss = loss + aux_loss
            stats.update(aux_stats)

        stats["masked_ratio"] = spec.num_target_patches / self.num_patches
        stats["num_blocks"] = spec.num_targets
        return loss, stats

    def _slot_losses(self, full_target, ctx_idx, blocks, preds):
        """Slot reconstruction (3a), hierarchical slot term (3b), cardinality (3c).

        Gradient routing is the whole design, so it is spelled out:

        * reconstruction   -> slot module only (teacher tokens are detached);
        * hierarchical     -> predictor + context encoder only (slot parameters
                              are passed in detached, so the slot module cannot
                              flatten its grouping to make the term trivially 0);
        * cardinality      -> head only (its inputs are detached slot attention).
        """
        B, N, D = full_target.shape
        stats: dict = {}
        sa = self.slots.slot_attention
        init = sa.init_slots(B, full_target.device)

        recon, slots_true, attn_true = self.slots(
            full_target, self.context_encoder.pos_embed, return_attn=True, slots_init=init
        )
        recon_loss = F.mse_loss(recon.float(), full_target.float())
        total = self.slot_recon_weight * recon_loss
        stats["loss_slot_recon"] = float(recon_loss.detach())
        with torch.no_grad():
            stats["slot_entropy_ratio"] = float(
                slot_attention_entropy(attn_true.float()) / math.log(self.slots.num_slots)
            )

        # ---- 3b: hierarchical slot term ------------------------------------
        lam = self.hierarchical_lambda()
        composite = full_target
        for blk, pred in zip(blocks, preds):
            composite = composite.scatter(
                1, blk.unsqueeze(-1).expand(-1, -1, D), pred.to(composite.dtype)
            )
        frozen = {k: v.detach() for k, v in sa.named_parameters()}
        with torch.set_grad_enabled(lam > 0 and torch.is_grad_enabled()):
            slots_pred = functional_call(sa, frozen, (composite,),
                                         {"slots_init": init.detach()})
            hier = chamfer_slot_distance(slots_pred.float(), slots_true.detach().float()).mean()
        if lam > 0:
            total = total + lam * hier
        stats["loss_hier"] = float(hier.detach())
        stats["hier_lambda"] = lam

        # ---- 3c: cardinality head ------------------------------------------
        if self.card_head is not None:
            with torch.no_grad():
                attn_full = attn_true.detach().float()
                bg = background_slot(attn_full)
                ctx_tokens = gather_tokens(full_target, ctx_idx)
                _, attn_ctx = sa(ctx_tokens, return_attn=True, slots_init=init.detach())
                usage = sorted_usage(attn_ctx.float())
            card_losses = []
            for blk in blocks:
                with torch.no_grad():
                    actual = region_foreground_mass(attn_full, blk, bg)
                predicted = self.card_head(usage, blk)
                card_losses.append(F.mse_loss(predicted.float(), actual))
            card_loss = torch.stack(card_losses).mean()
            total = total + self.card_weight * card_loss
            stats["loss_card"] = float(card_loss.detach())

        return total, stats

    def aux_parameters(self) -> list:
        """Slot module + cardinality head: optimised, but on their own objectives."""
        params = []
        if self.slots is not None:
            params += list(self.slots.parameters())
        if self.card_head is not None:
            params += list(self.card_head.parameters())
        return params

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

    @property
    def has_cardinality(self) -> bool:
        return self.slots is not None and self.card_head is not None

    @torch.no_grad()
    def _inference_slot_init(self, batch: int, device) -> torch.Tensor:
        """One fixed slot draw shared by every image, so scores are reproducible
        and independent of an image's position in its batch."""
        sa = self.slots.slot_attention
        gen = torch.Generator(device=device).manual_seed(_INFERENCE_SLOT_SEED)
        return sa.init_slots(1, device, generator=gen).expand(batch, -1, -1)

    @torch.no_grad()
    def cardinality_scale(self, full_target, context_idx, target_idx,
                          attn_full, bg) -> torch.Tensor:
        """|predicted - actual| foreground mass for a chunk of sweep positions.

        Args:
            full_target: (B, N, D) normalised teacher tokens.
            context_idx / target_idx: (M, Kc) / (M, Kt) sweep indices.
            attn_full: (B, S, N) slot attention over the full image.
            bg: (B,) background slot per image.

        Returns:
            (B, M) cardinality error per image and window position.
        """
        B, N, D = full_target.shape
        M = target_idx.size(0)
        device = full_target.device

        ctx_b = context_idx.to(device).unsqueeze(0).expand(B, -1, -1).reshape(B * M, -1)
        tgt_b = target_idx.to(device).unsqueeze(0).expand(B, -1, -1).reshape(B * M, -1)
        tokens_b = full_target.unsqueeze(1).expand(B, M, -1, -1).reshape(B * M, N, D)

        ctx_tokens = gather_tokens(tokens_b, ctx_b)
        init = self._inference_slot_init(B * M, device)
        _, attn_ctx = self.slots.slot_attention(ctx_tokens, return_attn=True, slots_init=init)
        predicted = self.card_head(sorted_usage(attn_ctx.float()), tgt_b)

        attn_b = attn_full.unsqueeze(1).expand(B, M, -1, -1).reshape(B * M, *attn_full.shape[1:])
        bg_b = bg.unsqueeze(1).expand(B, M).reshape(-1)
        actual = region_foreground_mass(attn_b, tgt_b, bg_b)

        return (predicted.float() - actual).abs().view(B, M)

    @torch.no_grad()
    def anomaly_grids(
        self,
        images: torch.Tensor,
        mask_bank,
        chunk: int = 16,
        distance: str | None = None,
        alpha: float | None = None,
        with_cardinality: bool = False,
    ):
        """Per-scale anomaly grids for a batch of images.

        For each window size in ``mask_bank`` the sweep errors are scattered
        back onto the patch grid and averaged by how many windows covered each
        patch.

        Returns:
            ``{window_size: (B, grid, grid) tensor}``; with ``with_cardinality``
            a pair ``(grids, cardinality_grids)`` in the same format, where each
            window's cardinality error is spread over every patch it covers.
        """
        if with_cardinality and not self.has_cardinality:
            raise ValueError(
                "Cardinality scoring requested, but this model was trained without "
                "a cardinality head (slots.enabled / slots.cardinality.enabled)."
            )

        self.eval()
        B = images.size(0)
        device = images.device
        full_target = self.encode_targets(images)

        attn_full = bg = None
        if with_cardinality:
            init = self._inference_slot_init(B, device)
            _, attn_full = self.slots.slot_attention(full_target, return_attn=True,
                                                     slots_init=init)
            attn_full = attn_full.float()
            bg = background_slot(attn_full)

        grids: dict[int, torch.Tensor] = {}
        card_grids: dict[int, torch.Tensor] = {}

        for window in mask_bank.scales():
            accum = torch.zeros(B, self.num_patches, device=device)
            count = torch.zeros(B, self.num_patches, device=device)
            card_accum = torch.zeros(B, self.num_patches, device=device)

            for ctx_idx, tgt_idx, _boxes in mask_bank.batched(window, chunk=chunk):
                err, tgt = self.sweep_scale(
                    images, ctx_idx, tgt_idx,
                    full_target=full_target, distance=distance, alpha=alpha,
                )                                                        # (B, M, Kt)

                flat_idx = tgt.reshape(1, -1).expand(B, -1)              # (B, M*Kt)
                accum.scatter_add_(1, flat_idx, err.reshape(B, -1).float())
                count.scatter_add_(1, flat_idx, torch.ones_like(err.reshape(B, -1)).float())

                if with_cardinality:
                    card = self.cardinality_scale(full_target, ctx_idx, tgt_idx,
                                                  attn_full, bg)          # (B, M)
                    card = card.unsqueeze(-1).expand(-1, -1, tgt.size(-1))
                    card_accum.scatter_add_(1, flat_idx, card.reshape(B, -1))

            grid = accum / count.clamp_min(1.0)
            grids[window] = grid.view(B, self.grid_size, self.grid_size)
            if with_cardinality:
                card_grids[window] = (card_accum / count.clamp_min(1.0)).view(
                    B, self.grid_size, self.grid_size)

        return (grids, card_grids) if with_cardinality else grids

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        ctx = self.context_encoder.num_parameters()
        pred = self.predictor.num_parameters()
        text = (
            f"LogicalJEPA(grid={self.grid_size}x{self.grid_size}, dim={self.embed_dim}, "
            f"encoder={ctx/1e6:.2f}M, predictor={pred/1e6:.2f}M, "
            f"trainable={(ctx+pred)/1e6:.2f}M, teacher={self.target_encoder.num_parameters()/1e6:.2f}M frozen)"
        )
        if self.slots is not None:
            text += (
                f"\n  + {self.slots.describe()}, hierarchical lambda={self.hier_weight} "
                f"(from {self.hier_start:.0%} of training)"
            )
        if self.card_head is not None:
            text += f"\n  + CardinalityHead({self.card_head.num_parameters()/1e3:.1f}K params)"
        return text


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
        slots_cfg=cfg.get("slots", {}),
    )
