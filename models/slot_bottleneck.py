"""Object-centric slot bottleneck (Phase 3a).

Motivation
----------
The predictor currently reasons over a flat 16x16 patch grid. Nothing in that
representation corresponds to a *component*, so "one screw too many" has to be
inferred from patch statistics rather than read off a count. Slot attention
gives the predictor a small set of vectors that compete to explain the scene,
which is the representation a cardinality question can actually be asked of.

Implemented from scratch: slots are queries, patch tokens are keys/values, and
attention is normalised **over the slots** (not over the tokens). That single
transpose is what makes the mechanism competitive -- each token must apportion
itself among slots, so two slots cannot both claim the same screw.

Slot identity is not stable across images
-----------------------------------------
This is a property of the mechanism, not a bug to fix: slots are initialised
from a shared Gaussian and permuted arbitrarily by the competition, so slot 3
may be a screw in one image and the board in another.

**Decision for this project: option (ii), aggregate-only cardinality.**
Everything downstream (the hierarchical loss in 3b, the cardinality head in 3c)
consumes slots as an *unordered set*:

* the hierarchical loss compares slot sets with a permutation-invariant
  reduction, never slot *i* against slot *i*;
* the cardinality head predicts a **scalar total** per region, never a
  per-component-type breakdown.

Option (i) -- Hungarian matching against a canonical ordering learned from
normals -- was rejected because the canonical ordering would itself have to be
estimated from the training set, adding a second failure mode (a wrong
canonical assignment) that is invisible in the loss and would surface only as
degraded logical AUROC. If per-type counts are ever needed, that is the change
to make, and it needs its own validation gate.

Validation gate
---------------
Before this module is wired into the predictor, it must pass
``visualize.py --mode slots``: slot attention maps must segment the image into
plausible components. A bottleneck that is not routing sensibly cannot be
repaired by a downstream loss, and building on one means debugging two things
at once.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import init_vit_weights, trunc_normal_


class SlotAttention(nn.Module):
    """Iterative competitive attention that groups tokens into slots.

    Args:
        dim: slot width.
        input_dim: width of the incoming patch tokens.
        num_slots: how many slots compete. Pick this from the scene, not by
            guessing -- see :func:`suggest_num_slots`.
        iters: refinement iterations. 3 is the standard choice; more iterations
            sharpen the assignment but cost memory through the unrolled graph.
        hidden_dim: width of the post-update MLP.
        eps: attention denominator guard.
        use_gru: GRU update (the original formulation) or a residual MLP.
    """

    def __init__(
        self,
        dim: int = 256,
        input_dim: int | None = None,
        num_slots: int = 8,
        iters: int = 3,
        hidden_dim: int | None = None,
        eps: float = 1e-8,
        use_gru: bool = True,
    ):
        super().__init__()
        input_dim = input_dim or dim
        hidden_dim = hidden_dim or dim * 2

        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim**-0.5
        self.use_gru = use_gru

        # Slots are sampled per image from a learned Gaussian, which is what
        # keeps them exchangeable (and hence identity-unstable, see module doc).
        self.slots_mu = nn.Parameter(torch.zeros(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        trunc_normal_(self.slots_mu, std=0.02)
        trunc_normal_(self.slots_log_sigma, std=0.02)

        self.norm_input = nn.LayerNorm(input_dim, eps=1e-6)
        self.norm_slots = nn.LayerNorm(dim, eps=1e-6)
        self.norm_pre_ff = nn.LayerNorm(dim, eps=1e-6)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(input_dim, dim, bias=False)
        self.to_v = nn.Linear(input_dim, dim, bias=False)

        if use_gru:
            self.gru = nn.GRUCell(dim, dim)
        else:
            self.update = nn.Sequential(
                nn.Linear(dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
            )

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )

    # ------------------------------------------------------------------ #
    def init_slots(self, batch: int, device, generator=None) -> torch.Tensor:
        """Sample the initial slots from the learned Gaussian."""
        mu = self.slots_mu.expand(batch, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch, self.num_slots, -1)
        noise = torch.randn(
            batch, self.num_slots, self.dim, device=device, generator=generator
        )
        return mu + sigma * noise

    def forward(
        self,
        tokens: torch.Tensor,
        num_iters: int | None = None,
        return_attn: bool = False,
        slots_init: torch.Tensor | None = None,
    ):
        """(B, N, input_dim) -> (B, num_slots, dim).

        Args:
            tokens: patch tokens to be grouped.
            num_iters: override the configured iteration count.
            return_attn: also return the (B, num_slots, N) attention map, which
                is what ``visualize.py --mode slots`` renders.
            slots_init: (B, num_slots, dim) starting slots. Two token sets that
                must be compared (a composite and its reference, or a context
                and its full image) need the *same* starting noise, otherwise
                part of the difference between their slot sets is just the draw.
        """
        B, N, _ = tokens.shape
        iters = num_iters or self.iters

        tokens = self.norm_input(tokens)
        k = self.to_k(tokens)
        v = self.to_v(tokens)

        slots = self.init_slots(B, tokens.device) if slots_init is None else slots_init
        attn = None

        for i in range(iters):
            prev = slots
            q = self.to_q(self.norm_slots(slots))

            logits = torch.einsum("bsd,bnd->bsn", q, k) * self.scale

            # THE competitive step: softmax over slots, so tokens are shared out
            # among slots rather than each slot independently ranking tokens.
            attn = logits.softmax(dim=1)

            # Weighted mean over tokens, normalised per slot so an unused slot
            # does not collapse to zero.
            weights = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.einsum("bsn,bnd->bsd", weights, v)

            # The last iteration is detached from the earlier ones only if the
            # caller asks for it; here the full unrolled graph is kept so the
            # bottleneck trains end to end.
            if self.use_gru:
                slots = self.gru(
                    updates.reshape(-1, self.dim), prev.reshape(-1, self.dim)
                ).reshape(B, self.num_slots, self.dim)
            else:
                slots = prev + self.update(torch.cat([prev, updates], dim=-1))

            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return (slots, attn) if return_attn else slots

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_slots={self.num_slots}, iters={self.iters}"


class SlotBottleneck(nn.Module):
    """Slot attention plus the projections that make it a drop-in bottleneck.

    ``encode`` maps patch tokens to slots; ``broadcast`` maps slots back to
    per-patch features, which is what lets the module be trained and validated
    *alone* as patches -> slots -> reconstructed patches before anything
    depends on it.
    """

    def __init__(
        self,
        token_dim: int = 256,
        slot_dim: int = 256,
        num_slots: int = 8,
        iters: int = 3,
        hidden_dim: int | None = None,
        use_gru: bool = True,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.slot_dim = slot_dim
        self.num_slots = num_slots

        self.slot_attention = SlotAttention(
            dim=slot_dim, input_dim=token_dim, num_slots=num_slots,
            iters=iters, hidden_dim=hidden_dim, use_gru=use_gru,
        )

        # Broadcast decoder: every patch queries the slot set by position.
        self.slot_to_token = nn.Linear(slot_dim, token_dim)
        self.decoder_query = nn.Linear(token_dim, token_dim)
        self.decoder_out = nn.Sequential(
            nn.LayerNorm(token_dim, eps=1e-6),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        self.apply(init_vit_weights)

    # ------------------------------------------------------------------ #
    def encode(self, tokens: torch.Tensor, return_attn: bool = False,
               slots_init: torch.Tensor | None = None):
        """Patch tokens -> slots."""
        return self.slot_attention(tokens, return_attn=return_attn, slots_init=slots_init)

    def broadcast(self, slots: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        """Slots -> per-patch features, addressed by position.

        Args:
            slots: (B, S, slot_dim).
            pos_embed: (1, N, token_dim) or (B, N, token_dim) positional queries.

        Returns:
            (B, N, token_dim) reconstructed patch features.
        """
        B = slots.size(0)
        if pos_embed.size(0) == 1:
            pos_embed = pos_embed.expand(B, -1, -1)

        values = self.slot_to_token(slots)                     # (B, S, D)
        queries = self.decoder_query(pos_embed)                # (B, N, D)

        # Each patch reads from the slot set; softmax over slots keeps it a
        # convex combination, so a patch is explained by the slots, not summed.
        logits = torch.einsum("bnd,bsd->bns", queries, values) * (self.token_dim**-0.5)
        weights = logits.softmax(dim=-1)
        mixed = torch.einsum("bns,bsd->bnd", weights, values)

        return self.decoder_out(mixed + pos_embed)

    def forward(self, tokens: torch.Tensor, pos_embed: torch.Tensor,
                return_attn: bool = False, slots_init: torch.Tensor | None = None):
        """Full round trip, used by the isolation gate."""
        slots, attn = self.encode(tokens, return_attn=True, slots_init=slots_init)
        recon = self.broadcast(slots, pos_embed)
        return (recon, slots, attn) if return_attn else (recon, slots)

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        return (
            f"SlotBottleneck(slots={self.num_slots}, dim={self.slot_dim}, "
            f"iters={self.slot_attention.iters}, "
            f"{self.num_parameters()/1e6:.2f}M params)"
        )


# --------------------------------------------------------------------------- #
def suggest_num_slots(max_objects: int, margin: int = 2) -> int:
    """Slot count for a scene with ``max_objects`` components, plus background.

    Both directions of error are costly and asymmetric:

    * **too few** slots force two components to share one, which destroys the
      "an extra object appeared" signal the whole approach depends on;
    * **too many** split a single component across slots, which corrupts any
      count read off the set.

    For the synthetic ``screw_board`` (4 screws + product body + indicator
    strip = 6) this suggests 8.
    """
    return max_objects + margin


@torch.no_grad()
def slot_attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Mean entropy of each token's distribution over slots, in nats.

    The headline diagnostic for the isolation gate. Low entropy means tokens
    commit to one slot (clean segmentation); entropy near ``log(num_slots)``
    means every token is spread evenly across slots, i.e. the bottleneck has
    collapsed to an averaging layer and is not routing at all.
    """
    probs = attn.transpose(1, 2).clamp_min(1e-9)          # (B, N, S)
    return -(probs * probs.log()).sum(dim=-1).mean()


@torch.no_grad()
def slot_usage(attn: torch.Tensor) -> torch.Tensor:
    """Fraction of total attention mass each slot receives. (B, S).

    A slot with ~0 mass is dead; if most slots are dead the effective slot
    count is far below ``num_slots`` and the configured value is misleading.
    """
    mass = attn.sum(dim=-1)                                # (B, S)
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-9)


def chamfer_slot_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant distance between two slot sets. (B, S, D) x2 -> (B,).

    Each slot is matched to its nearest (cosine) slot in the other set, in both
    directions, and the matched distances are averaged over slots. Slot *i* is
    never compared with slot *i*: slot order is arbitrary (see module doc).
    Symmetric, so a slot that appears in one set but has no counterpart in the
    other is penalised from both sides.
    """
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    dist = 1.0 - torch.einsum("bsd,btd->bst", a_n, b_n)          # (B, S, T)
    a_to_b = dist.min(dim=2).values.mean(dim=1)
    b_to_a = dist.min(dim=1).values.mean(dim=1)
    return 0.5 * (a_to_b + b_to_a)


def sorted_usage(attn: torch.Tensor) -> torch.Tensor:
    """Slot usage sorted in descending order. (B, S, N) -> (B, S).

    Sorting is what makes the vector usable as a feature despite unstable slot
    identity: "the largest group takes 60%" means the same thing in every image,
    "slot 3 takes 60%" does not.
    """
    mass = attn.sum(dim=-1)
    usage = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return usage.sort(dim=-1, descending=True).values


def background_slot(attn: torch.Tensor) -> torch.Tensor:
    """Index of the slot holding the most attention mass per image. (B, S, N) -> (B,)."""
    return attn.sum(dim=-1).argmax(dim=-1)


def region_foreground_mass(attn: torch.Tensor, region_idx: torch.Tensor,
                           bg: torch.Tensor) -> torch.Tensor:
    """Share of a region's attention that is *not* on the background slot.

    The scalar the cardinality head predicts. Phase 3a found slots group by
    component *type* (all screws in one slot), so instances cannot be counted
    directly; but the non-background mass inside a region grows with how many
    components it holds and drops to ~0 when a component is missing.

    Args:
        attn: (B, S, N) slot attention over the full image.
        region_idx: (B, K) patch indices of the region.
        bg: (B,) background slot per image, from :func:`background_slot`.

    Returns:
        (B,) values in [0, 1].
    """
    S = attn.size(1)
    cols = torch.gather(attn, 2, region_idx.unsqueeze(1).expand(-1, S, -1))  # (B, S, K)
    region = cols.sum(dim=-1)                                                # (B, S)
    region = region / region.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return 1.0 - region.gather(1, bg.view(-1, 1)).squeeze(1)


def build_slot_bottleneck(cfg, token_dim: int):
    """Construct a :class:`SlotBottleneck` from the ``slots`` config node.

    Returns ``None`` when ``slots.enabled`` is false, so callers can treat the
    bottleneck as an optional stage without branching on config themselves.
    """
    node = dict(cfg or {})
    if not node.get("enabled", False):
        return None

    return SlotBottleneck(
        token_dim=token_dim,
        slot_dim=node.get("dim", token_dim),
        num_slots=node.get("num_slots", 8),
        iters=node.get("iters", 3),
        hidden_dim=node.get("hidden_dim", None),
        use_gru=node.get("use_gru", True),
    )
