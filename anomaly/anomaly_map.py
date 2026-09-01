"""Turning per-scale patch errors into a pixel-resolution anomaly heatmap.

The sweep produces one 16x16 error grid per window scale. Three things have to
happen before those become a usable heatmap:

1. **Per-scale normalisation.** A 6x6 window and a 2x2 window produce errors on
   different scales -- larger windows are intrinsically harder to predict, so
   their raw error is uniformly higher. Fusing them without normalisation would
   let the large scale dominate purely through offset, not through evidence.
2. **Fusion.** How the scales combine is a design choice with a direct effect on
   the structural-vs-logical trade-off, so it is configurable and ablated.
3. **Upsampling and smoothing.** A 16x16 grid becomes a 256x256 map by bilinear
   interpolation, then a Gaussian blur removes the patch-grid blocking that
   would otherwise cost pixel-level AUROC.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

FUSION_MODES = ("mean", "max", "weighted", "product")


def normalize_grid(
    grid: torch.Tensor,
    mode: str = "global",
    eps: float = 1e-6,
    stats: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Normalise a scale's grid. ``(B, H, W) -> (B, H, W)``.

    The choice here decides what the *image-level* score can measure, and the
    two families behave very differently:

    ``global`` (default)
        Subtract a mean and divide by a std estimated **once over normal
        images, separately for every patch position** (see
        ``AnomalyScorer.fit_scale_stats``). Scales become comparable while the
        between-image differences survive.

        Making the statistics *positional* rather than a single scalar is what
        actually makes image-level detection work here. Prediction difficulty is
        strongly position-dependent even on flawless images -- object boundaries
        and high-frequency regions are always harder to predict than flat
        background -- and that positional baseline is far larger than the
        anomaly signal. Dividing it out turns the question from "is this region
        hard to predict?" into "is this region *harder than it normally is
        here?*", which is the question that matters. On the development set this
        lifted image AUROC from 0.62 to 0.85, and logical AUROC from 0.67 to
        0.94.

    ``zscore`` / ``minmax`` / ``median``
        Per-image statistics. These make localisation crisp but are *fatal* for
        image-level detection: forcing every map to mean 0 / std 1 discards
        exactly the between-image difference the image score needs, and image
        AUROC collapses towards chance while pixel AUROC stays high. Kept for
        the ablation, and because that failure is itself worth demonstrating.

    Args:
        grid: (B, H, W) per-scale error grid.
        mode: normalisation family, see above.
        eps: division guard.
        stats: ``(mean, std)`` for ``global`` mode. Each may be a scalar or a
            ``(H, W)`` array of per-position statistics. Falls back to ``zscore``
            when the statistics have not been fitted yet.
    """
    if mode in ("none", None):
        return grid

    B = grid.size(0)

    if mode == "global":
        if stats is None:
            # Not calibrated yet -- degrade to per-image rather than crash, so a
            # freshly built scorer still produces a usable heatmap.
            mode = "zscore"
        else:
            mean, std = stats
            mean_t = torch.as_tensor(mean, dtype=grid.dtype, device=grid.device)
            std_t = torch.as_tensor(std, dtype=grid.dtype, device=grid.device)
            # Broadcast a (H, W) table over the batch; a scalar broadcasts too.
            if mean_t.ndim == 2:
                mean_t = mean_t.unsqueeze(0)
            if std_t.ndim == 2:
                std_t = std_t.unsqueeze(0)
            return (grid - mean_t) / std_t.clamp_min(eps)

    flat = grid.reshape(B, -1)

    if mode == "zscore":
        mu = flat.mean(dim=1, keepdim=True)
        sd = flat.std(dim=1, keepdim=True).clamp_min(eps)
        out = (flat - mu) / sd
    elif mode == "minmax":
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        out = (flat - lo) / (hi - lo).clamp_min(eps)
    elif mode == "median":
        # Robust to a few extreme patches, useful when a defect is tiny.
        med = flat.median(dim=1, keepdim=True).values
        mad = (flat - med).abs().median(dim=1, keepdim=True).values.clamp_min(eps)
        out = (flat - med) / mad
    else:
        raise ValueError(f"Unknown normalisation mode: {mode}")

    return out.reshape_as(grid)


def fuse_scales(
    grids: dict[int, torch.Tensor],
    mode: str = "mean",
    weights: dict[int, float] | None = None,
    normalize: str = "global",
    scale_stats: dict[int, tuple[float, float]] | None = None,
    deviation: str = "signed",
) -> torch.Tensor:
    """Combine per-scale grids into one. ``{w: (B, H, W)} -> (B, H, W)``.

    Args:
        grids: per-window-size error grids from ``LogicalJEPA.anomaly_grids``.
        mode: ``mean`` averages the evidence (stable, the default);
            ``max`` fires if *any* scale is surprised (most sensitive, noisiest);
            ``weighted`` uses ``weights`` to bias towards structural (small) or
            logical (large) scales;
            ``product`` requires agreement across scales (most conservative).
        weights: per-window weights for ``weighted`` mode.
        normalize: per-scale normalisation applied before fusion.
        scale_stats: ``{window: (mean, std)}`` fitted on normal images, required
            by ``normalize='global'``.
        deviation: how a normalised grid is turned into "surprise".

            ``signed`` (default)
                Only *harder than normal* counts. This is the classic
                assumption: an anomaly is something the model cannot predict.
            ``absolute``
                ``|z|`` -- both harder *and easier* than normal count.

            The second mode exists because of a measured failure. On real LOCO
            ``pushpins``, logical AUROC came out at 0.449 +/- 0.003 -- reliably
            *below chance*, meaning anomalous images were scoring systematically
            lower than normal ones. The mechanism: a **missing** component leaves
            an empty region, and empty regions are *easier* to predict than the
            object that belongs there. Signed scoring reads that as "extra
            normal" and pushes the image down the ranking.

            ``absolute`` treats an unexpectedly-easy region as equally
            suspicious, which is the right prior when the anomaly class includes
            missing objects.
    """
    if not grids:
        raise ValueError("fuse_scales received no grids")

    if deviation not in ("signed", "absolute"):
        raise ValueError(
            f"Unknown deviation mode '{deviation}'. Use 'signed' or 'absolute'."
        )

    scales = sorted(grids.keys())
    normalised = [
        normalize_grid(grids[w], normalize, stats=(scale_stats or {}).get(w))
        for w in scales
    ]

    if deviation == "absolute":
        # Applied per scale, before fusion: a scale that is unexpectedly easy
        # must register as surprise in its own right, not be averaged away
        # against another scale that happens to be hard.
        normalised = [n.abs() for n in normalised]

    stack = torch.stack(normalised, dim=0)

    if mode == "mean":
        return stack.mean(dim=0)
    if mode == "max":
        return stack.max(dim=0).values
    if mode == "product":
        # Shift to positive before multiplying so signs cannot cancel.
        shifted = stack - stack.amin(dim=(1, 2, 3), keepdim=True) + 1e-3
        return shifted.prod(dim=0).pow(1.0 / len(scales))
    if mode == "weighted":
        if not weights:
            raise ValueError("mode='weighted' requires a weights mapping")
        w = torch.tensor(
            [weights.get(s, 0.0) for s in scales],
            device=stack.device, dtype=stack.dtype,
        )
        w = w / w.sum().clamp_min(1e-6)
        return (stack * w.view(-1, 1, 1, 1)).sum(dim=0)

    raise ValueError(f"Unknown fusion mode '{mode}'. Available: {FUSION_MODES}")


def gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    """1-D Gaussian kernel truncated at 3 sigma."""
    radius = max(int(3.0 * sigma + 0.5), 1)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def gaussian_blur(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on (B, 1, H, W). Reflect padding avoids a dark border."""
    if sigma <= 0:
        return maps

    k = gaussian_kernel1d(sigma, maps.device, maps.dtype)
    radius = (k.numel() - 1) // 2

    out = F.pad(maps, (radius, radius, 0, 0), mode="reflect")
    out = F.conv2d(out, k.view(1, 1, 1, -1))
    out = F.pad(out, (0, 0, radius, radius), mode="reflect")
    out = F.conv2d(out, k.view(1, 1, -1, 1))
    return out


def upsample_map(
    grid: torch.Tensor,
    out_size: int = 256,
    sigma: float = 4.0,
    mode: str = "bilinear",
) -> torch.Tensor:
    """16x16 patch grid -> smooth (B, 1, out_size, out_size) pixel map."""
    if grid.dim() == 3:
        grid = grid.unsqueeze(1)

    maps = F.interpolate(
        grid, size=(out_size, out_size), mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )
    return gaussian_blur(maps, sigma)


def build_anomaly_map(
    grids: dict[int, torch.Tensor],
    out_size: int = 256,
    fusion: str = "mean",
    weights: dict[int, float] | None = None,
    normalize: str = "global",
    sigma: float = 4.0,
    scale_stats: dict[int, tuple[float, float]] | None = None,
    deviation: str = "signed",
) -> torch.Tensor:
    """Full grid-to-heatmap pipeline. Returns (B, 1, out_size, out_size)."""
    fused = fuse_scales(
        grids, mode=fusion, weights=weights,
        normalize=normalize, scale_stats=scale_stats, deviation=deviation,
    )
    return upsample_map(fused, out_size=out_size, sigma=sigma)


def to_display_map(anomaly_map: torch.Tensor, lo: float | None = None,
                   hi: float | None = None) -> np.ndarray:
    """Scale a single map to uint8 for visualisation. (1, H, W) or (H, W) -> (H, W).

    ``lo``/``hi`` pin the colour range to dataset-level percentiles so heatmaps
    from different images stay comparable; without them each map is scaled to
    its own range and a perfectly normal image looks as alarming as a defective
    one.
    """
    arr = anomaly_map.detach().squeeze().cpu().numpy().astype(np.float32)
    lo = float(arr.min()) if lo is None else float(lo)
    hi = float(arr.max()) if hi is None else float(hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
