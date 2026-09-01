"""Image-level scoring and score calibration.

Two separate problems live here.

**Aggregation.** A heatmap must collapse to one number per image. Plain ``max``
is the obvious choice and the wrong one: it is decided by a single pixel, so one
noisy patch on a normal image produces a false alarm. ``topk`` -- the mean of the
highest-scoring fraction of the map -- is the default because it still responds
to small defects while requiring a *region* rather than a pixel to be surprising.

**Calibration.** Raw scores are not comparable across categories or ablation
arms, and a demo needs to say "normal" or "anomalous", which requires a
threshold. Both are fitted on held-out *normal* images only, so the unsupervised
setting is preserved: no anomaly ever informs the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .anomaly_map import build_anomaly_map

AGGREGATIONS = ("topk", "max", "mean", "topk_grid")


def aggregate_score(
    anomaly_map: torch.Tensor,
    mode: str = "topk",
    top_k_ratio: float = 0.01,
) -> torch.Tensor:
    """Collapse (B, 1, H, W) maps to (B,) image scores.

    Args:
        anomaly_map: pixel-resolution maps.
        mode: ``topk`` (mean of the hottest ``top_k_ratio`` of pixels), ``max``,
            or ``mean``.
        top_k_ratio: fraction of pixels used by ``topk``. 1% of a 256x256 map is
            655 pixels, roughly a 25x25 region -- about the size of the smallest
            defect worth flagging.
    """
    B = anomaly_map.size(0)
    flat = anomaly_map.reshape(B, -1)

    if mode == "max":
        return flat.max(dim=1).values
    if mode == "mean":
        return flat.mean(dim=1)
    if mode in ("topk", "topk_grid"):
        k = max(int(flat.size(1) * top_k_ratio), 1)
        return flat.topk(k, dim=1).values.mean(dim=1)

    raise ValueError(f"Unknown aggregation '{mode}'. Available: {AGGREGATIONS}")


@dataclass
class Calibration:
    """Normalisation and decision threshold fitted on normal images only.

    Attributes:
        score_mean/score_std: statistics of image scores over normal images.
        threshold: decision boundary in *raw* score units.
        map_lo/map_hi: percentiles of the pixel maps, used to pin the heatmap
            colour range so images are visually comparable.
        n_samples: how many normal images the fit used.
        scale_stats: ``{window: (mean, std)}`` of the raw per-scale error grids
            over normal images. These make the sweep scales comparable *without*
            per-image normalisation, which is what keeps image-level scores
            comparable between images.
    """

    score_mean: float = 0.0
    score_std: float = 1.0
    threshold: float = 0.0
    map_lo: float = 0.0
    map_hi: float = 1.0
    n_samples: int = 0
    scale_stats: dict = field(default_factory=dict)

    def normalize(self, scores: np.ndarray) -> np.ndarray:
        """Express scores as standard deviations above the normal mean."""
        return (np.asarray(scores) - self.score_mean) / max(self.score_std, 1e-6)

    def is_anomalous(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores) > self.threshold

    def confidence(self, score: float) -> float:
        """Map a raw score to [0, 1] via a logistic centred on the threshold.

        Purely a presentation device for the web demo -- it is a monotone
        transform of the score and changes no ranking, so it cannot affect any
        reported metric.
        """
        z = (float(score) - self.threshold) / max(self.score_std, 1e-6)
        return float(1.0 / (1.0 + np.exp(-z)))

    def to_dict(self) -> dict:
        return {
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "threshold": self.threshold,
            "map_lo": self.map_lo,
            "map_hi": self.map_hi,
            "n_samples": self.n_samples,
            # JSON keys must be strings and numpy grids must become nested
            # lists; from_dict reverses both.
            "scale_stats": {
                str(k): [np.asarray(v[0]).tolist(), np.asarray(v[1]).tolist()]
                for k, v in self.scale_stats.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Calibration":
        fields = {k: payload[k] for k in payload if k in cls.__annotations__}
        if "scale_stats" in fields:
            fields["scale_stats"] = {
                int(k): (np.asarray(v[0], dtype=np.float32),
                         np.asarray(v[1], dtype=np.float32))
                for k, v in (fields["scale_stats"] or {}).items()
            }
        return cls(**fields)


def fit_calibration(
    scores: np.ndarray,
    maps: np.ndarray | None = None,
    sigma_threshold: float = 3.0,
    map_percentiles: tuple[float, float] = (1.0, 99.5),
) -> Calibration:
    """Fit :class:`Calibration` from normal-image scores.

    The threshold is ``mean + sigma_threshold * std``. At 3 sigma a Gaussian
    score distribution yields roughly a 0.1% false-alarm rate, which is the
    right default for industrial inspection where stopping a line costs more
    than a missed borderline part. The value is exposed in the config so the
    trade-off can be moved.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("fit_calibration received no scores")

    mean = float(scores.mean())
    std = float(scores.std()) if scores.size > 1 else 1.0
    std = max(std, 1e-6)

    calib = Calibration(
        score_mean=mean,
        score_std=std,
        threshold=mean + sigma_threshold * std,
        n_samples=int(scores.size),
    )

    if maps is not None and np.size(maps) > 0:
        calib.map_lo = float(np.percentile(maps, map_percentiles[0]))
        calib.map_hi = float(np.percentile(maps, map_percentiles[1]))

    return calib


class AnomalyScorer:
    """Runs the masked sweep and produces heatmaps plus image scores.

    This is the object the evaluation script and the web server both use, so
    the demo and the reported numbers cannot drift apart.

    Args:
        model: a trained :class:`models.LogicalJEPA`.
        mask_bank: a :class:`masking.SweepMaskBank`.
        fusion / normalize / sigma / weights: heatmap construction settings.
        aggregation / top_k_ratio: image-score settings.
        distance: override the distance used for scoring (scoring ablation).
        chunk: sweep positions evaluated per forward pass; lower it if VRAM is
            tight.
    """

    def __init__(
        self,
        model,
        mask_bank,
        out_size: int = 256,
        fusion: str = "mean",
        normalize: str = "global",
        normalize_localization: str | None = None,
        deviation: str = "signed",
        sigma: float = 4.0,
        weights: dict[int, float] | None = None,
        aggregation: str = "topk",
        top_k_ratio: float = 0.01,
        distance: str | None = None,
        alpha: float | None = None,
        chunk: int = 16,
    ):
        self.model = model
        self.mask_bank = mask_bank
        self.out_size = out_size
        self.fusion = fusion
        # `normalize` remains the detection setting so older callers and saved
        # configs keep working; localisation defaults to it unless overridden.
        self.normalize = normalize
        self.normalize_detection = normalize
        self.normalize_localization = normalize_localization or normalize
        # signed = only harder-than-normal counts; absolute = |z|, so an
        # unexpectedly *easy* region (a missing object) also counts.
        self.deviation = deviation
        self.sigma = sigma
        self.weights = weights
        self.aggregation = aggregation
        self.top_k_ratio = top_k_ratio
        self.distance = distance
        self.alpha = alpha
        self.chunk = chunk
        self.calibration = Calibration()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score_batch(self, images: torch.Tensor, return_grids: bool = False) -> dict:
        """Score a batch of images.

        Returns a dict with ``scores`` (B,), ``maps`` (B, 1, H, W) and -- when
        ``return_grids`` -- the raw per-scale 16x16 grids, which the
        visualisation script uses to show what each scale contributed.
        """
        self.model.eval()

        grids = self.model.anomaly_grids(
            images, self.mask_bank, chunk=self.chunk,
            distance=self.distance, alpha=self.alpha,
        )

        def _maps(normalize: str) -> torch.Tensor:
            return build_anomaly_map(
                grids, out_size=self.out_size, fusion=self.fusion,
                weights=self.weights, normalize=normalize, sigma=self.sigma,
                scale_stats=self.calibration.scale_stats,
                deviation=self.deviation,
            )

        # Study 4 measured that no single normalisation is best at both jobs,
        # so the two are computed separately rather than compromised into one.
        detection_maps = _maps(self.normalize_detection)
        scores = aggregate_score(detection_maps, self.aggregation, self.top_k_ratio)

        if self.normalize_localization == self.normalize_detection:
            localization_maps = detection_maps          # skip the duplicate work
        else:
            localization_maps = _maps(self.normalize_localization)

        out = {
            "scores": scores,
            # `maps` stays the detection map so existing callers are unchanged.
            "maps": detection_maps,
            "detection_maps": detection_maps,
            "localization_maps": localization_maps,
        }
        if return_grids:
            out["grids"] = grids
        return out

    @torch.no_grad()
    def score_loader(self, loader, device, collect_maps: bool = True,
                     progress: bool = False) -> dict:
        """Score every image in a dataloader.

        Returns numpy arrays for ``scores``, ``labels``, ``defect_types`` and,
        when requested, ``maps`` and ``masks`` for pixel-level metrics.
        """
        iterator = loader
        if progress:
            from tqdm import tqdm
            iterator = tqdm(loader, desc="scoring", leave=False)

        scores, labels, defects, maps, loc_maps, masks = [], [], [], [], [], []

        for batch in iterator:
            images = batch["image"].to(device, non_blocking=True)
            out = self.score_batch(images)

            scores.append(out["scores"].float().cpu().numpy())
            labels.append(batch["label"].numpy())
            defects.extend(batch["defect_type"])

            if collect_maps:
                maps.append(out["detection_maps"].float().cpu().numpy())
                loc_maps.append(out["localization_maps"].float().cpu().numpy())
                if "mask" in batch:
                    masks.append(batch["mask"].float().cpu().numpy())

        result = {
            "scores": np.concatenate(scores) if scores else np.array([]),
            "labels": np.concatenate(labels) if labels else np.array([]),
            "defect_types": np.array(defects),
        }
        if collect_maps and maps:
            result["maps"] = np.concatenate(maps)
            result["localization_maps"] = np.concatenate(loc_maps)
            if masks:
                result["masks"] = np.concatenate(masks)

        return result

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def fit_scale_stats(self, loader, device, min_std: float = 1e-4) -> dict:
        """Estimate the per-position ``(mean, std)`` of every sweep scale.

        Fitted on normal images only. Two effects are removed at once:

        * larger windows are intrinsically harder to predict, so their raw error
          sits at a higher level than small windows;
        * within any scale, error varies strongly *by position* -- object edges
          and detailed regions are always harder than flat background, even on
          a flawless part.

        The second effect dominates, and leaving it in is what keeps image-level
        AUROC near chance while pixel-level metrics look fine. See
        :func:`anomaly.anomaly_map.normalize_grid` for the measured impact.

        Returns:
            ``{window: (mean_grid, std_grid)}`` with grids of shape (H, W).
        """
        self.model.eval()
        sums: dict[int, torch.Tensor] = {}
        sq_sums: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = {}

        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            grids = self.model.anomaly_grids(
                images, self.mask_bank, chunk=self.chunk,
                distance=self.distance, alpha=self.alpha,
            )
            for window, grid in grids.items():
                g = grid.double()
                sums[window] = sums.get(window, 0.0) + g.sum(dim=0)
                sq_sums[window] = sq_sums.get(window, 0.0) + (g**2).sum(dim=0)
                counts[window] = counts.get(window, 0) + g.size(0)

        stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for window in sums:
            n = max(counts[window], 1)
            mean = sums[window] / n
            var = (sq_sums[window] / n - mean**2).clamp_min(0.0)
            std = var.sqrt().clamp_min(min_std)

            if n < 8:
                # Too few normals for reliable per-position spread; fall back to
                # one pooled std so a small validation split cannot produce
                # near-zero divisors that manufacture false positives.
                std = torch.full_like(std, float(std.mean()))

            stats[window] = (mean.float().cpu().numpy(), std.float().cpu().numpy())

        return stats

    @torch.no_grad()
    def calibrate(self, loader, device, sigma_threshold: float = 3.0,
                  stats_loader=None) -> Calibration:
        """Fit the calibration on loaders of **normal images only**.

        Two passes: the first estimates the per-position statistics used to
        normalise each scale, the second measures the resulting image scores to
        place the threshold. The order matters -- the scores must be produced by
        the same normalisation inference will use.

        Args:
            loader: normal images used to fit the score threshold.
            device: compute device.
            sigma_threshold: threshold placement, in standard deviations.
            stats_loader: optional, usually the (larger) normal *training* set.
                The positional statistics are a per-patch mean and standard
                deviation, so they need substantially more images than the
                threshold does; a 20-image validation split gives noisy
                per-position spreads. Defaults to ``loader``.

        Passing a loader containing anomalies would leak label information into
        the threshold and invalidate the unsupervised claim, so the caller must
        hand over ``validation/good`` or ``train/good``.
        """
        scale_stats = self.fit_scale_stats(stats_loader or loader, device)
        self.calibration.scale_stats = scale_stats

        out = self.score_loader(loader, device, collect_maps=True)
        self.calibration = fit_calibration(
            out["scores"], out.get("maps"), sigma_threshold=sigma_threshold
        )
        self.calibration.scale_stats = scale_stats
        return self.calibration

    def predict(self, images: torch.Tensor) -> dict:
        """Single-batch inference for the demo: verdict, score, confidence, map."""
        out = self.score_batch(images, return_grids=True)
        raw = out["scores"].float().cpu().numpy()

        return {
            "score": raw,
            "z_score": self.calibration.normalize(raw),
            "is_anomalous": self.calibration.is_anomalous(raw),
            "confidence": np.array([self.calibration.confidence(s) for s in raw]),
            "maps": out["maps"],
            "grids": out["grids"],
        }


def build_scorer(cfg, model, mask_bank) -> AnomalyScorer:
    """Construct an :class:`AnomalyScorer` from the ``anomaly`` config node."""
    node = cfg.get("anomaly", {})
    weights = node.get("scale_weights", None)
    if weights:
        weights = {int(k): float(v) for k, v in dict(weights).items()}

    return AnomalyScorer(
        model=model,
        mask_bank=mask_bank,
        out_size=cfg.get_path("data.img_size", 256),
        fusion=node.get("fusion", "mean"),
        normalize=node.get("normalize_detection", node.get("normalize", "global")),
        normalize_localization=node.get("normalize_localization", None),
        deviation=node.get("deviation", "signed"),
        sigma=node.get("smooth_sigma", 4.0),
        weights=weights,
        aggregation=node.get("aggregation", "topk"),
        top_k_ratio=node.get("top_k_ratio", 0.01),
        distance=node.get("distance", None),
        alpha=node.get("alpha", None),
        chunk=node.get("sweep_chunk", 16),
    )
