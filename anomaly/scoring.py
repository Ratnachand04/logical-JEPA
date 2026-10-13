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
import torch.nn.functional as F

from .anomaly_map import fuse_scales, upsample_map

AGGREGATIONS = ("topk", "max", "mean", "topk_grid")
CARDINALITY_MODES = ("off", "add", "only")
MEMORY_MODES = ("off", "add", "only")


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
    # Same format as scale_stats, for the cardinality grids (Phase 3c).
    card_stats: dict = field(default_factory=dict)
    # Same format again, for the memory-bank grid (key 0).
    mem_stats: dict = field(default_factory=dict)

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
            "scale_stats": _stats_to_json(self.scale_stats),
            "card_stats": _stats_to_json(self.card_stats),
            "mem_stats": _stats_to_json(self.mem_stats),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Calibration":
        fields = {k: payload[k] for k in payload if k in cls.__annotations__}
        for key in ("scale_stats", "card_stats", "mem_stats"):
            if key in fields:
                fields[key] = _stats_from_json(fields[key])
        return cls(**fields)


def _stats_to_json(stats: dict) -> dict:
    return {str(k): [np.asarray(v[0]).tolist(), np.asarray(v[1]).tolist()]
            for k, v in (stats or {}).items()}


def _stats_from_json(payload: dict | None) -> dict:
    return {int(k): (np.asarray(v[0], dtype=np.float32), np.asarray(v[1], dtype=np.float32))
            for k, v in (payload or {}).items()}


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


class _MomentAccumulator:
    """Running per-position mean / std of ``{window: (B, H, W)}`` grids."""

    def __init__(self):
        self.sums: dict[int, torch.Tensor] = {}
        self.sq_sums: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = {}

    def add(self, grids: dict[int, torch.Tensor]) -> None:
        for window, grid in grids.items():
            g = grid.double()
            self.sums[window] = self.sums.get(window, 0.0) + g.sum(dim=0)
            self.sq_sums[window] = self.sq_sums.get(window, 0.0) + (g**2).sum(dim=0)
            self.counts[window] = self.counts.get(window, 0) + g.size(0)

    def finish(self, min_std: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for window in self.sums:
            n = max(self.counts[window], 1)
            mean = self.sums[window] / n
            var = (self.sq_sums[window] / n - mean**2).clamp_min(0.0)
            std = var.sqrt().clamp_min(min_std)

            if n < 8:
                # Too few normals for reliable per-position spread; fall back to
                # one pooled std so a small validation split cannot produce
                # near-zero divisors that manufacture false positives.
                std = torch.full_like(std, float(std.mean()))

            stats[window] = (mean.float().cpu().numpy(), std.float().cpu().numpy())
        return stats


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
        memory / memory_weight: the memory-bank channel (Study 11). The JEPA
            error asks "is this region what the context predicts?"; the memory
            bank asks "has a patch like this been seen on a normal part?". The
            bank holds the *teacher's* features of the images the statistics
            are fitted on -- in a few-shot run, exactly the K training images.
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
        cardinality: str = "off",
        cardinality_weight: float = 1.0,
        memory: str = "off",
        memory_weight: float = 1.0,
    ):
        if memory not in MEMORY_MODES:
            raise ValueError(f"Unknown memory mode '{memory}'. Available: {MEMORY_MODES}")
        if memory == "only" and cardinality != "off":
            raise ValueError("anomaly.memory=only scores the memory bank alone; "
                             "set anomaly.cardinality=off with it.")
        if cardinality not in CARDINALITY_MODES:
            raise ValueError(
                f"Unknown cardinality mode '{cardinality}'. Available: {CARDINALITY_MODES}"
            )
        if cardinality != "off" and not getattr(model, "has_cardinality", False):
            raise ValueError(
                f"anomaly.cardinality={cardinality} needs a checkpoint trained with a "
                f"cardinality head (slots.enabled=true)."
            )
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
        # off  -> the JEPA latent error alone (every previously reported number)
        # add  -> JEPA + cardinality_weight * cardinality, fused per patch
        # only -> the cardinality channel alone, to isolate what it contributes
        self.cardinality = cardinality
        self.cardinality_weight = cardinality_weight
        # off  -> no memory channel (every previously reported number)
        # add  -> JEPA + memory_weight * memory, fused per patch in z-units
        # only -> the memory bank alone, to isolate what it contributes
        self.memory = memory
        self.memory_weight = memory_weight
        self.memory_bank: torch.Tensor | None = None     # (M, D), unit-norm
        self.memory_owner: torch.Tensor | None = None    # (M,) index into memory_paths
        self.memory_paths: list[str] = []
        self.calibration = Calibration()

    @property
    def uses_cardinality(self) -> bool:
        return self.cardinality != "off"

    @property
    def uses_memory(self) -> bool:
        return self.memory != "off"

    # ------------------------------------------------------------------ #
    # Memory bank
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _teacher_features(self, images: torch.Tensor) -> torch.Tensor:
        """Unit-norm teacher embedding of every patch. (B, N, D)."""
        return F.normalize(self.model.encode_targets(images).float(), dim=-1)

    @torch.no_grad()
    def fit_memory_bank(self, loader, device) -> int:
        """Store the teacher features of every patch of every normal image.

        Fitted on the same loader as the positional statistics -- in a few-shot
        run, the K training images and nothing else. Returns the bank size.
        """
        self.model.eval()
        feats, owners, paths = [], [], []
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            f = self._teacher_features(images)
            for i in range(f.size(0)):
                owners.append(torch.full((f.size(1),), len(paths), dtype=torch.long))
                paths.append(str(batch["path"][i]))
            feats.append(f.reshape(-1, f.size(-1)))
        if not feats:
            raise ValueError("fit_memory_bank received no images")
        # Rounded to fp16 here, the precision it is saved at, so a bank loaded
        # by the demo scores exactly as the one evaluation used.
        self.memory_bank = torch.cat(feats).half().float()
        self.memory_owner = torch.cat(owners).to(self.memory_bank.device)
        self.memory_paths = paths
        return int(self.memory_bank.size(0))

    @torch.no_grad()
    def _memory_grid(self, images: torch.Tensor, paths=None) -> torch.Tensor:
        """Distance from each patch to its nearest bank patch. (B, H, W).

        An image that is itself in the bank -- only ever the case while fitting
        the statistics -- is compared with the *other* bank images, otherwise
        its distance would be exactly zero and the statistics meaningless.
        """
        if self.memory_bank is None:
            raise RuntimeError("anomaly.memory is on but no memory bank was fitted "
                               "or loaded; run calibrate() or load_memory_bank().")
        feats = self._teacher_features(images)
        bank = self.memory_bank.to(feats.device)
        owner = self.memory_owner.to(feats.device)
        index = {p: i for i, p in enumerate(self.memory_paths)}
        grid = self.model.grid_size

        out = []
        for i in range(feats.size(0)):
            dist = torch.cdist(feats[i], bank)                      # (N, M)
            own = index.get(str(paths[i])) if paths is not None else None
            if own is not None:
                dist[:, owner == own] = float("inf")
            out.append(dist.min(dim=1).values.reshape(grid, grid))
        return torch.stack(out)

    def memory_state(self) -> dict:
        """What :meth:`load_memory_bank` needs, for saving beside a checkpoint."""
        return {"features": self.memory_bank.half().cpu(),
                "owner": self.memory_owner.cpu(),
                "paths": list(self.memory_paths)}

    def load_memory_bank(self, state: dict, device=None) -> None:
        self.memory_bank = state["features"].float().to(device or "cpu")
        self.memory_owner = state["owner"].long().to(device or "cpu")
        self.memory_paths = list(state["paths"])

    def _grids(self, images: torch.Tensor):
        """``(jepa_grids, card_grids_or_None)`` for a batch."""
        kwargs = {"with_cardinality": True} if self.uses_cardinality else {}
        out = self.model.anomaly_grids(
            images, self.mask_bank, chunk=self.chunk,
            distance=self.distance, alpha=self.alpha, **kwargs,
        )
        return out if self.uses_cardinality else (out, None)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score_batch(self, images: torch.Tensor, return_grids: bool = False,
                    paths=None) -> dict:
        """Score a batch of images.

        Returns a dict with ``scores`` (B,), ``maps`` (B, 1, H, W) and -- when
        ``return_grids`` -- the raw per-scale 16x16 grids, which the
        visualisation script uses to show what each scale contributed.
        ``paths`` lets an image that is itself in the memory bank be compared
        with the rest of the bank only.
        """
        self.model.eval()

        # memory=only needs no sweep at all, which is most of the cost.
        grids, card_grids = ({}, None) if self.memory == "only" else self._grids(images)
        mem_grid = self._memory_grid(images, paths) if self.uses_memory else None

        def _maps(normalize: str) -> torch.Tensor:
            fused = None
            if self.cardinality != "only" and self.memory != "only":
                fused = fuse_scales(
                    grids, mode=self.fusion, weights=self.weights,
                    normalize=normalize, scale_stats=self.calibration.scale_stats,
                    deviation=self.deviation,
                )
            if card_grids is not None:
                # The cardinality error is |predicted - actual|, already
                # direction-free, so it is always read as signed: "a bigger
                # mismatch than normally occurs here".
                card = fuse_scales(
                    card_grids, mode="mean", normalize=normalize,
                    scale_stats=self.calibration.card_stats, deviation="signed",
                )
                fused = card if fused is None else fused + self.cardinality_weight * card
            if mem_grid is not None:
                # A nearest-neighbour distance has no "too easy" direction.
                mem = fuse_scales(
                    {0: mem_grid}, mode="mean", normalize=normalize,
                    scale_stats=self.calibration.mem_stats, deviation="signed",
                )
                fused = mem if fused is None else fused + self.memory_weight * mem
            return upsample_map(fused, out_size=self.out_size, sigma=self.sigma)

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
            if card_grids is not None:
                out["card_grids"] = card_grids
            if mem_grid is not None:
                out["memory_grid"] = mem_grid
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

        scores, labels, defects, paths, maps, loc_maps, masks = [], [], [], [], [], [], []

        for batch in iterator:
            images = batch["image"].to(device, non_blocking=True)
            out = self.score_batch(images, paths=batch.get("path"))

            scores.append(out["scores"].float().cpu().numpy())
            labels.append(batch["label"].numpy())
            defects.extend(batch["defect_type"])
            paths.extend(batch.get("path", []))

            if collect_maps:
                maps.append(out["detection_maps"].float().cpu().numpy())
                loc_maps.append(out["localization_maps"].float().cpu().numpy())
                if "mask" in batch:
                    masks.append(batch["mask"].float().cpu().numpy())

        result = {
            "scores": np.concatenate(scores) if scores else np.array([]),
            "labels": np.concatenate(labels) if labels else np.array([]),
            "defect_types": np.array(defects),
            "paths": list(paths),
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
        jepa_acc = _MomentAccumulator()
        card_acc = _MomentAccumulator()
        mem_acc = _MomentAccumulator()

        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            if self.memory != "only":
                grids, card_grids = self._grids(images)
                jepa_acc.add(grids)
                if card_grids is not None:
                    card_acc.add(card_grids)
            if self.uses_memory:
                # Bank images are compared with the OTHER bank images
                # (leave-one-image-out), so the spread is that of a new image.
                mem_acc.add({0: self._memory_grid(images, batch.get("path"))})

        # Stored on the instance rather than returned, so the return type (and
        # every existing caller) is unchanged when cardinality is off.
        self._fitted_card_stats = card_acc.finish(min_std)
        self._fitted_mem_stats = mem_acc.finish(min_std)
        return jepa_acc.finish(min_std)

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

        With ``memory`` on, the memory bank is built from ``stats_loader`` too:
        the same normal images the positional statistics come from.
        """
        if self.uses_memory:
            self.fit_memory_bank(stats_loader or loader, device)
        scale_stats = self.fit_scale_stats(stats_loader or loader, device)
        card_stats = getattr(self, "_fitted_card_stats", {})
        mem_stats = getattr(self, "_fitted_mem_stats", {})
        self.calibration.scale_stats = scale_stats
        self.calibration.card_stats = card_stats
        self.calibration.mem_stats = mem_stats

        out = self.score_loader(loader, device, collect_maps=True)
        self.calibration = fit_calibration(
            out["scores"], out.get("maps"), sigma_threshold=sigma_threshold
        )
        self.calibration.scale_stats = scale_stats
        self.calibration.card_stats = card_stats
        self.calibration.mem_stats = mem_stats
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
            "card_grids": out.get("card_grids"),
            "memory_grid": out.get("memory_grid"),
        }


def _cardinality_mode(value) -> str:
    """YAML reads a bare ``off`` as ``False`` (and ``on`` as ``True``)."""
    if value is False or value is None:
        return "off"
    if value is True:
        return "add"
    return str(value)


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
        cardinality=_cardinality_mode(node.get("cardinality", "off")),
        cardinality_weight=float(node.get("cardinality_weight", 1.0)),
        # Same YAML off/on pitfall as cardinality, so the same parser.
        memory=_cardinality_mode(node.get("memory", "off")),
        memory_weight=float(node.get("memory_weight", 1.0)),
    )
