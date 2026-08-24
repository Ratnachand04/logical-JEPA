"""PatchCore baseline -- nominal patch memory bank with coreset subsampling.

PatchCore builds a memory bank of patch features from normal images and scores a
test patch by its distance to the nearest bank entry. It is the natural
comparison for this project: it is the strongest well-established industrial
anomaly detector, and its failure mode on *logical* anomalies is structural
rather than incidental.

    A memory bank of patch features has no notion of where a patch was or how
    many like it there were. A correctly-manufactured screw photographed in the
    wrong position produces a patch feature that is already in the bank, so its
    nearest-neighbour distance is small and PatchCore calls it normal.

**One deliberate deviation from the paper.** The original uses a
WideResNet-50 pretrained on ImageNet. This project forbids pretrained
backbones, so features come from the *same* from-scratch target encoder the
JEPA model trained. This is the fair comparison for the research question --
both methods then see identical features and differ only in how they use them
(memory lookup vs contextual prediction) -- but it means the absolute numbers
are not comparable to published PatchCore results, which is stated in the
README and in the results table.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_patch_features(model, loader, device, progress: bool = False
                           ) -> tuple[torch.Tensor, int]:
    """Collect L2-normalised patch embeddings from every image in a loader.

    Returns:
        ``(features, grid_size)`` where features is ``(N*num_patches, D)``.
    """
    iterator = loader
    if progress:
        from tqdm import tqdm
        iterator = tqdm(loader, desc="patch features", leave=False)

    chunks = []
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        tokens = model.encode_targets(images)                  # (B, N, D)
        tokens = F.normalize(tokens, dim=-1)
        chunks.append(tokens.reshape(-1, tokens.size(-1)).cpu())

    return torch.cat(chunks, dim=0), model.grid_size


def greedy_coreset(features: torch.Tensor, ratio: float = 0.01,
                   seed: int = 0, device=None, max_iters: int | None = None
                   ) -> torch.Tensor:
    """Greedy k-center coreset selection.

    Repeatedly picks the point farthest from everything already selected, which
    keeps the bank's *coverage* of the feature manifold while shrinking it by
    ~100x. This is PatchCore's key efficiency trick; without it the bank holds
    every patch of every training image and nearest-neighbour search dominates
    inference time.

    Args:
        features: (N, D) L2-normalised features.
        ratio: fraction of points to keep.
        seed: RNG seed for the starting point.
        device: where to run the distance updates.

    Returns:
        Long tensor of selected row indices.
    """
    device = device or features.device
    feats = features.to(device)
    n = feats.size(0)

    budget = max(int(n * ratio), 1)
    if max_iters is not None:
        budget = min(budget, max_iters)
    if budget >= n:
        return torch.arange(n)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = int(torch.randint(n, (1,), generator=generator).item())

    selected = [start]
    # Distance from every point to the nearest selected centre, updated
    # incrementally so each iteration costs one (N, D) x (D,) product.
    min_dist = torch.cdist(feats, feats[start : start + 1]).squeeze(1)

    for _ in range(budget - 1):
        nxt = int(torch.argmax(min_dist).item())
        selected.append(nxt)
        dist = torch.cdist(feats, feats[nxt : nxt + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, dist)

    return torch.tensor(selected, dtype=torch.long)


class PatchCore:
    """Memory-bank anomaly detector over from-scratch patch features.

    Args:
        model: a trained :class:`models.LogicalJEPA`, used only as a frozen
            feature extractor via its target encoder.
        coreset_ratio: fraction of training patches kept in the bank.
        n_neighbors: neighbours averaged for the patch distance. >1 smooths the
            score and reduces sensitivity to a single outlier bank entry.
        sigma: Gaussian blur applied to the anomaly map, matching the JEPA path.
    """

    def __init__(self, model, coreset_ratio: float = 0.01, n_neighbors: int = 3,
                 sigma: float = 4.0, aggregation: str = "topk",
                 top_k_ratio: float = 0.01, seed: int = 0):
        self.model = model
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.aggregation = aggregation
        self.top_k_ratio = top_k_ratio
        self.seed = seed

        self.memory_bank: torch.Tensor | None = None
        self.grid_size = model.grid_size

        from anomaly.scoring import Calibration
        self.calibration = Calibration()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def fit(self, train_loader, device, logger=None) -> "PatchCore":
        """Build the memory bank from normal training images."""
        features, self.grid_size = extract_patch_features(
            self.model, train_loader, device, progress=True
        )
        if logger:
            logger.info(f"PatchCore: {features.size(0)} patch features collected")

        indices = greedy_coreset(features, self.coreset_ratio, self.seed, device=device)
        self.memory_bank = features[indices].to(device)

        if logger:
            logger.info(
                f"PatchCore: coreset {self.memory_bank.size(0)} / {features.size(0)} "
                f"({self.coreset_ratio:.1%})"
            )
        return self

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score_batch(self, images: torch.Tensor) -> dict:
        from anomaly.anomaly_map import gaussian_blur, upsample_map
        from anomaly.scoring import aggregate_score

        if self.memory_bank is None:
            raise RuntimeError("PatchCore.fit() must be called before scoring")

        tokens = self.model.encode_targets(images)
        tokens = F.normalize(tokens, dim=-1)                    # (B, N, D)
        B, N, D = tokens.shape

        dist = torch.cdist(tokens.reshape(B * N, D), self.memory_bank)   # (B*N, M)
        k = min(self.n_neighbors, dist.size(1))
        patch_scores = dist.topk(k, dim=1, largest=False).values.mean(dim=1)

        grid = patch_scores.reshape(B, self.grid_size, self.grid_size)
        maps = upsample_map(grid, out_size=images.size(-1), sigma=self.sigma)
        scores = aggregate_score(maps, self.aggregation, self.top_k_ratio)

        return {"scores": scores, "maps": maps, "grid": grid}

    @torch.no_grad()
    def score_loader(self, loader, device, collect_maps: bool = True,
                     progress: bool = False) -> dict:
        iterator = loader
        if progress:
            from tqdm import tqdm
            iterator = tqdm(loader, desc="PatchCore scoring", leave=False)

        scores, labels, defects, maps, masks = [], [], [], [], []
        for batch in iterator:
            images = batch["image"].to(device, non_blocking=True)
            out = self.score_batch(images)

            scores.append(out["scores"].float().cpu().numpy())
            labels.append(batch["label"].numpy())
            defects.extend(batch["defect_type"])
            if collect_maps:
                maps.append(out["maps"].float().cpu().numpy())
                if "mask" in batch:
                    masks.append(batch["mask"].float().cpu().numpy())

        result = {
            "scores": np.concatenate(scores) if scores else np.array([]),
            "labels": np.concatenate(labels) if labels else np.array([]),
            "defect_types": np.array(defects),
        }
        if collect_maps and maps:
            result["maps"] = np.concatenate(maps)
            if masks:
                result["masks"] = np.concatenate(masks)
        return result

    @torch.no_grad()
    def calibrate(self, loader, device, sigma_threshold: float = 3.0):
        from anomaly.scoring import fit_calibration

        out = self.score_loader(loader, device, collect_maps=True)
        self.calibration = fit_calibration(
            out["scores"], out.get("maps"), sigma_threshold=sigma_threshold
        )
        return self.calibration

    def describe(self) -> str:
        size = 0 if self.memory_bank is None else self.memory_bank.size(0)
        return f"PatchCore(bank={size}, k={self.n_neighbors}, ratio={self.coreset_ratio})"
