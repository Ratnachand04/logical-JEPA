"""Score a current image against reference images of a correct part.

Pipeline for one comparison:

1. **Align** the current photo onto the first reference (:mod:`anomaly.alignment`).
2. **Two channels** on the aligned image:
   ``jepa``   prediction error of the reference-conditioned JEPA, swept over the
              grid (:meth:`ReferenceJEPA.error_grids`), averaged over scales;
   ``memory`` distance of each patch to its nearest reference patch.
3. **Remove the normal baseline.** With one reference there are no normal
   statistics, so the reference is perturbed ``n_pseudo`` times (small affine
   and lighting changes, the variation between two photos of the same part),
   each perturbation is aligned and scored exactly like a real current image,
   and the per-position mean of those pseudo-normal grids is subtracted. With
   several references, each reference is also scored against the others, which
   adds real instance-to-instance variation.
4. **Common units.** Residuals are divided by global scales, and fused with
   global weights, both fitted once on *training* products
   (``scripts/eval_reference.py --calibrate``); the decision threshold is fitted
   the same way. Nothing is fitted on the product being inspected, which is
   what lets the model work on a product it has never seen.
5. Upsample, smooth, and take the mean of the top 1% as the image score.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image

from masking.sweep_mask import SweepMaskBank

from .alignment import align_to_reference
from .anomaly_map import upsample_map
from .scoring import aggregate_score

DEFAULT_CALIBRATION = {"jepa_scale": 1.0, "memory_scale": 1.0, "jepa_weight": 1.0,
                       "memory_weight": 1.0, "threshold": float("inf"), "rule": "ratio",
                       "fitted_on": []}


def load_calibration(checkpoint: str) -> dict:
    """Read ``reference_calibration.json`` beside a checkpoint, if present."""
    path = os.path.join(os.path.dirname(checkpoint), "reference_calibration.json")
    calib = dict(DEFAULT_CALIBRATION)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as h:
            calib.update(json.load(h))
    return calib


def to_array(image, size: int = 256) -> np.ndarray:
    """PIL image or array -> HxWx3 uint8 RGB at ``size`` x ``size``."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return np.array(image.convert("RGB").resize((size, size), Image.BILINEAR))


@dataclass
class ReferenceState:
    """Everything prepared once per set of references."""

    images: list[np.ndarray]                 # aligned-space reference arrays
    tokens: torch.Tensor                     # (1, R*N, D)
    jepa_baseline: torch.Tensor              # (G, G) mean pseudo-normal JEPA error
    memory_baseline: torch.Tensor            # (G, G) mean pseudo-normal memory distance
    pseudo_jepa: list[torch.Tensor] = field(default_factory=list)
    pseudo_memory: list[torch.Tensor] = field(default_factory=list)
    n_real: int = 0            # leading pseudo entries that are real second instances


class ReferenceScorer:
    """Compare current images with references using a :class:`ReferenceJEPA`."""

    def __init__(self, model, device, windows=(4,), out_size: int = 256, sigma: float = 4.0,
                 top_k_ratio: float = 0.01, n_pseudo: int = 8, align: bool = True,
                 calibration: dict | None = None, chunk: int = 16, seed: int = 0):
        self.model = model.eval()
        self.device = device
        self.bank = SweepMaskBank(model.grid_size, windows=tuple(windows), device=device)
        self.out_size = out_size
        self.sigma = sigma
        self.top_k_ratio = top_k_ratio
        self.n_pseudo = n_pseudo
        self.align = align
        self.chunk = chunk
        self.seed = seed
        self.calibration = dict(DEFAULT_CALIBRATION)
        self.calibration.update(calibration or {})
        self._normalize = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                                     T.Normalize([0.5] * 3, [0.5] * 3)])
        self._jitter = T.Compose([
            T.RandomAffine(degrees=3.0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ])

    # ------------------------------------------------------------------ #
    def _tensor(self, image: np.ndarray) -> torch.Tensor:
        return self._normalize(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _raw_grids(self, current: np.ndarray, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(jepa_grid, memory_grid), each (G, G), for an already aligned image."""
        x = self._tensor(current)
        grids = self.model.error_grids(x, tokens, self.bank, chunk=self.chunk)
        jepa = torch.stack(list(grids.values()), 0).mean(0)[0]
        memory = self.model.memory_grid(x, tokens)[0]
        return jepa.float(), memory.float()

    def _pseudo_normal(self, reference: np.ndarray, rng: random.Random) -> np.ndarray:
        torch.manual_seed(rng.randrange(1 << 30))
        jittered = np.asarray(self._jitter(Image.fromarray(reference)))
        # Fill the empty affine border from the reference, as alignment would.
        empty = jittered.sum(axis=2) == 0
        jittered = jittered.copy()
        jittered[empty] = reference[empty]
        return jittered

    def _align(self, current: np.ndarray, reference: np.ndarray):
        if not self.align:
            return current, np.ones(current.shape[:2], bool), {"aligned": False}
        return align_to_reference(current, reference)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def prepare(self, references) -> ReferenceState:
        """Encode the references and measure the normal baseline from them."""
        refs = [to_array(r, self.out_size) for r in references]
        if not refs:
            raise ValueError("at least one reference image is required")
        stack = torch.cat([self._tensor(r) for r in refs], 0).unsqueeze(0)    # (1, R, C, H, W)
        tokens = self.model.encode_references(stack)

        rng = random.Random(self.seed)
        pj, pm = [], []
        for i in range(max(self.n_pseudo, len(refs) if len(refs) > 1 else 0)):
            j = i % len(refs)
            if len(refs) > 1 and i < len(refs):
                # A real second instance: reference j scored against the others.
                others = [k for k in range(len(refs)) if k != j]
                sub = torch.cat([self._tensor(refs[k]) for k in others], 0).unsqueeze(0)
                sub_tokens = self.model.encode_references(sub)
                current, _, _ = self._align(refs[j], refs[others[0]])
                jg, mg = self._raw_grids(current, sub_tokens)
            else:
                current, _, _ = self._align(self._pseudo_normal(refs[j], rng), refs[0])
                jg, mg = self._raw_grids(current, tokens)
            pj.append(jg)
            pm.append(mg)
        return ReferenceState(images=refs, tokens=tokens,
                              jepa_baseline=torch.stack(pj).mean(0),
                              memory_baseline=torch.stack(pm).mean(0),
                              pseudo_jepa=pj, pseudo_memory=pm,
                              n_real=len(refs) if len(refs) > 1 else 0)

    @torch.no_grad()
    def residuals(self, current, state: ReferenceState) -> dict:
        """Align, compute both channels and subtract the normal baseline."""
        cur = to_array(current, self.out_size)
        aligned, valid, info = self._align(cur, state.images[0])
        jepa, memory = self._raw_grids(aligned, state.tokens)
        return {"aligned": aligned, "valid": valid, "align": info,
                "jepa": jepa - state.jepa_baseline, "memory": memory - state.memory_baseline,
                "jepa_raw": jepa, "memory_raw": memory}

    def fuse(self, res: dict, channels=("jepa", "memory")) -> torch.Tensor:
        """Residual grids -> one (G, G) grid in common units."""
        c = self.calibration
        fused = torch.zeros_like(res["jepa"])
        if "jepa" in channels:
            fused = fused + c["jepa_weight"] * res["jepa"] / c["jepa_scale"]
        if "memory" in channels:
            fused = fused + c["memory_weight"] * res["memory"] / c["memory_scale"]
        return fused

    def to_map(self, grid: torch.Tensor, valid: np.ndarray) -> torch.Tensor:
        amap = upsample_map(grid[None, None].to(self.device), out_size=self.out_size,
                            sigma=self.sigma)
        if valid is not None and not valid.all():
            # Out-of-view pixels were filled from the reference; never flag them.
            invalid = torch.from_numpy(~valid).to(amap.device)
            amap[0, 0][invalid] = amap.min()
        return amap

    def image_score(self, amap: torch.Tensor) -> float:
        return float(aggregate_score(amap, "topk", self.top_k_ratio)[0])

    def normal_level(self, state: ReferenceState) -> float:
        """Score a normal image of *this* product would get: the mean of the
        leave-one-out scores of the pseudo-normals (each scored against the
        baseline of the others). Products differ about twofold in how much two
        normal instances differ, so verdicts are taken relative to this level.

        With two or more references, only the real second instances (each
        reference scored against the others) enter the mean: perturbed copies
        of one photo understate how much two real parts differ."""
        key = (self.calibration["jepa_scale"], self.calibration["memory_scale"],
               self.calibration["jepa_weight"], self.calibration["memory_weight"])
        cached = getattr(state, "_level", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        n = len(state.pseudo_jepa)
        scores = []
        for i in range(n):
            rest = [j for j in range(n) if j != i]
            res = {"jepa": state.pseudo_jepa[i] - torch.stack([state.pseudo_jepa[j] for j in rest]).mean(0),
                   "memory": state.pseudo_memory[i] - torch.stack([state.pseudo_memory[j] for j in rest]).mean(0)}
            scores.append(self.image_score(self.to_map(self.fuse(res), None)))
        use = scores[:state.n_real] if state.n_real >= 2 else scores
        level = max(float(np.mean(use)), 1e-6)
        state._level = (key, level)
        return level

    def threshold_for(self, n_references: int) -> float:
        """Calibrated threshold for this many references: the largest calibrated
        reference count not above it, else the default threshold."""
        by_refs = self.calibration.get("by_references") or {}
        rule = self.calibration.get("rule", "ratio")
        usable = sorted(int(k) for k in by_refs if int(k) <= n_references)
        if usable:
            return float(by_refs[str(usable[-1])][rule])
        return float(self.calibration["threshold"])

    def decision_value(self, score: float, state: ReferenceState) -> float:
        """The statistic compared with the threshold: ``ratio`` (score over this
        product's normal level, the default) or ``absolute``."""
        if self.calibration.get("rule", "ratio") == "ratio":
            return score / self.normal_level(state)
        return score

    @torch.no_grad()
    def compare(self, current, state: ReferenceState, sensitivity: float = 1.0) -> dict:
        """Full comparison of one current image. The result is JSON-friendly
        except for the arrays, which the caller renders.

        ``sensitivity`` scales the calibrated threshold: above 1 flags more,
        below 1 flags less. Normal variation differs between products, so a user
        may need to move it for their product.
        """
        res = self.residuals(current, state)
        amap = self.to_map(self.fuse(res), res["valid"])
        score = self.image_score(amap)
        value = self.decision_value(score, state)
        threshold = self.threshold_for(len(state.images)) / max(sensitivity, 1e-6)
        return {
            "score": score,
            "relative_score": score / self.normal_level(state),
            "decision_value": value,
            "rule": self.calibration.get("rule", "ratio"),
            "threshold": threshold,
            "is_anomalous": bool(value > threshold),
            "map": amap[0, 0].float().cpu().numpy(),
            "aligned": res["aligned"],
            "valid": res["valid"],
            "alignment": {"aligned": bool(res["align"].get("aligned")),
                          "cc": res["align"].get("cc")},
            "jepa_grid": res["jepa"].cpu().numpy(),
            "memory_grid": res["memory"].cpu().numpy(),
        }
