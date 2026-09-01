"""Multi-seed aggregation for experiment results.

MVTec LOCO test splits are small -- a category may hold well under a hundred
test images -- so a single-seed AUROC carries a standard deviation large enough
to reorder ablation arms by itself. Every headline number in this project is
therefore reported as ``mean ± std`` over several seeds, and anything derived
from one run is explicitly labelled as such.

The comparison helper deliberately refuses to call a difference meaningful when
the seed spread swamps it: :func:`compare_arms` reports the gap in units of the
pooled standard deviation, so "arm A beats arm B" has to survive the noise
before it is stated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass
class SeedStat:
    """Summary of one metric across seeds."""

    metric: str
    values: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        clean = self._clean()
        return sum(clean) / len(clean) if clean else float("nan")

    @property
    def std(self) -> float:
        """Sample standard deviation (ddof=1); 0.0 when only one seed ran."""
        clean = self._clean()
        if len(clean) < 2:
            return 0.0
        mu = sum(clean) / len(clean)
        return math.sqrt(sum((v - mu) ** 2 for v in clean) / (len(clean) - 1))

    @property
    def sem(self) -> float:
        """Standard error of the mean -- the right bar for 'is this different?'."""
        clean = self._clean()
        return self.std / math.sqrt(len(clean)) if len(clean) > 1 else 0.0

    @property
    def spread(self) -> tuple[float, float]:
        clean = self._clean()
        return (min(clean), max(clean)) if clean else (float("nan"), float("nan"))

    def _clean(self) -> list[float]:
        return [v for v in self.values if v is not None and not _isnan(v)]

    def format(self, precision: int = 4) -> str:
        """``0.9312 ± 0.0087`` -- or ``0.9312 (1 seed)`` when unreplicated."""
        if self.n == 0:
            return "n/a"
        if len(self._clean()) < 2:
            return f"{self.mean:.{precision}f} (1 seed)"
        return f"{self.mean:.{precision}f} ± {self.std:.{precision}f}"

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.spread
        return {
            "metric": self.metric,
            "mean": self.mean,
            "std": self.std,
            "sem": self.sem,
            "min": lo,
            "max": hi,
            "n_seeds": len(self._clean()),
            "values": list(self.values),
        }


def _isnan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def aggregate_seeds(
    per_seed_results: Sequence[Mapping[str, Any]],
    metrics: Iterable[str] | None = None,
) -> dict[str, SeedStat]:
    """Collect ``{metric: SeedStat}`` from a list of per-seed result dicts.

    Non-numeric fields (category names, checkpoint paths) are skipped rather
    than coerced, so the same result dicts the single-seed path produces can be
    passed straight in.
    """
    if not per_seed_results:
        return {}

    if metrics is None:
        keys: list[str] = []
        for result in per_seed_results:
            for key, value in result.items():
                if key in keys:
                    continue
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    keys.append(key)
        metrics = keys

    stats: dict[str, SeedStat] = {}
    for metric in metrics:
        stat = SeedStat(metric=metric)
        for result in per_seed_results:
            if metric in result:
                try:
                    stat.values.append(float(result[metric]))
                except (TypeError, ValueError):
                    pass
        if stat.values:
            stats[metric] = stat
    return stats


def format_seed_row(
    arm: str,
    stats: Mapping[str, SeedStat],
    metrics: Sequence[str],
    precision: int = 4,
) -> dict[str, str]:
    """One table row of ``mean ± std`` strings, for the console summary."""
    row: dict[str, str] = {"arm": arm}
    n_seeds = max((s.n for s in stats.values()), default=0)
    row["seeds"] = str(n_seeds)
    for metric in metrics:
        row[metric] = stats[metric].format(precision) if metric in stats else "n/a"
    return row


def flatten_for_csv(
    arm: str,
    description: str,
    stats: Mapping[str, SeedStat],
    metrics: Sequence[str],
) -> dict[str, Any]:
    """A CSV row carrying mean, std and seed count as separate columns."""
    row: dict[str, Any] = {"arm": arm, "description": description}
    row["n_seeds"] = max((s.n for s in stats.values()), default=0)
    for metric in metrics:
        if metric in stats:
            row[f"{metric}_mean"] = stats[metric].mean
            row[f"{metric}_std"] = stats[metric].std
        else:
            row[f"{metric}_mean"] = float("nan")
            row[f"{metric}_std"] = float("nan")
    return row


def compare_arms(
    a_name: str,
    a: SeedStat,
    b_name: str,
    b: SeedStat,
    threshold_sigmas: float = 1.0,
) -> str:
    """State whether two arms actually differ, given the seed noise.

    The gap is expressed in pooled standard deviations. Below
    ``threshold_sigmas`` the arms are reported as indistinguishable rather than
    ranked -- which is the honest reading when the spread is as large as the
    difference.
    """
    if a.n == 0 or b.n == 0:
        return f"{a_name} vs {b_name}: insufficient data"

    gap = a.mean - b.mean
    pooled = math.sqrt((a.std**2 + b.std**2) / 2.0) if (a.n > 1 and b.n > 1) else 0.0

    if pooled <= 1e-12:
        detail = "(single seed -- no spread available)" if a.n < 2 or b.n < 2 else ""
        winner = a_name if gap > 0 else b_name
        return f"{winner} leads by {abs(gap):.4f} {detail}".strip()

    sigmas = abs(gap) / pooled
    if sigmas < threshold_sigmas:
        return (
            f"{a_name} vs {b_name}: indistinguishable "
            f"(gap {gap:+.4f} = {sigmas:.1f}σ of seed noise)"
        )

    winner, loser = (a_name, b_name) if gap > 0 else (b_name, a_name)
    return (
        f"{winner} > {loser} by {abs(gap):.4f} ({sigmas:.1f}σ of seed noise)"
    )


def parse_seeds(text: str | None, default: Sequence[int] = (0,)) -> list[int]:
    """Parse ``--seeds 0,1,2`` (also accepts ``0 1 2`` and ranges ``0-2``)."""
    if not text:
        return list(default)

    seeds: list[int] = []
    for chunk in str(text).replace(",", " ").split():
        if "-" in chunk and not chunk.startswith("-"):
            lo, _, hi = chunk.partition("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(chunk))

    if not seeds:
        raise ValueError(f"could not parse any seed from '{text}'")
    return seeds
