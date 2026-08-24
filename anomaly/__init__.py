"""Anomaly scoring, heatmap construction and evaluation metrics.

The pipeline is::

    masked sweep --> per-scale 16x16 error grids   (embedding_error)
                 --> normalise, fuse, upsample     (anomaly_map)
                 --> image score + calibration     (scoring)
                 --> AUROC / AU-PRO, split by type (metrics)
"""

from .anomaly_map import (
    FUSION_MODES,
    build_anomaly_map,
    fuse_scales,
    gaussian_blur,
    normalize_grid,
    to_display_map,
    upsample_map,
)
from .embedding_error import (
    DISTANCES,
    combined_distance,
    cosine_distance,
    jepa_loss,
    l1_distance,
    l2_distance,
    patch_distance,
    smooth_l1_distance,
)
from .metrics import (
    compute_pro,
    evaluate_split,
    image_auroc,
    image_average_precision,
    pixel_auroc,
    summarize,
)
from .scoring import (
    AGGREGATIONS,
    AnomalyScorer,
    Calibration,
    aggregate_score,
    build_scorer,
    fit_calibration,
)

__all__ = [
    # embedding_error
    "DISTANCES",
    "cosine_distance",
    "l2_distance",
    "l1_distance",
    "smooth_l1_distance",
    "combined_distance",
    "patch_distance",
    "jepa_loss",
    # anomaly_map
    "FUSION_MODES",
    "normalize_grid",
    "fuse_scales",
    "upsample_map",
    "gaussian_blur",
    "build_anomaly_map",
    "to_display_map",
    # scoring
    "AGGREGATIONS",
    "aggregate_score",
    "Calibration",
    "fit_calibration",
    "AnomalyScorer",
    "build_scorer",
    # metrics
    "image_auroc",
    "image_average_precision",
    "pixel_auroc",
    "compute_pro",
    "evaluate_split",
    "summarize",
]
