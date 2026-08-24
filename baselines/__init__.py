"""Comparison baselines, all trained from scratch alongside Logical-JEPA.

===============  ==========================================================
Autoencoder      pixel reconstruction error -- the classical detector
PatchCore        nominal patch memory bank with coreset subsampling
===============  ==========================================================

Both are expected to handle structural anomalies and to struggle with logical
ones, for reasons documented in their modules. Establishing that gap is the
point of including them.
"""

from .autoencoder import (
    AutoencoderScorer,
    ConvAutoencoder,
    reconstruction_loss,
    train_autoencoder,
)
from .patchcore import PatchCore, extract_patch_features, greedy_coreset

__all__ = [
    "ConvAutoencoder",
    "AutoencoderScorer",
    "reconstruction_loss",
    "train_autoencoder",
    "PatchCore",
    "greedy_coreset",
    "extract_patch_features",
]
