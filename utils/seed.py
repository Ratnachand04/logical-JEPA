"""Reproducibility helpers for Logical-JEPA.

Anomaly-detection numbers on MVTec LOCO are noisy across seeds (the test sets
are small), so every experiment in this repository is seeded explicitly and the
seed is recorded in the result files.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """Seed python, numpy and torch RNGs.

    Args:
        seed: base seed.
        deterministic: if True, force cuDNN into deterministic mode. This makes
            runs bit-reproducible but noticeably slower, so it is off by
            default and only enabled for the ablation study.

    Returns:
        The seed that was applied (useful for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Some conv/attention kernels have no deterministic implementation; warn
        # rather than crash so training still runs.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding.

    Without this every worker inherits the same numpy seed and the synthetic
    augmentations repeat across workers.
    """
    base = torch.initial_seed() % 2**31
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
