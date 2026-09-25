"""RNG seed helpers for reproducible experiment orchestration."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def repeat_seed(base_seed: int, repeat_index: int) -> int:
    """Deterministically derive per-repeat seeds from a base seed."""
    return int(base_seed) + int(repeat_index) * 9973
