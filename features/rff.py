from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RFFConfig:
    random_seed: int = 42


def rff_features(
    X: pd.DataFrame,
    factor_names: List[str],
    n_factors: int,
    cfg: RFFConfig,
) -> pd.DataFrame:
    """
    Random Fourier Features:
      phi(x) = sqrt(2/d) * [cos(gamma * xW), sin(gamma * xW)]
    Returns 2*n_factors columns.

    Notes:
    - W ~ N(0,1)
    - gamma sampled from a small discrete set (as in your original code)
    """
    X = X[factor_names].to_numpy(dtype=np.float64, copy=False)

    rng = np.random.default_rng(cfg.random_seed)
    omega = rng.normal(0.0, 1.0, size=(X.shape[1], n_factors))
    gamma = rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    proj = X @ omega * gamma
    cos_part = np.sqrt(2.0 / n_factors) * np.cos(proj)
    sin_part = np.sqrt(2.0 / n_factors) * np.sin(proj)

    feats = np.hstack([cos_part, sin_part])
    cols = [f"RFF_{i}" for i in range(1, 2 * n_factors + 1)]
    return pd.DataFrame(feats, index=None, columns=cols)
