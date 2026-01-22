from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


def expected_return_and_cov(
    factors: pd.DataFrame,
    expected_return: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given factor exposures (rows = assets) and expected returns (per asset),
    compute:
      F_t = factors^T * expected_return
      Cov = cov(factors^T)

    Returns:
      F_t: (n_features,)
      Cov: (n_features, n_features)
    """
    # Align
    expected_return = expected_return.reindex(factors.index)
    X = factors.to_numpy(dtype=np.float64, copy=False)
    y = expected_return.to_numpy(dtype=np.float64, copy=False)

    # F_t = X^T y
    F_t = X.T @ y  # shape: (n_features,)

    # covariance of features across assets
    cov = np.cov(X.T).astype(np.float64)  # shape: (n_features, n_features)
    return F_t, cov


def ridge_markowitz_weights(
    F_t: np.ndarray,
    cov: np.ndarray,
    lambdas: Iterable[float],
) -> pd.DataFrame:
    """
    Solve (cov + lambda I) w = F_t for each lambda.
    """
    n = F_t.shape[0]
    I = np.eye(n, dtype=np.float64)

    out = {}
    for lam in lambdas:
        Q = cov + float(lam) * I
        w = np.linalg.solve(Q, F_t)
        out[f"Lambda_{lam}"] = w

    return pd.DataFrame(out)
