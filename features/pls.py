from __future__ import annotations

import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from ..utils import get_logger

logger = get_logger(__name__)


def pls_reduce(X: pd.DataFrame, y: pd.Series, n_components: int) -> pd.DataFrame:
    """
    Partial Least Squares dimensionality reduction.

    This is optional. If you do not want PLS, just skip this step.
    """
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)

    X_reduced = pls.transform(X)
    cols = [f"PLS_Component_{i}" for i in range(1, n_components + 1)]
    return pd.DataFrame(X_reduced, index=X.index, columns=cols)
