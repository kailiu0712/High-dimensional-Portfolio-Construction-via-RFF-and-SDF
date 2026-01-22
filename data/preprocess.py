from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


def merge_on_keys(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Left-merge sequentially on ['TradingDay', 'SecuCode'].
    The first df is treated as the base/universe dataframe.
    """
    if not dfs:
        raise ValueError("dfs is empty")

    merged = dfs[0].copy()
    for df in dfs[1:]:
        merged = merged.merge(df, on=["TradingDay", "SecuCode"], how="left")
    return merged


def add_returns(df: pd.DataFrame, price_col: str, ret_col: str, next_ret_col: str) -> pd.DataFrame:
    """
    Compute per-asset returns and next-day returns:
    ret[t] = pct_change(price[t])
    next_ret[t] = ret[t+1] aligned to day t
    """
    out = df.sort_values(["SecuCode", "TradingDay"]).copy()
    out[ret_col] = out.groupby("SecuCode")[price_col].pct_change()
    out[next_ret_col] = out.groupby("SecuCode")[ret_col].shift(-1)
    return out


def forward_fill_by_asset(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    """
    Forward-fill factor columns within each SecuCode.
    """
    out = df.sort_values(["SecuCode", "TradingDay"]).copy()
    out[factor_cols] = out.groupby("SecuCode")[factor_cols].ffill()
    return out


def zscore_by_day(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    """
    Z-score standardize factor columns within each TradingDay:
    z = (x - mean_day) / std_day
    """
    out = df.copy()

    def _z(g: pd.DataFrame) -> pd.DataFrame:
        x = g[factor_cols]
        mu = x.mean(axis=0)
        sig = x.std(axis=0).replace(0.0, np.nan)
        g.loc[:, factor_cols] = (x - mu) / sig
        return g

    out = out.groupby("TradingDay", group_keys=False).apply(_z)
    return out


def drop_rows_with_missing(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    return out.dropna(subset=required_cols)
