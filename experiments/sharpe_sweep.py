from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..config import ExperimentConfig
from ..features.rff import RFFConfig, rff_features
from ..features.pls import pls_reduce
from ..portfolio.markowitz import expected_return_and_cov, ridge_markowitz_weights
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SweepResult:
    sharpe: pd.DataFrame  # index: n_factors, columns: Lambda_*
    mean: pd.DataFrame
    std: pd.DataFrame


def _compute_past_return_avg(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Past rolling average return per asset, shifted by 1 day.
    Requires df sorted by SecuCode, TradingDay and includes 'ret'.
    """
    return df.groupby("SecuCode")["ret"].transform(lambda x: x.rolling(window, min_periods=window).mean().shift(1))


def run_sharpe_sweep(
    merged_df: pd.DataFrame,
    factor_names: List[str],
    cfg: ExperimentConfig,
    use_pls: bool = False,
) -> SweepResult:
    """
    For each n_factors and lambda:
      - build expected return proxy using past rolling avg returns
      - build RFF factors (optionally PLS-reduced)
      - solve ridge Markowitz weights in feature space
      - compute daily realized return using same-day realized returns ("ret") (research-style)
      - aggregate across days -> mean/std -> Sharpe
    """
    df = merged_df.copy()
    df = df.sort_values(["TradingDay", "SecuCode"]).reset_index(drop=True)

    # Compute realized returns proxy fields
    if "ret" not in df.columns:
        raise ValueError("merged_df must contain column 'ret'. Compute it during preprocessing.")

    df["past_ret_ave"] = _compute_past_return_avg(df, window=5)
    df = df.dropna(subset=["past_ret_ave", "ret"])

    trading_days = df["TradingDay"].unique()

    cols = [f"Lambda_{lam}" for lam in cfg.lambdas]
    mean_df = pd.DataFrame(index=cfg.n_factors_range, columns=cols, dtype=float)
    std_df = pd.DataFrame(index=cfg.n_factors_range, columns=cols, dtype=float)
    sharpe_df = pd.DataFrame(index=cfg.n_factors_range, columns=cols, dtype=float)

    rff_cfg = RFFConfig(random_seed=cfg.random_seed)

    for n_factors in tqdm(cfg.n_factors_range, desc="Sweeping n_factors"):
        for lam in cfg.lambdas:
            daily_returns: List[float] = []

            for day in trading_days:
                day_df = df[df["TradingDay"] == day]

                # Expected returns proxy (in-sample expectation estimator)
                exp_ret = day_df["past_ret_ave"]
                realized_ret = day_df["ret"]

                # RFF features over specified factor columns
                feats = rff_features(day_df[factor_names], factor_names=factor_names, n_factors=n_factors, cfg=rff_cfg)
                feats.index = day_df.index  # preserve alignment

                # Optional PLS reduction
                if use_pls:
                    feats = pls_reduce(feats, exp_ret, n_components=cfg.n_pls_components)

                # Compute F_t and cov in feature space
                F_t, cov = expected_return_and_cov(feats, exp_ret)

                # Solve for weights in feature space
                w_df = ridge_markowitz_weights(F_t, cov, lambdas=[lam])
                w = w_df[f"Lambda_{lam}"].to_numpy(dtype=np.float64)

                # Out-of-sample style evaluation: use realized returns
                ofs_Ft, _ = expected_return_and_cov(feats, realized_ret)
                day_return = float(w @ ofs_Ft)
                daily_returns.append(day_return)

            mu = float(np.mean(daily_returns))
            sig = float(np.std(daily_returns, ddof=0))
            mean_df.loc[n_factors, f"Lambda_{lam}"] = mu
            std_df.loc[n_factors, f"Lambda_{lam}"] = sig
            sharpe_df.loc[n_factors, f"Lambda_{lam}"] = (mu / sig) if sig > 0 else np.nan

    return SweepResult(sharpe=sharpe_df, mean=mean_df, std=std_df)


def save_sharpe_plot(result: SweepResult, cfg: ExperimentConfig) -> str:
    """
    Save sharpe plot to cfg.output_root/cfg.plot_filename and return filepath.
    """
    cfg.ensure_dirs()
    out_path = cfg.output_root / cfg.plot_filename

    plt.figure(figsize=(9, 4), dpi=cfg.plot_dpi)
    ax = plt.gca()

    for col in result.sharpe.columns:
        ax.plot(result.sharpe.index, result.sharpe[col], label=col)

    ax.set_xlabel("Number of RFF Factors")
    ax.set_ylabel("Sharpe Ratio (mean/std over days)")
    ax.set_title("Sharpe Ratio vs Number of Factors")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", frameon=False)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return str(out_path)
