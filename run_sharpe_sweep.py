from __future__ import annotations

from pathlib import Path

from factor_portfolio.config import DataSpec, ExperimentConfig
from factor_portfolio.data.loaders import load_by_years, standardize_keys
from factor_portfolio.data.preprocess import (
    merge_on_keys,
    add_returns,
    forward_fill_by_asset,
    zscore_by_day,
    drop_rows_with_missing,
)
from factor_portfolio.experiments.sharpe_sweep import run_sharpe_sweep, save_sharpe_plot
from factor_portfolio.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = ExperimentConfig(
        start_year=2017,
        end_year=2024,
        data_root=Path("data"),      # <- set your local data folder here
        output_root=Path("outputs"), # <- where plots/tables go
    )

    years = range(cfg.start_year, cfg.end_year + 1)

    # ---- Declare dataset specs (NO sensitive paths) ----
    # Adjust file_template to match your directory layout.
    universe_spec = DataSpec(
        name="universe_weights",
        file_template="universe/{year}/stock_index_weight.parquet",
        file_type="parquet",
        usecols=["TradingDay", "SecuCode", "IndexW300"],
        filter_expr="IndexW300 > 0",
    )

    valuation_spec = DataSpec(
        name="valuation",
        file_template="factors/{year}/Factors_Valuation.csv",
        file_type="csv",
        usecols=["TradingDay", "SecuCode", "ClosePrice", "EP_TTM"],
    )

    valuation_rank_spec = DataSpec(
        name="valuation_rank",
        file_template="factors/{year}/Factors_ValuationRank.csv",
        file_type="csv",
        usecols=["TradingDay", "SecuCode", "REP_TTM"],
    )

    technical_spec = DataSpec(
        name="technical1",
        file_template="factors/{year}/Factors_Technical1.csv",
        file_type="csv",
        usecols=["TradingDay", "SecuCode", "RT_2M", "TO_1M"],
    )

    financial_spec = DataSpec(
        name="financial",
        file_template="factors/{year}/Factors_Financial.csv",
        file_type="csv",
        usecols=["TradingDay", "SecuCode", "ROE"],
    )

    growth_spec = DataSpec(
        name="growth",
        file_template="factors/{year}/Factors_Growth.csv",
        file_type="csv",
        usecols=["TradingDay", "SecuCode", "E_Growth2"],
    )

    shareholder_spec = DataSpec(
        name="shareholder",
        file_template="factors/{year}/Factors_Shareholder.csv",
        file_type="csv",
        usecols=["TradingDay", "SecuCode", "SHNum"],
    )

    risk_div_spec = DataSpec(
        name="risk_div",
        file_template="factors/{year}/Factors_RiskDiv.csv",
        file_type="csv",
        usecols=["TradingDay", "SecuCode", "DivR1"],
    )

    # ---- Load ----
    logger.info("Loading datasets...")
    weight_df = standardize_keys(load_by_years(cfg.data_root, universe_spec, years))
    valuation_df = standardize_keys(load_by_years(cfg.data_root, valuation_spec, years))
    valuation_rank_df = standardize_keys(load_by_years(cfg.data_root, valuation_rank_spec, years))
    technical_df = standardize_keys(load_by_years(cfg.data_root, technical_spec, years))
    financial_df = standardize_keys(load_by_years(cfg.data_root, financial_spec, years))
    growth_df = standardize_keys(load_by_years(cfg.data_root, growth_spec, years))
    shareholder_df = standardize_keys(load_by_years(cfg.data_root, shareholder_spec, years))
    div_df = standardize_keys(load_by_years(cfg.data_root, risk_div_spec, years))

    # ---- Merge ----
    merged = merge_on_keys([
        weight_df,
        valuation_df,
        valuation_rank_df,
        technical_df,
        financial_df,
        growth_df,
        shareholder_df,
        div_df,
    ])

    # ---- Returns ----
    merged = add_returns(merged, price_col=cfg.price_col, ret_col=cfg.ret_col, next_ret_col=cfg.next_ret_col)

    # ---- Define factor list (your "7-factor" example) ----
    factor_names = ["EP_TTM", "REP_TTM", "RT_2M", "TO_1M", "ROE", "E_Growth2", "SHNum", "DivR1"]

    # ---- Fill + standardize (optional but recommended) ----
    if cfg.ffill_by_asset:
        merged = forward_fill_by_asset(merged, factor_cols=factor_names)

    if cfg.standardize_by_day:
        merged = zscore_by_day(merged, factor_cols=factor_names)

    # Keep only required columns
    required = ["TradingDay", "SecuCode", "ret", "past_dummy"]  # placeholder to build list below
    required = ["TradingDay", "SecuCode", "ret"] + factor_names
    merged = drop_rows_with_missing(merged, required_cols=required)

    logger.info("Merged dataset ready. Unique trading days: %d", merged["TradingDay"].nunique())

    # ---- Run sweep ----
    result = run_sharpe_sweep(
        merged_df=merged,
        factor_names=factor_names,
        cfg=cfg,
        use_pls=False,  # set True if you want PLS reduction
    )

    # ---- Save plot ----
    plot_path = save_sharpe_plot(result, cfg)
    logger.info("Saved plot: %s", plot_path)

    # Optional: save sharpe table
    sharpe_csv = cfg.output_root / "sharpe_table.csv"
    result.sharpe.to_csv(sharpe_csv)
    logger.info("Saved Sharpe table: %s", sharpe_csv)


if __name__ == "__main__":
    main()
