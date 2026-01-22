from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from ..config import DataSpec
from ..utils import get_logger

logger = get_logger(__name__)


def _read_one_file(path: Path, file_type: str, usecols: List[str]) -> pd.DataFrame:
    if file_type.lower() == "parquet":
        return pd.read_parquet(path, columns=usecols)
    if file_type.lower() == "csv":
        return pd.read_csv(path, usecols=usecols)
    raise ValueError(f"Unsupported file_type={file_type}. Use 'csv' or 'parquet'.")


def load_by_years(
    data_root: Path,
    spec: DataSpec,
    years: Iterable[int],
) -> pd.DataFrame:
    """
    Load a dataset for multiple years and concatenate.
    """
    frames: List[pd.DataFrame] = []
    for year in years:
        rel = spec.file_template.format(year=year)
        path = data_root / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing file for {spec.name}: {path}")

        df = _read_one_file(path, spec.file_type, spec.usecols)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    if spec.filter_expr:
        out = out.query(spec.filter_expr)

    return out


def standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize key columns:
    - TradingDay -> datetime64
    - SecuCode -> zero-padded string length 6
    """
    df = df.copy()
    if "TradingDay" in df.columns:
        df["TradingDay"] = pd.to_datetime(df["TradingDay"])
    if "SecuCode" in df.columns:
        df["SecuCode"] = df["SecuCode"].astype(str).str.zfill(6)
    return df
