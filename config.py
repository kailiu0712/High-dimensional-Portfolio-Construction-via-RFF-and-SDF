from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DataSpec:
    """
    Declares how to load each dataset.

    - file_template: relative path template, can reference {year}
    - file_type: "csv" or "parquet"
    - usecols: columns to load
    - filter_expr: optional pandas query applied after loading (e.g., "IndexW300 > 0")
    """
    name: str
    file_template: str
    file_type: str
    usecols: List[str]
    filter_expr: Optional[str] = None


@dataclass
class ExperimentConfig:
    # Years
    start_year: int = 2017
    end_year: int = 2024

    # Data root (set this to your project data directory)
    data_root: Path = Path("data")

    # Output root
    output_root: Path = Path("outputs")

    # Universe filter column (e.g., IndexW300)
    universe_filter_col: str = "IndexW300"
    universe_filter_threshold: float = 0.0

    # Return construction
    price_col: str = "ClosePrice"
    ret_col: str = "ret"
    next_ret_col: str = "next_ret"

    # Missing data handling
    ffill_by_asset: bool = True  # forward-fill by SecuCode
    dropna_after_merge: bool = True

    # Feature standardization ("zscore" recommended)
    standardize_by_day: bool = True

    # Modeling / hyperparameters
    random_seed: int = 42

    # RFF
    n_pls_components: int = 10
    n_factors_range: List[int] = field(default_factory=lambda: [
        10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
        85, 90, 95, 100, 110, 120, 130, 140, 150, 170, 200
    ])
    lambdas: List[float] = field(default_factory=lambda: [1e-5, 1e-3, 1e-1, 1.0, 10.0])

    # Plot settings
    plot_dpi: int = 200
    plot_filename: str = "sharpe_vs_num_factors.png"

    def ensure_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
