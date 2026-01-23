# Factor Portfolio (RFF + Markowitz) Research Toolkit

Contributor: Kaihe Liu
Tutor: Prof. Xi Wang at PKU

A lightweight, modular Python research project for cross-sectional factor data preparation, Random Fourier Features (RFF) factor expansion, optional Partial Least Squares (PLS) dimensionality reduction, and ridge-regularized Markowitz portfolio construction. Includes an experiment runner to sweep hyperparameters and produce Sharpe-vs-factor-count plots.

## Features

- **Data loading by year** (CSV/Parquet) with a clean `DataSpec` abstraction
- **Cross-sectional preprocessing**
  - key standardization (`TradingDay`, `SecuCode`)
  - sequential left-merge across datasets
  - return construction (`ret`, `next_ret`)
  - optional forward-fill by asset
  - optional per-day z-score standardization
- **Feature engineering**
  - Random Fourier Features (RFF)
  - optional PLS dimensionality reduction
- **Portfolio construction**
  - ridge-regularized Markowitz in feature space: solve `(Cov + λI) w = F_t`
- **Experiment**
  - hyperparameter sweep over `n_factors` and `λ`
  - compute mean/std Sharpe (mean/std over days)
  - export plot + CSV tables

---

## Repository Structure

```
├─ README.md
├─ src/
│  └─ factor_portfolio/
│     ├─ config.py
│     ├─ utils.py
│     ├─ data/
│     │  ├─ loaders.py
│     │  └─ preprocess.py
│     ├─ features/
│     │  ├─ rff.py
│     │  └─ pls.py
│     ├─ portfolio/
│     │  └─ markowitz.py
│     └─ experiments/
|        └─ sharpe_sweep.py
|     └─ run_sharpe_sweep.py
```
