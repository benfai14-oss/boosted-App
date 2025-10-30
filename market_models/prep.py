"""
market_models.prep
-------------------

Data preparation functions used by the forecasting pipeline.  These
utilities consolidate price series and climate risk measures into a
single table, compute log returns and construct lagged features for
use in the simple ARIMAX model defined in :mod:`market_models.arimax`.

These routines are designed to be composable and testable.  They
accept ``pandas.DataFrame`` objects directly and do not depend on
project‑specific file formats.  See the CLI in :mod:`interface.cli` for
examples of how to call these functions from the command line.
"""

from __future__ import annotations

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np


def merge_price_and_risk(
    price_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    *,
    price_col: str = "price_front_fut",
    risk_col: str = "global_risk_0_100",
    lags: int = 1,
    mode: str = "legacy",
    exog_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merge price and risk data frames and construct feature matrix.

    This function supports two modes:
    - 'legacy': aligns two time series, computes log returns, and constructs lagged features of returns and risk index.
    - 'auto': computes log returns and a normalized risk column, includes optional exogenous columns without creating lags,
      returning data ready for extended ARIMAX to auto-build lags/seasonality.

    Parameters
    ----------
    price_df : pandas.DataFrame
        Data frame containing at least a ``date`` column and a price
        column (by default ``price_front_fut``).  The ``date`` column
        must be convertible to a datetime and unique.
    risk_df : pandas.DataFrame
        Data frame containing at least a ``date`` column and a risk
        column (by default ``global_risk_0_100``).
    price_col : str, default "price_front_fut"
        Name of the price column in ``price_df``.
    risk_col : str, default "global_risk_0_100"
        Name of the risk column in ``risk_df``.
    lags : int, default 1
        Number of lagged observations to include for both the price
        returns and the risk index.  Used only in 'legacy' mode.
    mode : str, default "legacy"
        Mode of operation: 'legacy' to build lagged features (default),
        'auto' to prepare data without lags for extended ARIMAX.
    exog_cols : list of str, optional
        List of additional exogenous column names present in ``risk_df`` to include in 'auto' mode.

    Returns
    -------
    (df, features) : (pandas.DataFrame, list[str])
        In 'legacy' mode, ``df`` is indexed by date containing target ``y`` and lagged features,
        and ``features`` is the list of exogenous column names.
        In 'auto' mode, ``df`` includes ``y``, ``risk`` (normalized), and any exogenous columns,
        with no lagged features, and ``features`` is an empty list.

    Raises
    ------
    ValueError
        If required columns are missing or if mode is invalid.
    """
    # Validate mode
    if mode not in {"legacy", "auto"}:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'legacy' or 'auto'.")
    # Copy and convert date columns to datetime
    p = price_df.copy()
    r = risk_df.copy()
    if "date" not in p.columns:
        raise ValueError("price_df must contain 'date' column.")
    if "date" not in r.columns:
        raise ValueError("risk_df must contain 'date' column.")
    p["date"] = pd.to_datetime(p["date"], errors="raise")
    r["date"] = pd.to_datetime(r["date"], errors="raise")
    # Validate required columns
    if price_col not in p.columns:
        raise ValueError(f"price_df must contain price column '{price_col}'.")
    if risk_col not in r.columns:
        raise ValueError(f"risk_df must contain risk column '{risk_col}'.")
    # Merge on date
    df = pd.merge(p[["date", price_col]], r, on="date", how="inner")
    df = df.sort_values("date").set_index("date")
    # Compute log return as target
    df["y"] = np.log(df[price_col] / df[price_col].shift(1))
    # Drop rows with missing values in price or risk_col before further processing
    df = df.dropna(subset=[price_col, risk_col, "y"])
    if mode == "legacy":
        # Normalize risk to 0..1 scale
        df["risk_norm"] = df[risk_col] / 100.0
        feature_cols: List[str] = []
        for i in range(1, lags + 1):
            y_lag = f"y_lag{i}"
            r_lag = f"risk_lag{i}"
            df[y_lag] = df["y"].shift(i)
            df[r_lag] = df["risk_norm"].shift(i)
            feature_cols.extend([y_lag, r_lag])
        df = df.dropna()
        return df[["y"] + feature_cols], feature_cols
    else:  # mode == "auto"
        # Normalize risk to 0..1 scale and rename column to 'risk'
        df["risk"] = df[risk_col] / 100.0
        # Include specified exogenous columns if provided
        exog_cols = exog_cols or []
        missing_exog = [col for col in exog_cols if col not in df.columns]
        if missing_exog:
            raise ValueError(f"Exogenous columns missing from merged data frame: {missing_exog}")
        # Select columns: y, risk, and exog_cols
        cols = ["y", "risk"] + exog_cols
        df_auto = df[cols].dropna()
        return df_auto, []
