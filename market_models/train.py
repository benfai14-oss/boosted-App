from __future__ import annotations

"""
market_models.train (extended)
------------------------------

High-level training/forecast entry point for Layer C using the extended
ARIMAX. This version supports two preparation modes and seasonal terms.

• mode="legacy": maintain backward compatibility. We construct explicit
  lag columns in :mod:`market_models.prep` and pass them as
  `feature_cols` to the model.
• mode="auto": we only compute the target `y` and a base risk column
  normalised to 0..1 (named `risk`). The model will auto-build lags and
  seasonality (p, q, sin/cos).

Outputs a compact table with horizons, cumulative returns, price
forecasts and naive 95% bands from residual variance.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from .prep import merge_price_and_risk
from .arimax import fit_arimax, forecast_arimax


def train_and_forecast(
    price_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    *,
    horizons: List[int] = (30, 60, 90),
    price_col: str = "price_front_fut",
    risk_col: str = "global_risk_0_100",
    lags: int = 1,
    mode: str = "legacy",
    p: int = 2,
    q: int = 2,
    include_seasonal: bool = True,
    season_period: int = 52,
    exog_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Train ARIMAX and generate multi‑horizon forecasts.

    Parameters
    ----------
    price_df : DataFrame with columns [date, price_col]
    risk_df  : DataFrame with columns [date, risk_col]
    horizons : iterable of horizons in days (converted to weekly steps)
    price_col, risk_col : input column names
    lags : (legacy mode) number of lags to engineer for y and risk
    mode : "legacy" | "auto" (see module docstring)
    p, q : AR and risk lag orders when mode="auto"
    include_seasonal : add sin/cos seasonal terms when mode="auto"
    season_period : seasonal period (52 for weekly data)
    exog_cols : optional list of extra exogenous columns to include
                (names must exist in inputs and start with "exog_")

    Returns
    -------
    DataFrame with one row per horizon: [horizon_days, steps,
    price_forecast, cum_return, lo95, hi95].
    """
    if mode not in {"legacy", "auto"}:
        raise ValueError("mode must be 'legacy' or 'auto'")

    # 1) Prepare data
    merged, features = merge_price_and_risk(
        price_df,
        risk_df,
        price_col=price_col,
        risk_col=risk_col,
        lags=lags,
        mode=mode,
        exog_cols=exog_cols,
    )
    if merged.empty:
        raise ValueError("Merged DataFrame is empty – cannot train model")

    # 2) Fit model
    if mode == "legacy":
        params = fit_arimax(merged, features)
    else:  # auto
        params = fit_arimax(
            merged,
            feature_cols=None,
            p=p,
            q=q,
            include_seasonal=include_seasonal,
            season_period=season_period,
            extra_exog=exog_cols,
        )

    # 3) Prepare last observation and baseline risk path
    last_observation = merged.iloc[-1]

    # Retrieve last price aligned to the index of merged (DateIndex)
    # When inputs are weekly, merged index is the weekly date; match it.
    if "date" in price_df.columns:
        # exact match on date
        last_price_row = price_df.loc[price_df["date"] == last_observation.name, price_col]
        if len(last_price_row) == 0:
            # fallback: take the last known price
            last_price = float(price_df[price_col].iloc[-1])
        else:
            last_price = float(last_price_row.iloc[0])
    else:
        last_price = float(price_df[price_col].iloc[-1])

    # Risk baseline for the future path
    if mode == "legacy":
        if "risk_lag1" in last_observation:
            last_risk_norm = float(last_observation["risk_lag1"])
        elif "risk_norm" in merged.columns:
            last_risk_norm = float(merged["risk_norm"].iloc[-1])
        else:
            raise ValueError("Could not infer last risk value in legacy mode.")
        risk_order = max(1, lags)
    else:
        # auto mode: base risk column is named 'risk' (0..1)
        if "risk" not in merged.columns:
            raise ValueError("Expected a base 'risk' column in auto mode.")
        last_risk_norm = float(merged["risk"].iloc[-1])
        risk_order = max(1, q)

    rows: List[Dict[str, Any]] = []

    for horizon_days in horizons:
        steps = max(1, int(horizon_days) // 7)
        # Build a flat future risk path and ensure it is long enough
        risk_seq = [last_risk_norm] * (steps + risk_order + 2)

        returns, prices = forecast_arimax(
            last_price,
            last_observation,
            params,
            risk_seq,
            horizon=steps,
        )
        cum_return = float(np.sum(returns))
        price_fc = float(prices[-1])

        resid_var = float(params.get("resid_var", 0.0))
        std_err = float(np.sqrt(max(0.0, resid_var) * steps))
        lo = cum_return - 1.96 * std_err
        hi = cum_return + 1.96 * std_err

        rows.append(
            {
                "horizon_days": int(horizon_days),
                "steps": int(steps),
                "price_forecast": price_fc,
                "cum_return": cum_return,
                "lo95": lo,
                "hi95": hi,
            }
        )

    return pd.DataFrame(rows)
