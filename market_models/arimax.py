"""
market_models.arimax (extended)
--------------------------------

A more realistic yet transparent ARIMAX implementation for commodity
prices. This module keeps a light dependency footprint (NumPy/Pandas)
while adding:

• Configurable lag orders for the endogenous return (p) and the risk
  regressor (q).
• Optional seasonal component via harmonic (sin/cos) terms with a
  configurable period (weekly data ⇒ 52 by default).
• Support for additional exogenous variables passed as columns prefixed
  with `exog_`.
• Backward compatibility: the legacy `fit_arimax(df, feature_cols)`
  call still works; if `feature_cols` is omitted, features are built
  automatically from `y` and either `risk_*` lags or a base risk column.

The forecast routine maintains the same signature as before:
`forecast_arimax(last_price, last_row, params, risk_future, horizon)`.
It now also handles seasonal terms and extra exogenous variables.

This is NOT a full statistical package. There is no automatic order
selection nor full inference suite. It is intentionally simple,
explicit and auditable.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------

def _detect_base_risk_column(df: pd.DataFrame) -> Optional[str]:
    """Return the name of a base risk column if present.

    We prioritise a normalised risk column if found; otherwise we
    accept a 0..100 index and normalise it.
    """
    if "risk" in df.columns:
        return "risk"
    if "global_risk_0_100" in df.columns:
        return "global_risk_0_100"
    return None

def _ensure_risk_normalised(series: pd.Series) -> pd.Series:
    """Normalise risk to 0..1 if it looks like 0..100."""
    s = series.astype(float)
    if s.max(skipna=True) > 1.5:  # likely 0..100
        return s / 100.0
    return s

def _build_auto_features(
    df: pd.DataFrame,
    p: int = 2,
    q: int = 2,
    include_seasonal: bool = True,
    season_period: int = 52,
    extra_exog: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """Construct a design frame with lagged features and (optional) seasonals.

    Parameters
    ----------
    df : DataFrame with an index or column containing time order and at least
         the following columns:
         - `y` (weekly log return)
         - either precomputed `risk_lag*` columns OR a base risk column among
           {`risk`, `global_risk_0_100`} from which lags will be created.
    p, q : int
        Number of lags for y and risk respectively (p>=1, q>=1 recommended).
    include_seasonal : bool
        If True, add harmonic terms `season_sin`, `season_cos`.
    season_period : int
        Period of the seasonality (52 for weekly data).
    extra_exog : list[str]
        Optional names of additional exogenous columns to include
        (they will be used as-is, but we require they start with `exog_`).

    Returns
    -------
    (design_df, feature_cols, meta)
        design_df : DataFrame aligned on the valid range after lags.
        feature_cols : list of feature names in the order used for OLS.
        meta : dict with keys like `season_period` and `season_phase`.
    """
    if "y" not in df.columns:
        raise ValueError("Expected column 'y' (weekly log return) in df.")

    work = df.copy()

    # Build y lags y_lag1..y_lagp
    for k in range(1, max(1, p) + 1):
        work[f"y_lag{k}"] = work["y"].shift(k)

    # Risk lags: either use existing risk_lag* or build them from a base column
    risk_cols_present = [c for c in work.columns if c.startswith("risk_lag")]
    if len(risk_cols_present) < q:
        base = _detect_base_risk_column(work)
        if base is None and q > 0:
            raise ValueError(
                "No risk base column found ('risk' or 'global_risk_0_100') and not enough risk_lag* present."
            )
        if base is not None:
            risk_base = _ensure_risk_normalised(work[base])
            for k in range(1, max(1, q) + 1):
                work[f"risk_lag{k}"] = risk_base.shift(k)

    # Seasonality via harmonic terms
    feature_cols: List[str] = []
    for k in range(1, max(1, p) + 1):
        feature_cols.append(f"y_lag{k}")
    for k in range(1, max(1, q) + 1):
        feature_cols.append(f"risk_lag{k}")

    season_meta: Dict[str, float] = {"season_period": float(season_period), "season_phase": 0.0}

    if include_seasonal:
        # Determine a phase; if index is datetime-like, use its position within the period
        if isinstance(work.index, pd.DatetimeIndex):
            # Map each date to an integer step; take the last one as phase
            # We use ISO week number modulo season_period for weekly data
            phase_vals = (work.index.isocalendar().week.astype(int) % season_period).to_numpy()
            last_phase = int(phase_vals[-1]) if len(phase_vals) else 0
        else:
            last_phase = (len(work) - 1) % season_period
        season_meta.update({"season_phase": float(last_phase)})
        # Harmonics (k=1). You can extend to multiple harmonics if desired.
        t = np.arange(len(work))
        omega = 2.0 * math.pi / float(season_period)
        work["season_sin"] = np.sin(omega * t)
        work["season_cos"] = np.cos(omega * t)
        feature_cols += ["season_sin", "season_cos"]

    # Extra exogenous variables
    if extra_exog:
        for name in extra_exog:
            if not name.startswith("exog_"):
                raise ValueError("Extra exogenous columns must start with 'exog_'.")
            if name not in work.columns:
                raise ValueError(f"Extra exogenous column '{name}' not found in df.")
            feature_cols.append(name)

    # Drop rows with NaNs created by shifting
    design = work.dropna(subset=["y"] + feature_cols).copy()

    return design, feature_cols, season_meta

# ---------------------------------------------------------------------
# Estimation (OLS) and forecasting
# ---------------------------------------------------------------------

def _ols_fit(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return OLS coefficients and residual variance.

    Residual variance is SSE / (n - k).
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    n, k = X.shape
    resid_var = float(resid.T @ resid / max(1, n - k))
    return beta, resid_var


def fit_arimax(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    *,
    p: int = 2,
    q: int = 2,
    include_seasonal: bool = True,
    season_period: int = 52,
    extra_exog: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Fit an extended ARIMAX by OLS.

    Backward compatible signature: if `feature_cols` is provided, we use
    those columns directly (legacy mode). Otherwise we auto‑build
    features with orders (p, q) and optional seasonality.

    Returns a dict with keys: `intercept`, `coeffs`, `resid_var`,
    `feature_cols`, `season_meta`, and optionally `exog_last`.
    """
    if feature_cols is None:
        design, feat, season_meta = _build_auto_features(
            df, p=p, q=q, include_seasonal=include_seasonal,
            season_period=season_period, extra_exog=extra_exog,
        )
        used_cols = feat
    else:
        # Legacy mode: trust provided columns; do not alter df.
        design = df.dropna(subset=["y"] + feature_cols).copy()
        used_cols = list(feature_cols)
        # Best effort: infer reasonable season meta (no seasonal forecast unless provided)
        season_meta = {"season_period": float(season_period), "season_phase": 0.0}

    X = design[used_cols].to_numpy()
    y = design["y"].to_numpy()
    X_design = np.column_stack([np.ones(len(design)), X])

    beta, resid_var = _ols_fit(y, X_design)
    intercept = float(beta[0])
    coeff_map = {used_cols[i]: float(beta[i + 1]) for i in range(len(used_cols))}

    # Keep last values of any extra exogenous for forecasting fallback
    exog_last = {c: float(design[c].iloc[-1]) for c in used_cols if c.startswith("exog_")}

    return {
        "intercept": intercept,
        "coeffs": coeff_map,
        "resid_var": resid_var,
        "feature_cols": used_cols,
        "season_meta": season_meta,
        "exog_last": exog_last,
    }


def forecast_arimax(
    last_price: float,
    last_row: pd.Series,
    params: Dict[str, any],
    risk_future: List[float],
    horizon: int,
) -> Tuple[List[float], List[float]]:
    """Recursive multi‑step forecast using the fitted extended ARIMAX.

    • `risk_future` should be on a 0..1 scale if your model was trained
      on a normalised risk. If your training used 0..100, pass the same
      scale here – the lags are handled mechanically.
    • Seasonal terms are generated using the stored `season_meta`.
    • Extra exog (`exog_*`) are held constant at their last observed
      values unless present in `last_row`.
    """
    intercept = params["intercept"]
    coeffs = params["coeffs"]
    feature_cols = params["feature_cols"]
    season_meta = params.get("season_meta", {"season_period": 52.0, "season_phase": 0.0})
    season_period = int(season_meta.get("season_period", 52.0))
    phase = float(season_meta.get("season_phase", 0.0))

    current = last_row.copy().to_dict()

    # Ensure risk_future long enough
    if len(risk_future) < horizon:
        risk_future = list(risk_future) + [risk_future[-1]] * (horizon - len(risk_future))

    omega = 2.0 * math.pi / float(max(1, season_period))
    forecasts: List[float] = []
    prices: List[float] = []
    price = float(last_price)

    for step in range(horizon):
        X_vals: List[float] = []
        for col in feature_cols:
            if col.startswith("y_lag"):
                k = int(col.split("lag")[1])
                X_vals.append(float(current.get(f"y_lag{k}", 0.0)))
            elif col.startswith("risk_lag"):
                k = int(col.split("lag")[1])
                idx = step + (k - 1)
                X_vals.append(float(risk_future[idx]))
            elif col == "season_sin":
                X_vals.append(math.sin(omega * (phase + step + 1)))
            elif col == "season_cos":
                X_vals.append(math.cos(omega * (phase + step + 1)))
            elif col.startswith("exog_"):
                X_vals.append(float(current.get(col, params.get("exog_last", {}).get(col, 0.0))))
            else:
                X_vals.append(float(current.get(col, 0.0)))

        y_hat = float(intercept + sum(coeffs[c] * X_vals[i] for i, c in enumerate(feature_cols)))
        forecasts.append(y_hat)

        price = price * math.exp(y_hat)
        prices.append(price)

        # Update lag structure for next step
        for col in feature_cols:
            if col.startswith("y_lag"):
                k = int(col.split("lag")[1])
                if k == 1:
                    current[col] = y_hat
                else:
                    prev = f"y_lag{k-1}"
                    current[col] = current.get(prev, 0.0)
            elif col.startswith("risk_lag"):
                k = int(col.split("lag")[1])
                if k == 1:
                    current[col] = risk_future[step]
                else:
                    prev = f"risk_lag{k-1}"
                    current[col] = current.get(prev, risk_future[step])
        # Season phase advances implicitly via (phase + step + 1)

    return forecasts, prices
