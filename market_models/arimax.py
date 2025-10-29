"""
market_models.arimax
--------------------

This module implements a very simple autoregressive integrated model
with an exogenous regressor (ARIMAX) tailored to commodity price
series.  It eschews dependencies on heavy statistical packages by
using NumPy to perform ordinary least squares estimation.  The model
assumes that the target series ``y`` (the log return of the price) is
related to its own lagged values and lagged values of an exogenous
risk index.  In matrix form::

    y_t = β_0 + β_1 y_{t-1} + … + β_p y_{t-p} + γ_1 r_{t-1} + … + γ_q r_{t-q} + ε_t

where the coefficients β_i and γ_i are estimated via least squares.

The functions provided here are intentionally simple and do not
include automatic order selection.  They are meant for educational
purposes and to serve as a baseline against which more elaborate
models may be compared.  See :mod:`market_models.train` for an
orchestration function that uses these routines to produce
multi‑horizon forecasts.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def fit_arimax(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, any]:
    """Fit a simple ARIMAX model using ordinary least squares.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame containing the target column ``y`` and the exogenous
        features specified in ``feature_cols``.  The data should
        already be cleaned of NaNs.
    feature_cols : list[str]
        Names of the exogenous and lagged endogenous columns used as
        predictors.  The order of columns determines the order of
        coefficients in the result.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``intercept``: the estimated intercept β_0
        - ``coeffs``: mapping from feature name to coefficient
        - ``resid_var``: variance of residuals (σ²)
        - ``feature_cols``: the list of feature columns for reference

    Notes
    -----
    We solve (X'X)β = X'y using ``numpy.linalg.lstsq``.  The residual
    variance is estimated as the sum of squared residuals divided by
    (n – k), where ``n`` is the number of observations and ``k`` the
    number of parameters.
    """
    # Design matrix with intercept
    X = df[feature_cols].values
    y = df["y"].values
    n_obs, n_feat = X.shape
    X_design = np.column_stack([np.ones(n_obs), X])
    # Solve for beta
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    intercept = beta[0]
    coeffs = {feature_cols[i]: beta[i + 1] for i in range(n_feat)}
    # Compute residual variance
    y_hat = X_design @ beta
    resid = y - y_hat
    resid_var = np.dot(resid, resid) / max(1, n_obs - (n_feat + 1))
    return {
        "intercept": float(intercept),
        "coeffs": coeffs,
        "resid_var": float(resid_var),
        "feature_cols": feature_cols,
    }


def forecast_arimax(
    last_price: float,
    last_row: pd.Series,
    params: Dict[str, any],
    risk_future: List[float],
    horizon: int,
) -> Tuple[List[float], List[float]]:
    """Generate multi‑step forecasts using a fitted ARIMAX model.

    Parameters
    ----------
    last_price : float
        The most recent observed price.  This serves as the base for
        constructing price forecasts.
    last_row : pandas.Series
        A series containing the most recent values of the target
        (``y``) and the lagged features used in the model.  It must
        have at least the columns appearing in ``params['feature_cols']``.
    params : dict
        The model parameters returned by :func:`fit_arimax`.
    risk_future : list[float]
        A sequence of future risk values (normalised between 0 and 1).
        Its length should be at least equal to ``horizon``.  These
        values will be used to populate the risk lags for forecasting.
    horizon : int
        Number of periods ahead to forecast.  The returned lists will
        have this length.

    Returns
    -------
    (returns, prices) : (list[float], list[float])
        A pair of lists where ``returns[i]`` is the predicted log
        return at step i+1 and ``prices[i]`` is the predicted price
        level at that same step.

    Notes
    -----
    This function performs a simple recursive forecast: predicted
    returns are fed back into the lag structure for the next step.  The
    exogenous risk sequence is assumed known in advance.  If the
    supplied ``risk_future`` is shorter than ``horizon`` the remaining
    values will be filled with the last available risk level.
    """
    # Prepare data
    intercept = params["intercept"]
    coeffs = params["coeffs"]
    feature_cols = params["feature_cols"]
    # Extract current lag values
    current_lags = last_row.copy().to_dict()
    # Ensure risk_future length
    if len(risk_future) < horizon:
        risk_future = risk_future + [risk_future[-1]] * (horizon - len(risk_future))
    forecasts = []
    prices = []
    price = last_price
    for step in range(horizon):
        # Build feature vector for this step
        X = []
        for col in feature_cols:
            if col.startswith("risk_lag"):
                lag_no = int(col.split("lag")[1])
                # Use future risk series shifted by (lag_no-1) from current step
                idx = step + (lag_no - 1)
                X.append(risk_future[idx])
            elif col.startswith("y_lag"):
                lag_no = int(col.split("lag")[1])
                key = f"y_lag{lag_no}"
                X.append(current_lags.get(key, 0.0))
            else:
                X.append(current_lags.get(col, 0.0))
        # Compute predicted return
        y_hat = intercept + sum(coeffs[col] * X[i] for i, col in enumerate(feature_cols))
        forecasts.append(float(y_hat))
        # Update price
        price = price * np.exp(y_hat)
        prices.append(float(price))
        # Update lag structure for next iteration
        # Shift existing y lags
        for col in feature_cols:
            if col.startswith("y_lag"):
                lag_no = int(col.split("lag")[1])
                if lag_no == 1:
                    current_lags[col] = y_hat
                else:
                    # y_lag2 takes previous y_lag1, etc.
                    prev_col = f"y_lag{lag_no - 1}"
                    current_lags[col] = current_lags.get(prev_col, 0.0)
            elif col.startswith("risk_lag"):
                lag_no = int(col.split("lag")[1])
                # risk_lag1 is risk_future at step, risk_lag2 is previous step, etc.
                if lag_no == 1:
                    current_lags[col] = risk_future[step]
                else:
                    prev_col = f"risk_lag{lag_no - 1}"
                    current_lags[col] = current_lags.get(prev_col, risk_future[step])
    return forecasts, prices
