"""
market_models
=============

This package exposes a set of baseline time series modelling routines
tailored to agricultural commodities.  It offers functions for
preparing merged datasets (combining the ``silver`` data and the
computed global climate risk index), fitting simple ARIMAX‑style
regressions and generating multi‑horizon forecasts.  The goal of
these models is not to compete with state‑of‑the‑art machine learning
but to provide interpretable baselines that link climate risk to
market prices.  Users can easily swap the underlying model
implementation with their own or add additional forecasting modules.

Modules
-------

prep
    Data preparation routines: merge price and risk data, compute
    log returns, handle missing values and create exogenous matrices.
arimax
    Fit a simple ARIMAX‑like model (linear regression on the lagged
    price changes with the global risk index as an exogenous variable).
train
    Orchestrate the end‑to‑end process: read input files, call
    preprocessing, fit the model and produce forecasts at specified
    horizons.  The output can be stored as JSON for downstream
    consumption.
"""

from .prep import merge_price_and_risk
from .arimax import fit_arimax
from .train import train_and_forecast

__all__ = ["merge_price_and_risk", "fit_arimax", "train_and_forecast"]
