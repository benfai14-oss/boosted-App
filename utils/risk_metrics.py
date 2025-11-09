"""
Risk metrics utilities for evaluating the performance of commodities and hedging strategies.

This module provides a collection of functions to compute common risk metrics used
in financial analysis, including value-at-risk, conditional value-at-risk (expected
shortfall), beta coefficients, Sharpe ratio, Sortino ratio and maximum drawdown.

These functions operate on sequences of returns or price levels and avoid any use
of machine learning. They are intended to be used by the forecasting and hedging
layers to assess the risk-adjusted performance of strategies.

All functions are implemented using numpy and pandas; they provide sensible
defaults and handle missing data gracefully by automatically dropping NaNs.
"""

from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
import pandas as pd


def _to_returns_array(returns: Sequence[float]) -> np.ndarray:
    """Convert a sequence of returns to a NumPy array while dropping NaNs.

    Parameters
    ----------
    returns : sequence of float
        The input return series.

    Returns
    -------
    numpy.ndarray
        Array of valid returns (NaNs removed).
    """
    arr = np.asarray(list(returns), dtype=float)
    arr = arr[~np.isnan(arr)]
    return arr


def value_at_risk(returns: Sequence[float], confidence_level: float = 0.95) -> float:
    """Compute the Value at Risk (VaR) of a return series.

    VaR measures the maximum expected loss over a specified time horizon
    at a given confidence level. For example, a 95% VaR of -5% means that
    in 95% of cases the loss should not exceed 5%.

    Parameters
    ----------
    returns : sequence of float
        Return series (fractions, not percentages). For example, 0.01 for 1%.
    confidence_level : float, default 0.95
        Confidence level for VaR calculation.

    Returns
    -------
    float
        The VaR as a negative number representing potential loss. If no data is
        available, returns 0.0.

    Notes
    -----
    This implementation uses the historical simulation approach: the
    (1 - confidence_level) percentile of the return distribution.
    """
    arr = _to_returns_array(returns)
    if arr.size == 0:
        return 0.0
    var = np.percentile(arr, (1 - confidence_level) * 100)
    return float(var)


def conditional_value_at_risk(returns: Sequence[float], confidence_level: float = 0.95) -> float:
    """Compute the Conditional Value at Risk (CVaR) or Expected Shortfall.

    CVaR is the expected loss given that the loss exceeds the VaR threshold.
    It provides additional insight into tail risk beyond the single point
    estimate provided by VaR.

    Parameters
    ----------
    returns : sequence of float
        Return series as fractions.
    confidence_level : float, default 0.95
        Confidence level used to determine the tail threshold.

    Returns
    -------
    float
        The CVaR (negative number). If not enough data, returns the VaR.

    Notes
    -----
    - CVaR captures the average of the worst (1 - confidence_level) percent
      of losses.
    - If no losses exceed the VaR, CVaR equals VaR.
    """
    arr = _to_returns_array(returns)
    if arr.size == 0:
        return 0.0
    threshold = np.percentile(arr, (1 - confidence_level) * 100)
    tail_losses = arr[arr <= threshold]
    if tail_losses.size == 0:
        return float(threshold)
    return float(tail_losses.mean())


def beta(returns: Sequence[float], benchmark_returns: Sequence[float]) -> float:
    """Compute the beta coefficient of an asset relative to a benchmark.

    Beta measures the sensitivity of an asset's returns to the returns of
    a benchmark. A beta greater than 1 indicates that the asset is more
    volatile than the benchmark; a beta between 0 and 1 indicates it is
    less volatile.

    Parameters
    ----------
    returns : sequence of float
        Returns of the asset.
    benchmark_returns : sequence of float
        Returns of the benchmark.

    Returns
    -------
    float
        The beta coefficient. Returns 0.0 if there are fewer than two
        observations or if the benchmark variance is zero.
    """
    x = _to_returns_array(returns)
    y = _to_returns_array(benchmark_returns)
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x = x[:n]
    y = y[:n]
    cov = np.cov(x, y, ddof=1)[0, 1]
    var_bench = np.var(y, ddof=1)
    if var_bench == 0.0:
        return 0.0
    return float(cov / var_bench)


def sharpe_ratio(returns: Sequence[float], risk_free_rate: float = 0.0) -> float:
    """Calculate the annualised Sharpe ratio of a return series.

    The Sharpe ratio measures the excess return per unit of volatility.
    Higher values indicate better risk-adjusted performance.

    Parameters
    ----------
    returns : sequence of float
        Return series as fractions.
    risk_free_rate : float, default 0.0
        Risk-free rate of return (also a fraction).

    Returns
    -------
    float
        The annualised Sharpe ratio. Returns 0.0 if the standard deviation
        of returns is zero or there is no data.

    Notes
    -----
    - Annualisation assumes 252 trading periods per year.
    - Excess return is calculated as return minus risk-free rate.
    """
    arr = _to_returns_array(returns)
    if arr.size == 0:
        return 0.0
    excess = arr - risk_free_rate
    mean_excess = excess.mean()
    std_excess = excess.std(ddof=1)
    if std_excess == 0.0:
        return 0.0
    sharpe = mean_excess / std_excess
    return float(sharpe * np.sqrt(252))


def sortino_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    target: float = 0.0,
) -> float:
    """Compute the annualised Sortino ratio of a return series.

    The Sortino ratio penalises only downside volatility (returns below a
    target threshold), making it more appropriate when upside volatility
    should not be considered harmful.

    Parameters
    ----------
    returns : sequence of float
        Returns as fractions.
    risk_free_rate : float, default 0.0
        Risk-free rate of return.
    target : float, default 0.0
        Target return threshold below which volatility is penalised.

    Returns
    -------
    float
        The annualised Sortino ratio. Returns 0.0 if there is no downside
        deviation or insufficient data.
    """
    arr = _to_returns_array(returns)
    if arr.size == 0:
        return 0.0
    excess = arr - risk_free_rate
    mean_excess = excess.mean()
    downside = arr[arr < target] - target
    if downside.size == 0:
        return 0.0
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0.0:
        return 0.0
    ratio = mean_excess / downside_dev
    return float(ratio * np.sqrt(252))


def max_drawdown(price_series: Sequence[float]) -> Tuple[float, float]:
    """Compute the maximum drawdown and its duration from a price series.

    Parameters
    ----------
    price_series : sequence of float
        Prices or cumulative returns.

    Returns
    -------
    tuple of (float, float)
        A tuple containing the maximum drawdown (negative value) and
        the drawdown duration expressed as a fraction of the series length.

    Notes
    -----
    Drawdown is calculated relative to the cumulative maximum of the series,
    and duration is the proportion of periods spent below the previous peak.
    """
    prices = np.asarray(list(price_series), dtype=float)
    if prices.size == 0:
        return (0.0, 0.0)
    cummax = np.maximum.accumulate(prices)
    drawdowns = (prices - cummax) / cummax
    max_dd = drawdowns.min()
    below_peak = (prices < cummax).sum()
    duration = below_peak / len(prices)
    return float(max_dd), float(duration)

# --- Compatibility aliases for hedging module ---
def calculate_var(returns, alpha: float = 0.05) -> float:
    """Alias for value_at_risk for backward compatibility."""
    return abs(value_at_risk(returns, confidence_level=1 - alpha))


def calculate_cvar(returns, alpha: float = 0.05) -> float:
    """Alias for conditional_value_at_risk for backward compatibility."""
    return abs(conditional_value_at_risk(returns, confidence_level=1 - alpha))
