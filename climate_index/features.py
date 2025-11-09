"""
climate_index.features
----------------------

This module transforms raw
meteorological signals into a bounded index suitable for
aggregation into a climate risk index.  The functions defined here
operate on ``pandas.Series`` objects and return new series aligned on
the same index.  They are deliberately free of side effects so that
callers may chain them together or embed them inside pipelines without
surprise.

The two core transformations implemented are:

* ``seasonal_zscore`` – computes a z‑score within a seasonal period
  (e.g. week of year).  This transformation removes persistent
  seasonal patterns from the data.  By default the seasonal
  grouping variable is the ISO week number, but it can be changed via
  the ``season_col`` argument.
* ``logistic_scale`` – maps a continuous signal onto the interval
  [0, 100] via a logistic (sigmoid) function.  The slope and midpoint of
  the logistic curve may be adjusted through the ``a`` and ``b``
  parameters.

"""

from __future__ import annotations

from typing import Optional, Union
import pandas as pd
import numpy as np

def seasonal_zscore(
    series: pd.Series,
    *,
    season_col: Optional[str] = None,
    group_by: str = "weekofyear",
    ddof: int = 0,
) -> pd.Series:
    """Compute a z‑score within each seasonal group.

    Parameters
    ----------
    series : pandas.Series
        The input time series.  Its index must be a ``DatetimeIndex``.
    season_col : str, optional
        Name of a precomputed seasonal grouping column in ``series``.  If
        provided, this column will be used to group observations.
        Otherwise the ``group_by`` argument determines the grouping.
    group_by : {"weekofyear", "month"}, default "weekofyear"
        If ``season_col`` is not supplied, this argument controls how
        the index is grouped into seasons.  By default observations
        sharing the same ISO week number belong to the same group.
    ddof : int, default 0
        Degrees of freedom used when normalising by the standard
        deviation.  The default ``0`` normalises by the population
        standard deviation. 

    Returns
    -------
    pandas.Series
        A new series with the same index as ``series`` containing the
        z‑scores.  Observations in groups of length one will result in
        NaN since a standard deviation cannot be computed.

    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series.index must be a DatetimeIndex")

    df = series.to_frame(name="value").copy()
    if season_col is not None:
        # Use an existing grouping column
        df["_season"] = df[season_col]
    else:
        if group_by == "weekofyear":
            df["_season"] = df.index.isocalendar().week
        elif group_by == "month":
            df["_season"] = df.index.month
        else:
            raise ValueError(f"Unsupported group_by: {group_by}")

    # Compute group mean and std
    grouped = df.groupby("_season")
    mean = grouped["value"].transform("mean")
    std = grouped["value"].transform("std", ddof=ddof)
    # Avoid division by zero: where std==0, result is NaN
    z = (df["value"] - mean) / std
    z.name = series.name
    return z


def logistic_scale(
    x: Union[pd.Series, np.ndarray, float],
    *,
    a: float = 1.0,
    b: float = 0.0,
    lower: float = 0.0,
    upper: float = 100.0,
) -> Union[pd.Series, np.ndarray, float]:
    """Scale a real‑valued input using a logistic function.

    The logistic function is defined as::

        f(x) = lower + (upper - lower) / (1 + exp(-a * (x - b)))

    where ``a`` controls the slope and ``b`` controls the midpoint.
    Larger values of ``a`` produce a steeper transition between the
    lower and upper asymptotes, while ``b`` shifts the function
    horizontally.

    Parameters
    ----------
    x : array‑like or scalar
        Input values to transform.  Can be a ``pandas.Series``,
        ``numpy.ndarray`` or scalar float.
    a : float, default 1.0
        Steepness parameter controlling how quickly the function
        saturates.  A larger ``a`` results in a steeper curve.
    b : float, default 0.0
        Midpoint parameter.  The logistic function will map
        ``x=b`` to the midpoint between ``lower`` and ``upper``.
    lower : float, default 0.0
        Lower bound of the output range.
    upper : float, default 100.0
        Upper bound of the output range.

    Returns
    -------
    same type as input
        The transformed values.  The return type matches the type of
        ``x`` – if the input is a ``Series`` the output will be a
        ``Series`` with the same index.  Scalars and arrays are
        returned as their original types.
    """
    # Convert to NumPy for the computation
    is_series = isinstance(x, pd.Series)
    arr = x.to_numpy() if is_series else np.asarray(x)
    y = lower + (upper - lower) / (1.0 + np.exp(-a * (arr - b)))
    # Cast back to original type
    if is_series:
        return pd.Series(y, index=x.index, name=x.name)
    if np.isscalar(x):
        return float(y)  # ensure Python scalar return
    return y
