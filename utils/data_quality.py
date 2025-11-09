"""Utility functions for assessing and improving the quality of
financial and meteorological time‑series data.

This module centralises a collection of routines that help ensure
the reliability and interpretability of data prior to modelling,
visualisation or risk analysis.  In any quantitative finance
project, data quality is paramount: missing values, outliers or
erroneous data types can lead to spurious results and false
conclusions.  The functions defined here provide a simple yet
comprehensive toolbox for interrogating, cleaning and summarising
data frames.

Design principles
-----------------

* **Transparency** – Each function clearly states its intent and
  returns standard Python structures (mostly pandas objects) so
  intermediate outputs can be inspected.  Whenever a function
  produces a summary or report, it exposes the raw statistics
  underlying that summary.

* **Robustness** – If a third‑party library is unavailable (for
  example, ``statsmodels`` for stationarity tests), the functions
  degrade gracefully by returning ``None`` or raising a clear
  exception instead of failing silently.

* **Extensibility** – New metrics or quality checks can be added
  without altering the existing API.  The final section of the
  module demonstrates how to compose multiple checks into a single
  report.

Typical usage
-------------

Before feeding a DataFrame into a forecasting model or
visualisation pipeline you can run:

>>> from utils.data_quality import generate_quality_report
>>> report = generate_quality_report(my_dataframe)
>>> print(report["missingness"])

to inspect where missing values occur and address them early.  See
each function’s documentation below for further examples and
details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # The Augmented Dickey–Fuller test is available in statsmodels.
    from statsmodels.tsa.stattools import adfuller  # type: ignore
    _HAS_STATSMODELS = True
except Exception:
    # If statsmodels is not installed, stationarity checks will
    # gracefully degrade.
    _HAS_STATSMODELS = False


def calculate_missingness(df: pd.DataFrame) -> pd.Series:
    """Compute the proportion of missing values for each column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose missing value rates are to be
        calculated.

    Returns
    -------
    pd.Series
        A series indexed by column name containing the fraction of
        missing values in each column.  A value of ``0.0`` means
        no missing entries were found, whereas ``1.0`` indicates
        an entirely missing column.

    Examples
    --------

    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    >>> calculate_missingness(df)
    a    0.333333
    b    0.000000
    dtype: float64
    """
    return df.isna().mean()


def detect_outliers(
    df: pd.DataFrame, *, method: str = "iqr", threshold: float = 1.5
) -> pd.DataFrame:
    """Identify potential outliers in numeric columns.

    Outlier detection is important in financial and meteorological
    contexts because extreme values can disproportionately
    influence models and summaries.  Two classic methods are
    supported:

    ``'iqr'``
        Uses the interquartile range (IQR).  Observations below
        ``Q1 - threshold * IQR`` or above ``Q3 + threshold * IQR``
        are flagged as outliers.  The default threshold of 1.5
        follows the common box‑plot convention.

    ``'zscore'``
        Uses the standard score.  Observations with absolute
        z‑score greater than ``threshold`` are flagged.  A typical
        value for ``threshold`` is 3.0, corresponding to three
        standard deviations from the mean.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyse.  Only numeric columns are
        considered.
    method : {{'iqr', 'zscore'}}, optional
        Which detection method to use.  Default is ``'iqr'``.
    threshold : float, optional
        Sensitivity parameter controlling how far from the
        centre a value must be to be flagged.  Default is 1.5 for
        IQR and 3.0 for z‑score detection.

    Returns
    -------
    pd.DataFrame
        A Boolean DataFrame of the same shape as ``df`` where
        ``True`` entries indicate potential outliers.  Non‑numeric
        columns are filled with ``False``.

    Notes
    -----
    Outlier detection does not remove or modify the original data.
    Use the returned mask to decide how to handle extreme values.

    """
    numeric = df.select_dtypes(include=[np.number])
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    if numeric.empty:
        return mask
    if method == "iqr":
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outliers = (numeric < lower) | (numeric > upper)
    elif method == "zscore":
        mean = numeric.mean()
        std = numeric.std(ddof=0)
        # Avoid division by zero
        std_replaced = std.replace(0.0, np.nan)
        zscore = (numeric - mean) / std_replaced
        outliers = zscore.abs() > threshold
    else:
        raise ValueError(f"Unknown method: {method}")
    mask.loc[outliers.index, outliers.columns] = outliers
    return mask


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return common summary statistics for numeric columns.

    The output contains mean, median, standard deviation, minimum
    and maximum for each numeric column.  If the DataFrame has
    non‑numeric columns they will be ignored.  Results are
    returned in a long‑format DataFrame to facilitate merging or
    further processing.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to compute statistics.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a multi‑index of (column, statistic)
        and a single column ``value`` containing the statistic.

    Examples
    --------

    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    >>> compute_summary_statistics(df)
                   value
    column statistic      
    x      mean        2.0
           median      2.0
           std         1.0
           min         1.0
           max         3.0
    y      mean        5.0
           median      5.0
           std         1.0
           min         4.0
           max         6.0
    """
    numeric = df.select_dtypes(include=[np.number])
    stats: Dict[Tuple[str, str], float] = {}
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        stats[(col, "mean")] = float(series.mean())
        stats[(col, "median")] = float(series.median())
        stats[(col, "std")] = float(series.std(ddof=0))
        stats[(col, "min")] = float(series.min())
        stats[(col, "max")] = float(series.max())
    return pd.DataFrame.from_dict(stats, orient="index", columns=["value"])


def check_stationarity(series: pd.Series, *, alpha: float = 0.05) -> Dict[str, Optional[float]]:
    """Perform a stationarity check on a time series using the ADF test.

    Stationarity is a critical assumption for many time series models.
    This function applies the Augmented Dickey–Fuller test to assess
    whether a series has a unit root.  If the underlying library
    ``statsmodels`` is unavailable, it returns ``None`` values and
    leaves it to the caller to decide how to proceed.

    Parameters
    ----------
    series : pd.Series
        The time series to test.  Index should be date‐like but
        this is not enforced.
    alpha : float, optional
        Significance level for the test.  Default is 0.05.

    Returns
    -------
    dict
        A dictionary containing the test statistic and p‑value.
        If statsmodels is not installed, both will be ``None``.

    Notes
    -----
    A small p‑value (less than ``alpha``) leads to rejection of the
    null hypothesis of a unit root, suggesting the series is
    stationary.  This is a probabilistic test and not definitive.
    """
    if not _HAS_STATSMODELS:
        return {"test_statistic": None, "pvalue": None}
    try:
        # Drop missing values as adfuller cannot handle NaNs
        cleaned = series.dropna().astype(float)
        result = adfuller(cleaned)
        statistic, pvalue = result[0], result[1]
        return {"test_statistic": float(statistic), "pvalue": float(pvalue)}
    except Exception:
        # On any unexpected error, return Nones rather than raising
        return {"test_statistic": None, "pvalue": None}


def correlation_matrix(df: pd.DataFrame, *, method: str = "pearson") -> pd.DataFrame:
    """Compute the correlation matrix of numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.  Non‑numeric columns are ignored.
    method : {{'pearson', 'spearman', 'kendall'}}, optional
        Method used to compute correlation.  See :func:`pandas.DataFrame.corr`
        for details.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix of all numeric columns.  The
        diagonal will consist of ones by definition.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr(method=method)


def normalize_series(series: pd.Series) -> pd.Series:
    """Standardise a series to zero mean and unit variance.

    Standardisation is a common pre‑processing step which
    facilitates comparison of variables measured on different
    scales.  If the series has zero variance, the normalised
    version will be all zeros to avoid division by zero.

    Parameters
    ----------
    series : pd.Series
        The numeric series to normalise.

    Returns
    -------
    pd.Series
        The rescaled series.
    """
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def detect_trend(series: pd.Series, *, window: int = 5) -> pd.Series:
    """Estimate the underlying trend in a time series via a moving average.

    A simple yet effective way to capture trend is by computing a
    rolling mean over a fixed window.  This function returns the
    smoothed series; the difference between the original series and
    the smoothed series represents deviations from the trend.

    Parameters
    ----------
    series : pd.Series
        Input time series.  Non‑numeric values are coerced to
        ``float`` where possible.
    window : int, optional
        Size of the rolling window.  Must be a positive integer.

    Returns
    -------
    pd.Series
        The smoothed series of the same length as the input.  For
        the first ``window - 1`` values where the rolling window is
        incomplete, the mean of the available data is used.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    # Use rolling mean with min_periods=1 to avoid NaNs at start
    return series.astype(float).rolling(window, min_periods=1).mean()


def impute_missing_values(
    df: pd.DataFrame,
    *,
    strategy: str = "mean",
    fill_value: Optional[float] = None,
) -> pd.DataFrame:
    """Impute missing values in numeric columns according to a strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Data with potential missing entries.
    strategy : str, optional
        One of ``'mean'``, ``'median'``, ``'ffill'``, ``'bfill'`` or
        ``'constant'``.  Mean and median strategies compute the
        respective statistic column wise; forward/backward fill
        propagate last/next valid observation; constant uses
        ``fill_value``.
    fill_value : float, optional
        Value to use with the ``'constant'`` strategy.  Ignored
        otherwise.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with missing values replaced.  Non‑numeric
        columns are left unchanged except for forward/backward fill
        which apply to all columns.
    """
    df_imp = df.copy()
    if strategy == "mean":
        for col in df_imp.select_dtypes(include=[np.number]).columns:
            df_imp[col] = df_imp[col].fillna(df_imp[col].mean())
    elif strategy == "median":
        for col in df_imp.select_dtypes(include=[np.number]).columns:
            df_imp[col] = df_imp[col].fillna(df_imp[col].median())
    elif strategy == "ffill":
        df_imp = df_imp.fillna(method="ffill")
    elif strategy == "bfill":
        df_imp = df_imp.fillna(method="bfill")
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("fill_value must be provided for constant strategy")
        df_imp = df_imp.fillna(fill_value)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return df_imp


def validate_schema(df: pd.DataFrame, schema: Dict[str, Any]) -> bool:
    """Validate a DataFrame against a simple schema definition.

    The schema is a mapping from column names to expected dtypes.  If
    the DataFrame contains any column not listed in the schema or
    columns whose dtype does not match the expected type, the
    function returns ``False``.  Missing columns are also
    considered errors.  This lightweight validation is intended as
    a sanity check and does not enforce complex constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame to validate.
    schema : Dict[str, type]
        Mapping of expected column names to Python or numpy dtype
        objects, e.g. ``{"date": object, "price": float}``.

    Returns
    -------
    bool
        ``True`` if the DataFrame conforms to the schema, ``False``
        otherwise.
    """
    # Check for unexpected or missing columns
    df_cols = set(df.columns)
    schema_cols = set(schema.keys())
    if df_cols != schema_cols:
        return False
    for col, expected in schema.items():
        actual = df[col].dtype
        # numpy dtypes can be compared via ==
        if not np.issubdtype(actual, expected):
            return False
    return True


def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Compile a data quality report for the given DataFrame.

    The report gathers multiple diagnostics into a single dict so
    that calling code can serialise or present the results easily.
    It summarises missing values, potential outliers, summary
    statistics and the dtypes of columns.  The intention is to
    provide one central place to call when starting an analysis or
    debugging data issues.

    Parameters
    ----------
    df : pd.DataFrame
        Data to report on.

    Returns
    -------
    dict
        Nested dictionary with keys ``'missingness'``,
        ``'outliers'``, ``'summary_statistics'`` and ``'dtypes'``.  The
        outlier mask is itself a DataFrame of booleans.  The
        summary statistics are returned as a long‑format DataFrame
        described in :func:`compute_summary_statistics`.
    """
    report: Dict[str, Any] = {}
    report["missingness"] = calculate_missingness(df)
    report["outliers"] = detect_outliers(df)
    report["summary_statistics"] = compute_summary_statistics(df)
    report["dtypes"] = df.dtypes.apply(lambda dt: dt.name).to_dict()
    return report
