"""
climate_index.regional_score
---------------------------

The purpose of this module is to combine multiple normalised 
meteorological indicators into a single
regional risk score.  A regional risk score quantifies the degree of
climate stress experienced by a given region at a particular time and
is computed as a weighted linear combination of z‑scored signals
followed by a logistic scaling.  

As inputs, the main function expects a ``pandas.DataFrame`` with
columns corresponding to normalised anomalies (e.g. ``temp_anom``,
``precip_anom``, ``ndvi``, ``enso``).  The
``factor_weights`` argument is a simple mapping from column name to
weight. 

The output is a ``pandas.Series`` indexed like the input DataFrame
whose values lie in the range [0, 100].  A higher score indicates
greater climate stress.


"""

from __future__ import annotations

from typing import Dict, Iterable
import pandas as pd
import numpy as np
from .features import logistic_scale


def compute_regional_score(
    df: pd.DataFrame,
    factor_weights: Dict[str, float],
    *,
    scale_a: float = 1.0,
    scale_b: float = 0.0,
) -> pd.Series:
    """Compute a regional climate risk score.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame with columns corresponding to normalised anomalies
        (e.g. ``temp_anom``, ``precip_anom``, ``ndvi``, ``enso``).  All
        columns specified in ``factor_weights`` should exist in
        ``df``.  Missing columns are silently ignored.
    factor_weights : dict[str, float]
        Mapping from column names to their relative importance.  The
        weights do not have to sum to one as they are internally
        normalised.
    scale_a : float, default 1.0
        Steepness parameter for the logistic scaling.  See
        :func:`climate_index.features.logistic_scale`.
    scale_b : float, default 0.0
        Midpoint parameter for the logistic scaling.

    Returns
    -------
    pandas.Series
        A series of shape ``(len(df),)`` containing the regional
        climate risk scores in the range [0, 100].

    """
    # Filter weights to those actually present in df
    present = {k: v for k, v in factor_weights.items() if k in df.columns}
    if not present:
        raise ValueError(
            "None of the factor_weights keys are present in the DataFrame columns."
        )
    # Normalise weights so that they sum to one
    total_weight = sum(present.values())
    if total_weight == 0:
        raise ValueError("Sum of factor_weights must be non-zero")
    weights = {k: v / total_weight for k, v in present.items()}
    # Compute weighted sum of anomalies
    combined = sum(df[k] * w for k, w in weights.items())
    combined.name = "combined_anomaly"
    # Apply logistic scaling to map to 0..100
    return logistic_scale(combined, a=scale_a, b=scale_b)
