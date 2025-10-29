"""
climate_index
===============

This package contains a simple yet extensible implementation of the
``Global Climate Risk Index`` used throughout the project.  Its sole
purpose is to encapsulate the logic for computing climate‐related risk
metrics based on weather anomalies, vegetation indices and large‐scale
oscillation indicators (ENSO).  The project originally stored these
routines in other modules but certain users expect to find them under a
``climate_index`` namespace.  To preserve backwards compatibility and
provide a clear separation of concerns, this package exports three
primary modules:

* ``features`` – utilities for normalising and scaling meteorological
  time series such that they become comparable across seasons and
  locations.  This includes a seasonal z‐score transformation and a
  logistic scaling to map continuous signals into a bounded range.
* ``regional_score`` – a helper module for combining multiple normalised
  features into a single regional risk score.  It defines sensible
  defaults for factor weights but allows callers to provide bespoke
  weightings through the configuration YAML files shipped with the
  project.
* ``global_index`` – the orchestration layer which brings together
  regional scores and region weights (e.g. share of world production)
  into a single global risk index per commodity.  This module also
  computes contributions per region for transparency and downstream
  explanation.

All functions are pure and deterministic given their inputs.  They
operate on ``pandas.DataFrame`` objects indexed by date.  The API
surface has been intentionally kept minimal to ease unit testing and
avoid hidden state.  For more details on the underlying statistical
assumptions see the documentation in each submodule.
"""

from .features import seasonal_zscore, logistic_scale
from .regional_score import compute_regional_score
from .global_index import compute_global_index

__all__ = [
    "seasonal_zscore",
    "logistic_scale",
    "compute_regional_score",
    "compute_global_index",
]
