"""Top‑level module for utility functions.

The :mod:`utils` package collects common functionality used across
the global climate hedging project.  By exposing submodules at
package level, you can import them succinctly:

>>> from utils import risk_metrics, scenario, data_quality, cache

This file also documents the purpose of each utility module.
"""

from . import risk_metrics  # noqa: F401
from . import scenario  # noqa: F401
from . import data_quality  # noqa: F401
from . import cache  # noqa: F401
from . import documentation  # noqa: F401  # documentation module defines a long docstring

__all__ = [
    "risk_metrics",
    "scenario",
    "data_quality",
    "cache",
    "documentation",
]
