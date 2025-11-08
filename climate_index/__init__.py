"""
Module for building simple climate-aware indices from silver-layer datasets.

This package consumes:
- data/silver/weather/weather_anomalies.*
- data/silver/market/market_prices.*
- data/silver/agri/agri_indicators.*

and exposes functions to compute:
- regional climate stress scores
- a global climate-commodity risk index
"""

from .simple_index import build_regional_climate_score, build_global_climate_index
