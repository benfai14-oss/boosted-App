"""Client package for fetching raw data.

This package exposes classes that wrap external data sources and return
`pandas.DataFrame` objects with a normalised schema.  Each client is
responsible for handling network errors gracefully and producing
deterministic fallback data when external services are unavailable.

The clients currently implemented are:

* :class:`WeatherClient` – downloads daily weather data from the Open‑Meteo
  archive API and computes weekly anomalies.
* :class:`MarketClient` – retrieves front‑month futures price history
  from Yahoo! Finance.
* :class:`AgriClient` – obtains high‑level agricultural production
  indicators from the World Bank API (as a placeholder).

You can add additional clients here (e.g. for sentiment, shipping or
satellite data) and integrate them into the ingestion pipeline.
"""

from .weather_client import WeatherClient
from .market_client import MarketClient
from .agri_client import AgriClient

__all__ = ["WeatherClient", "MarketClient", "AgriClient"]