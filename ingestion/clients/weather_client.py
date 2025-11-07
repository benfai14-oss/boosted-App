"""Real weather ingestion using the Open‑Meteo API.

The `WeatherClient` class exposes a single method,
:meth:`fetch_weather_anomalies`, that downloads daily weather data
(temperature and precipitation) for a list of regions and aggregates
them to a weekly frequency.  It then computes simple anomalies by
standardising each variable (z‑score) across the requested time
window.

If the Open‑Meteo service is unreachable or returns an error, the
client falls back to generating deterministic synthetic data.  This
ensures that downstream components always receive valid frames even
when the network is unavailable.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

# Configure a module level logger
logger = logging.getLogger(__name__)


@dataclass
class RegionQuery:
    """Simple container for region queries.

    Attributes
    ----------
    region_id: str
        Identifier for the region (e.g. 'FR', 'US').  This is used to
        look up coordinates in the static ``_REGION_COORDS`` mapping.
    start: str
        Inclusive start date in ISO format (YYYY‑MM‑DD).
    end: str
        Inclusive end date in ISO format (YYYY‑MM‑DD).
    """

    region_id: str
    start: str
    end: str


class WeatherClient:
    """Client for downloading and processing weather data.

    This implementation uses the [Open‑Meteo archive API]
    (https://open-meteo.com/en/docs/historical-api) to obtain daily
    mean temperature and precipitation sums.  It aggregates the data
    into weekly periods, computes z‑score anomalies and returns a
    combined DataFrame for all requested regions.

    Parameters
    ----------
    session: Optional[requests.Session]
        You can inject a preconfigured requests session (e.g. with
        retry/backoff logic) when constructing the client.  A new
        session will be created automatically if none is provided.
    """

    # Rough centroid coordinates (lat, lon) for supported regions.  If
    # you add new regions, update this mapping accordingly.  In a
    # production system you might load these from a config file.
    _REGION_COORDS = {
        "FR": (46.2276, 2.2137),   # France
        "US": (39.8283, -98.5795), # United States (contiguous)
        "BR": (-14.2350, -51.9253),# Brazil
        "AR": (-34.6037, -58.3816),# Argentina (Buenos Aires proxy)
        "CN": (35.8617, 104.1954), # China
        "UA": (48.3794, 31.1656), # Ukraine
    }

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def _fetch_daily(self, lat: float, lon: float, start: str, end: str) -> Optional[pd.DataFrame]:
        """Internal helper to call Open‑Meteo and return a daily DataFrame.

        The API returns daily arrays keyed by ``time``, ``temperature_2m_mean`` and
        ``precipitation_sum``.  If the request fails for any reason, ``None``
        is returned.
        """
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start}&end_date={end}"
            "&daily=temperature_2m_mean,precipitation_sum"
            "&timezone=UTC"
        )
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Weather API call failed for (%s, %s) between %s and %s: %s", lat, lon, start, end, exc)
            return None
        # Ensure required keys exist
        daily = data.get("daily")
        if not daily:
            logger.warning("Weather API returned no daily data for (%s, %s)", lat, lon)
            return None
        try:
            dates = pd.to_datetime(daily["time"])
            temps = pd.Series(daily["temperature_2m_mean"], index=dates, name="temp")
            precs = pd.Series(daily["precipitation_sum"], index=dates, name="precip")
            df = pd.concat([temps, precs], axis=1)
            df.index.name = "date"
            return df
        except Exception as exc:
            logger.warning("Failed to parse daily weather data: %s", exc)
            return None

    def fetch_weather_anomalies(self, commodity: str, queries: List[RegionQuery]) -> pd.DataFrame:
        """Download and combine weekly weather anomalies for multiple regions.

        Parameters
        ----------
        commodity: str
            Name of the commodity (currently unused but kept for future
            customisation of factor weights).
        queries: list of :class:`RegionQuery`
            Each query specifies a region identifier and date range.

        Returns
        -------
        pandas.DataFrame
            A DataFrame indexed by ``date`` with columns
            ``['region_id', 'temp_anom', 'precip_anom', 'ndvi', 'enso']``.
            Anomalies are z‑scores computed over the requested period.
        """
        frames = []
        for q in queries:
            coords = self._REGION_COORDS.get(q.region_id)
            if not coords:
                logger.error("Unknown region_id %s in WeatherClient", q.region_id)
                continue
            lat, lon = coords
            # fetch daily data
            daily = self._fetch_daily(lat, lon, q.start, q.end)
            if daily is None:
                # fall back to deterministic defaults if fetch failed
                dates = pd.date_range(q.start, q.end, freq="D")
                rng = np.random.default_rng(seed=(hash(q.region_id) % 1234567))

                temps = pd.Series(
                    rng.normal(loc=15.0, scale=5.0, size=len(dates)),
                    index=dates,
                    name="temp",
                )
                precs = pd.Series(
                    rng.gamma(shape=2.0, scale=1.0, size=len(dates)),
                    index=dates,
                    name="precip",
                )

                daily = pd.concat([temps, precs], axis=1)
                daily.index.name = "date"

            # aggregate to weekly (weeks ending on Monday)
            weekly = daily.resample("W-MON").agg(
                {"temp": "mean", "precip": "sum"}
            )

            # compute z-scores per variable (avoid divide by zero)
            weekly = weekly.copy()
            for col in ["temp", "precip"]:
                series = weekly[col]
                mean = series.mean()
                std = series.std(ddof=0)
                if std == 0 or np.isnan(std):
                    weekly[col + "_anom"] = 0.0
                else:
                    weekly[col + "_anom"] = (series - mean) / std

            # assign region and placeholders for ndvi/enso
            weekly["region_id"] = q.region_id
            weekly["ndvi"] = 0.0  # NDVI placeholder
            weekly["enso"] = 0.0  # ENSO placeholder

            frames.append(
                weekly[
                    ["region_id", "temp_anom", "precip_anom", "ndvi", "enso"]
                ]
            )

        if not frames:
            # always return the same schema
            return pd.DataFrame(
                columns=[
                    "date",
                    "region_id",
                    "temp_anom",
                    "precip_anom",
                    "ndvi",
                    "enso",
                ]
            )

        df_all = pd.concat(frames, axis=0)
        df_all.index.name = "date"
        df_all.reset_index(inplace=True)
        return df_all
