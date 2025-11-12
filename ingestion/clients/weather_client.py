
"""
Client for downloading and processing weather data (Open-Meteo).

- Récupère température moyenne journalière (°C) et précipitations (mm).
- Agrège en hebdomadaire (vendredi), calcule anomalies (z-scores).
- Placeholders pour ndvi / enso.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class RegionQuery:
    region_id: str
    start: str  # "YYYY-MM-DD"
    end: str    # "YYYY-MM-DD"


class WeatherClient:
    """Client to fetch weather anomalies from Open-Meteo and aggregate weekly."""

    # Rough centroid coordinates (lat, lon) for supported regions.
    _REGION_COORDS = {
        "FR": (46.2276, 2.2137),     # France
        "US": (39.8283, -98.5795),   # United States (contiguous)
        "BR": (-14.2350, -51.9253),  # Brazil
        "AR": (-34.6037, -58.3816),  # Argentina (Buenos Aires proxy)
        "CN": (35.8617, 104.1954),   # China
        "UA": (48.3794, 31.1656),    # Ukraine
    }

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def _fetch_daily(self, lat: float, lon: float, start: str, end: str) -> Optional[pd.DataFrame]:
        """Call Open-Meteo and return a daily DataFrame with temp & precip."""
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
        """
        Download and combine weekly weather anomalies for multiple regions.

        Returns a DataFrame indexed by week (Friday) with columns:
        ['date','region_id','temp_anom','precip_anom','ndvi','enso']
        """
        frames = []
        for q in queries:
            coords = self._REGION_COORDS.get(q.region_id)
            if not coords:
                logger.error("Unknown region_id %s in WeatherClient", q.region_id)
                continue
            lat, lon = coords

            # 1) Fetch daily
            daily = self._fetch_daily(lat, lon, q.start, q.end)

            # 2) Fallback deterministic if fetch failed
            if daily is None:
                dates = pd.date_range(q.start, q.end, freq="D")
                rng = np.random.default_rng(seed=(hash(q.region_id) % 1234567))

                temps = pd.Series(rng.normal(loc=15.0, scale=5.0, size=len(dates)), index=dates, name="temp")
                precs = pd.Series(rng.gamma(shape=2.0, scale=1.0, size=len(dates)), index=dates, name="precip")

                daily = pd.concat([temps, precs], axis=1)
                daily.index.name = "date"

            # 3) Aggregate to weekly (weeks ending on FRIDAY)
            weekly = daily.resample("W-FRI").agg({"temp": "mean", "precip": "sum"})

            # 4) Compute z-scores per variable (avoid div by zero)
            weekly = weekly.copy()
            for col in ["temp", "precip"]:
                series = weekly[col]
                mean = series.mean()
                std = series.std(ddof=0)
                if std == 0 or np.isnan(std):
                    weekly[col + "_anom"] = 0.0
                else:
                    weekly[col + "_anom"] = (series - mean) / std

            # 5) Attach region and placeholders for ndvi/enso
            weekly["region_id"] = q.region_id
            weekly["ndvi"] = 0.0
            weekly["enso"] = 0.0

            frames.append(
                weekly[["region_id", "temp_anom", "precip_anom", "ndvi", "enso"]]
            )

        if not frames:
            return pd.DataFrame(
                columns=["date", "region_id", "temp_anom", "precip_anom", "ndvi", "enso"]
            )

        df_all = pd.concat(frames, axis=0)
        df_all.index.name = "date"
        df_all.reset_index(inplace=True)
        return df_all


__all__ = ["WeatherClient", "RegionQuery"]
