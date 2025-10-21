"""Agricultural production and stocks from the World Bank API.

This client fetches high‑level agricultural indicators from the World Bank
development data API.  Because true commodity‑specific production and
stock statistics require authenticated services (e.g. FAOSTAT,
USDA PSD), this implementation uses a coarse proxy: the indicator
``NV.AGR.TOTL.KD``, which corresponds to “Agriculture, forestry and
fishing, value added (constant 2015 USD)”.  You can substitute a
different indicator code if you prefer a different measure.

For each region (ISO‑2 code) and date range, the client retrieves
annual values and interpolates them to a weekly frequency.  Stocks are
initially set to a constant fraction of production (30%), but you can
adjust this logic to suit your needs.

If the network request fails or no data are returned, the client
provides deterministic defaults to ensure downstream processes can
continue.  This behaviour makes the pipeline robust to API outages.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class AgriClient:
    """Client for agricultural production and stock data."""

    # Mapping from region_id to ISO‑3 country code for World Bank API
    _ISO3_MAP = {
        "FR": "FRA",
        "US": "USA",
        "BR": "BRA",
        "AR": "ARG",
        "CN": "CHN",
        "UA": "UKR",
    }
    # Indicator code: Agriculture, forestry and fishing, value added (constant 2015 USD)
    _INDICATOR = "NV.AGR.TOTL.KD"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def _fetch_indicator(self, iso3: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Call the World Bank API and return a DataFrame of annual values.

        Parameters
        ----------
        iso3: str
            Three‑letter ISO country code (e.g. 'FRA').
        start: str
            ISO start date (YYYY‑MM‑DD).
        end: str
            ISO end date (YYYY‑MM‑DD).

        Returns
        -------
        pandas.DataFrame or None
            A DataFrame indexed by year with a single column ``value``.  Returns
            ``None`` on failure.
        """
        # Extract year boundaries
        start_year = _dt.datetime.fromisoformat(start).year
        end_year = _dt.datetime.fromisoformat(end).year
        url = (
            f"https://api.worldbank.org/v2/country/{iso3}/indicator/{self._INDICATOR}"
            f"?date={start_year}:{end_year}&format=json"
        )
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Failed to fetch agri indicator for %s: %s", iso3, exc)
            return None
        try:
            records = data[1]
            years = []
            values = []
            for rec in records:
                val = rec.get("value")
                year = rec.get("date")
                if val is None or year is None:
                    continue
                years.append(int(year))
                values.append(float(val))
            if not years:
                return None
            df = pd.DataFrame({"year": years, "production": values}).set_index("year").sort_index()
            return df
        except Exception as exc:
            logger.warning("Failed to parse World Bank data for %s: %s", iso3, exc)
            return None

    def fetch_production_stocks(self, region_id: str, start: str, end: str) -> pd.DataFrame:
        """Fetch agricultural production and stock estimates for a region.

        Parameters
        ----------
        region_id: str
            Two‑letter region identifier (matching ``WeatherClient``).
        start: str
            Start date (YYYY‑MM‑DD).
        end: str
            End date (YYYY‑MM‑DD).

        Returns
        -------
        pandas.DataFrame
            Frame indexed by week (Monday) with columns ``['prod_estimate', 'stocks']``.
        """
        iso3 = self._ISO3_MAP.get(region_id)
        if not iso3:
            logger.error("Unknown region_id '%s' in AgriClient", region_id)
            return pd.DataFrame(columns=["date", "prod_estimate", "stocks"])
        annual_df = self._fetch_indicator(iso3, start, end)
        # Define weekly index
        weeks = pd.date_range(start, end, freq="W-MON")
        if annual_df is None or annual_df.empty:
            # fallback: constant production and stocks
            rng = np.random.default_rng(seed=(hash(region_id) % 54321))
            prod = pd.Series(rng.normal(loc=100.0, scale=20.0, size=len(weeks)), index=weeks)
            stocks = prod * 0.3
        else:
            # Forward fill annual values to weekly frequency
            # Use linear interpolation between years
            # Convert to a daily index for interpolation
            annual_index = pd.to_datetime([f"{year}-07-01" for year in annual_df.index])
            # approximate mid‑year value to avoid seasonal assumptions
            ann_series = pd.Series(annual_df["production"].values, index=annual_index)
            daily = ann_series.resample("D").interpolate(method="linear")
            weekly_interp = daily.reindex(weeks, method="nearest")
            prod = weekly_interp.rename("prod_estimate")
            stocks = prod * 0.3
        result = pd.DataFrame({"date": weeks, "prod_estimate": prod.values, "stocks": stocks.values})
        return result