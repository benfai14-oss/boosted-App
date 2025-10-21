"""Market data ingestion from Yahoo Finance.

This client uses the unoffical Yahoo Finance CSV download endpoint to obtain
historical daily prices for commodity futures.  It supports a handful of
agricultural commodities via a static symbol map.  You can extend
``_SYMBOL_MAP`` with additional symbols as needed.

If the request fails (due to lack of network connectivity or invalid
symbols) the client falls back to generating synthetic price paths.  A
deterministic pseudo‑random generator keyed by the commodity name
ensures repeatability across runs.

The returned DataFrame includes weekly spot and front‑month prices
(taken from the daily close) and a simple 30‑day realised volatility
calculated from the log returns.
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


class MarketClient:
    """Client for fetching commodity futures prices from Yahoo Finance."""

    # Mapping from generic commodity names to Yahoo Finance futures symbols
    # e.g. 'wheat' -> 'ZW=F' (CBOT wheat), 'corn' -> 'ZC=F'.  Feel free to
    # extend this mapping for other commodities.
    _SYMBOL_MAP = {
        "wheat": "ZW=F",
        "corn": "ZC=F",
        "soybean": "ZS=F",
        "soybeans": "ZS=F",
        "cocoa": "CC=F",
        "coffee": "KC=F",
        "sugar": "SB=F",
    }

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def _download_csv(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Internal helper to download a CSV from Yahoo Finance.

        Parameters
        ----------
        symbol: str
            Yahoo Finance ticker (e.g. 'ZW=F').
        start: str
            ISO start date (inclusive).
        end: str
            ISO end date (inclusive).

        Returns
        -------
        pandas.DataFrame or None
            Daily OHLCV frame indexed by date, or ``None`` on failure.
        """
        # Convert ISO dates to Unix timestamps
        start_ts = int(_dt.datetime.fromisoformat(start).timestamp())
        # add one day to end date to make it inclusive (Yahoo uses end_ts exclusive)
        end_dt = _dt.datetime.fromisoformat(end) + _dt.timedelta(days=1)
        end_ts = int(end_dt.timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
            f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history&includeAdjustedClose=true"
        )
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Failed to download prices for %s: %s", symbol, exc)
            return None
        try:
            import io
            df = pd.read_csv(io.StringIO(resp.text))
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            return df
        except Exception as exc:
            logger.warning("Failed to parse CSV for %s: %s", symbol, exc)
            return None

    def fetch_prices(self, commodity: str, start: str, end: str) -> pd.DataFrame:
        """Download weekly price and realised volatility for a commodity.

        Parameters
        ----------
        commodity: str
            Name of the commodity (case‑insensitive).  Must be present in
            ``_SYMBOL_MAP``.
        start: str
            ISO start date (YYYY‑MM‑DD).
        end: str
            ISO end date (YYYY‑MM‑DD).

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``['price_spot', 'price_front_fut', 'realized_vol_30d']``
            indexed by weekly dates (Monday week start).
        """
        key = commodity.lower()
        symbol = self._SYMBOL_MAP.get(key)
        if not symbol:
            logger.error("Unknown commodity '%s' in MarketClient", commodity)
            return pd.DataFrame(columns=["price_spot", "price_front_fut", "realized_vol_30d"])
        daily = self._download_csv(symbol, start, end)
        if daily is None or daily.empty:
            # Fallback: simulate a geometric Brownian motion around a nominal price
            logger.info("Generating synthetic price path for %s", commodity)
            dates = pd.date_range(start, end, freq="D")
            rng = np.random.default_rng(seed=(hash(key) % 987654))
            dt = 1/252
            mu = 0.05
            sigma = 0.2
            price0 = 100.0
            prices = [price0]
            for _ in range(1, len(dates)):
                shock = rng.normal(loc=(mu - 0.5 * sigma**2) * dt, scale=sigma * np.sqrt(dt))
                prices.append(prices[-1] * np.exp(shock))
            daily = pd.DataFrame({"Close": prices}, index=dates)
        # Compute daily log returns
        daily = daily.copy()
        daily["log_ret"] = np.log(daily["Close"]).diff()
        # rolling 30‑day vol (std of log returns)
        daily["vol_30"] = daily["log_ret"].rolling(window=30, min_periods=1).std(ddof=0) * np.sqrt(252)
        # Resample to weekly (Monday start)
        weekly = daily.resample("W-MON").agg({
            "Close": "last",    # end‑of‑week price
            "vol_30": "last",   # last vol in the week
        })
        weekly.rename(columns={"Close": "price_spot"}, inplace=True)
        weekly["price_front_fut"] = weekly["price_spot"]  # front month approx = spot
        weekly.rename(columns={"vol_30": "realized_vol_30d"}, inplace=True)
        weekly.index.name = "date"
        weekly.reset_index(inplace=True)
        return weekly[["date", "price_spot", "price_front_fut", "realized_vol_30d"]]