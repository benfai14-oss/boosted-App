from __future__ import annotations

"""
MarketClient — fetch weekly commodity futures from Yahoo Finance.

- Downloads daily close prices via yfinance
- Handles column quirks: 'Close' vs 'Adj Close', MultiIndex, symbol-suffixed columns
- Computes 30-day realized volatility
- Resamples to weekly (Friday)
- Returns:
  ['date','commodity','price_spot','price_front_fut','realized_vol_30d','data_source']
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

YAHOO_SYMBOLS = {
    "soybean": "ZS=F",
    "corn": "ZC=F",
    "wheat": "ZW=F",
    "coffee": "KC=F",
    "sugar": "SB=F",
    "cocoa": "CC=F",
}


def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index, utc=True)
    out = df.copy()
    out.index = idx
    return out


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns (yfinance can return them)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(x) for x in tup if x not in ("", None)])
            for tup in df.columns.values
        ]
    return df


def _normalize_close_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we end up with a single column named 'close'.

    Accepts any of:
      - 'Close', 'Adj Close'
      - 'close', 'adj_close'
      - 'Close_ZS=F', 'Adj Close_ZS=F'
      - 'close_zs=f', 'adj_close_zs=f'
    Preference: adj_close* > close*
    """
    df = _flatten_columns(df).copy()

    # lower-case + normalize spaces to underscore
    rename_map = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    cols = list(df.columns)
    adj_candidates = [c for c in cols if "adj_close" in c]
    close_candidates = [c for c in cols if "close" in c]

    pick: Optional[str] = None
    if adj_candidates:
        pick = "adj_close" if "adj_close" in adj_candidates else adj_candidates[0]
    elif close_candidates:
        pick = "close" if "close" in close_candidates else close_candidates[0]

    if pick is None:
        raise KeyError(f"No close-like column found. Available: {cols}")

    if pick != "close":
        df["close"] = df[pick]

    return df[["close"]].copy()


class MarketClient:
    """Download weekly futures prices from Yahoo Finance (robust to column quirks)."""

    def __init__(self) -> None:
        pass

    def _download(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        try:
            # Keep explicit auto_adjust to avoid surprises
            raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
            if raw is None or raw.empty:
                logger.warning("No data for %s between %s and %s", symbol, start, end)
                return None
            raw = _to_utc_index(raw)
            df = _normalize_close_column(raw)
            return df
        except Exception as exc:
            logger.error("Failed to download/normalize %s: %s", symbol, exc)
            return None

    def fetch_prices(self, commodity: str, start: str, end: str) -> pd.DataFrame:
        """
        Returns weekly (Friday) prices for the given commodity.

        Columns:
        ['date','commodity','price_spot','price_front_fut','realized_vol_30d','data_source']
        """
        symbol = YAHOO_SYMBOLS.get(commodity.lower())
        if not symbol:
            raise ValueError(f"Unknown commodity: {commodity}")

        df = self._download(symbol, start, end)
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["date", "commodity", "price_spot", "price_front_fut", "realized_vol_30d", "data_source"]
            )

        df = df.copy()
        # daily realized vol on % returns, annualized
        df["returns"] = df["close"].pct_change()
        df["realized_vol_30d"] = df["returns"].rolling(30, min_periods=10).std() * np.sqrt(252)
        df.drop(columns=["returns"], inplace=True)

        # Weekly (Friday) aggregation
        weekly = df.resample("W-FRI").agg({"close": "last", "realized_vol_30d": "last"})
        weekly.rename(columns={"close": "price_spot"}, inplace=True)
        weekly["price_front_fut"] = weekly["price_spot"]  # proxy
        weekly["commodity"] = commodity
        weekly["data_source"] = "yahoo"

        # === FIX: name index before reset, then convert to UTC explicitly ===
        weekly.index = pd.to_datetime(weekly.index, utc=True)
        weekly.index.name = "date"
        weekly = weekly.reset_index()

        return weekly[["date", "commodity", "price_spot", "price_front_fut", "realized_vol_30d", "data_source"]]
