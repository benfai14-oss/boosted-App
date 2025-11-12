
from __future__ import annotations

"""
Usage:
    python -m scripts.pull_all --commodity wheat [--force]
"""


"""
Pull Layer A data (rolling 15 years) and write Silver CSVs.

What this script does:
- WEATHER: downloads daily temperature (°C) and precipitation (mm) per region,
  aggregates to weekly (Friday), then computes z-score anomalies; persists to
  data/silver/weather/weather_anomalies.csv (incremental by default).
- MARKET: downloads front futures "Close" weekly (Friday) for supported
  commodities (via yfinance in MarketClient), computes 30-day realized vol,
  and persists to data/silver/market/market_prices.csv (incremental).
- AGRI: placeholder/example supply series; persists to
  data/silver/agri/agri_supply.csv (incremental).

Key detail: ALL timestamps are normalized to timezone-aware UTC to avoid
"Cannot compare tz-naive and tz-aware timestamps" errors when merging old+new.
"""



import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
from pandas import DataFrame

# ---- Clients (must exist in your repo) ---------------------------------------
from ingestion.clients.weather_client import WeatherClient, RegionQuery
from ingestion.clients.market_client import MarketClient
from ingestion.clients.agri_client import AgriClient

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # repo root
SILVER = ROOT / "data" / "silver"
WEATHER_OUT = SILVER / "weather" / "weather_anomalies.csv"
MARKET_OUT = SILVER / "market" / "market_prices.csv"
AGRI_OUT = SILVER / "agri" / "agri_supply.csv"

# Ensure folders exist
for p in [WEATHER_OUT.parent, MARKET_OUT.parent, AGRI_OUT.parent]:
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Time window (rolling 15 years ending today)
# ------------------------------------------------------------------------------
TODAY_UTC = pd.Timestamp.utcnow().normalize()  # 00:00 UTC today
START_DATE = (TODAY_UTC - pd.DateOffset(years=15)).date().isoformat()
END_DATE = TODAY_UTC.date().isoformat()

# ------------------------------------------------------------------------------
# Helpers: UTC date handling + incremental merge
# ------------------------------------------------------------------------------
def ensure_utc_datetime(s: pd.Series) -> pd.Series:
    """
    Parse any date-like series to timezone-aware UTC datetimes.
    Handles mixed naive/aware inputs safely.
    """
    return pd.to_datetime(s, utc=True, errors="coerce")


def read_csv_optional(path: Path, date_col: Optional[str] = None) -> Optional[DataFrame]:
    """
    Read a CSV if present. If date_col is provided, coerce it to UTC.
    Returns None if file is missing/empty.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    if date_col and date_col in df.columns:
        df[date_col] = ensure_utc_datetime(df[date_col])
    return df


def incremental_update(
    existing: Optional[DataFrame],
    new: DataFrame,
    on: Iterable[str],
    date_col: str = "date",
) -> DataFrame:
    """
    Merge existing + new rows, normalize dates to UTC, sort by date, and drop dups.
    Keeps the *last* occurrence per key.
    """
    if new is None or new.empty:
        return existing if existing is not None else pd.DataFrame()

    nw = new.copy()
    if date_col in nw.columns:
        nw[date_col] = ensure_utc_datetime(nw[date_col])

    if existing is None or existing.empty:
        out = nw.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        return out

    ex = existing.copy()
    if date_col in ex.columns:
        ex[date_col] = ensure_utc_datetime(ex[date_col])

    combined = pd.concat([ex, nw], ignore_index=True)
    combined[date_col] = ensure_utc_datetime(combined[date_col])
    combined = combined.dropna(subset=[date_col]).sort_values(date_col)

    combined = combined.drop_duplicates(subset=list(on), keep="last").reset_index(drop=True)
    return combined


# ------------------------------------------------------------------------------
# Build: Weather
# ------------------------------------------------------------------------------
def build_weather_anomalies(commodity: str, force: bool = False) -> None:
    logger.info("Building weather anomalies dataset (rolling 15y) for %s ...", commodity)

    # Regions to query (must match WeatherClient._REGION_COORDS)
    regions = ["FR", "US", "BR", "AR", "CN", "UA"]
    queries = [
        RegionQuery(region_id=r, start=START_DATE, end=END_DATE) for r in regions
    ]

    client = WeatherClient()
    df_new = client.fetch_weather_anomalies(commodity=commodity, queries=queries)

    # Normalize date to UTC immediately
    if "date" in df_new.columns:
        df_new["date"] = ensure_utc_datetime(df_new["date"])

    if force:
        out = df_new.sort_values("date").reset_index(drop=True)
        out.to_csv(WEATHER_OUT, index=False)
        logger.info("Saved weather anomalies to %s with shape %s", WEATHER_OUT, out.shape)
        return

    existing = read_csv_optional(WEATHER_OUT, date_col="date")
    out = incremental_update(existing, df_new, on=("date", "region_id"), date_col="date")
    out.to_csv(WEATHER_OUT, index=False)
    logger.info("Saved weather anomalies to %s with shape %s", WEATHER_OUT, out.shape)


# ------------------------------------------------------------------------------
# Build: Market
# ------------------------------------------------------------------------------
def build_market_prices(commodity: str, force: bool = False) -> None:
    logger.info("Building market prices dataset (rolling 15y) for %s ...", commodity)

    client = MarketClient()
    df_new = client.fetch_prices(commodity=commodity, start=START_DATE, end=END_DATE)

    # Expect columns: date (weekly Friday), commodity, price_spot/price_front_fut, realized_vol_30d, data_source
    if "date" in df_new.columns:
        df_new["date"] = ensure_utc_datetime(df_new["date"])

    if force:
        out = df_new.sort_values("date").reset_index(drop=True)
        out.to_csv(MARKET_OUT, index=False)
        logger.info("Saved market prices to %s with shape %s", MARKET_OUT, out.shape)
        return

    existing = read_csv_optional(MARKET_OUT, date_col="date")
    out = incremental_update(existing, df_new, on=("date", "commodity"), date_col="date")
    out.to_csv(MARKET_OUT, index=False)
    logger.info("Saved market prices to %s with shape %s", MARKET_OUT, out.shape)


# ------------------------------------------------------------------------------
# Build: Agri (example supply series)
# ------------------------------------------------------------------------------
def build_agri_supply(commodity: str, force: bool = False) -> None:
    logger.info("Building agri supply dataset (rolling 15y) for %s ...", commodity)

    client = AgriClient()
    df_new = client.fetch_supply(commodity=commodity, start=START_DATE, end=END_DATE)
    # Expect columns: date, region_id, prod_estimate, stocks
    if "date" in df_new.columns:
        df_new["date"] = ensure_utc_datetime(df_new["date"])

    if force:
        out = df_new.sort_values("date").reset_index(drop=True)
        out.to_csv(AGRI_OUT, index=False)
        logger.info("Saved agri supply to %s with shape %s", AGRI_OUT, out.shape)
        return

    existing = read_csv_optional(AGRI_OUT, date_col="date")
    out = incremental_update(existing, df_new, on=("date", "region_id"), date_col="date")
    out.to_csv(AGRI_OUT, index=False)
    logger.info("Saved agri supply to %s with shape %s", AGRI_OUT, out.shape)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull Layer A data (rolling 15 years) and write Silver CSVs."
    )
    parser.add_argument(
        "--commodity",
        required=True,
        help="Commodity key supported by clients (e.g., soybean, corn, wheat, coffee, sugar, cocoa).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite outputs instead of incremental merging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting rolling data pull (15 years) for %s ...", args.commodity)

    build_weather_anomalies(commodity=args.commodity, force=args.force)
    build_market_prices(commodity=args.commodity, force=args.force)
    build_agri_supply(commodity=args.commodity, force=args.force)

    logger.info("Done.")


if __name__ == "__main__":
    main()
