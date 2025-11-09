"""
Master pull script (Layer A orchestration).

Single entry point to fetch and persist all upstream datasets:
- Weather anomalies by region
- Market prices & realised volatility by commodity
- Agricultural production / stocks proxies by region

Outputs are written under `data/silver/` and are consumed by
downstream layers (climate index, models, hedging, visualization).

Usage:
    python -m scripts.pull_all
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List
import pandas as pd

from ingestion.clients import WeatherClient, MarketClient, AgriClient
from ingestion.clients.weather_client import RegionQuery

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

REGIONS: List[str] = ["FR", "US", "BR", "AR", "CN", "UA"]

COMMODITIES: List[str] = [
    "wheat",
    "corn",
    "soybean",
    "cocoa",
    "coffee",
    "sugar",
]

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

# ---------------------------------------------------------------------------
# 1) WEATHER
# ---------------------------------------------------------------------------

def build_weather_anomalies() -> None:
    logger.info("Building weather anomalies dataset...")

    client = WeatherClient()
    queries = [
        RegionQuery(region_id=rid, start=START_DATE, end=END_DATE)
        for rid in REGIONS
    ]

    df = client.fetch_weather_anomalies(commodity="wheat", queries=queries)

    if df.empty:
        logger.warning("Weather anomalies dataframe is empty. No file written.")
        return

    out_dir = Path("data/silver/weather")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "weather_anomalies.csv"
    df.to_csv(out_file, index=False)
    logger.info("Saved weather anomalies to %s with shape %s", out_file, df.shape)

# ---------------------------------------------------------------------------
# 2) MARKET
# ---------------------------------------------------------------------------

def build_market_prices() -> None:
    logger.info("Building market prices dataset...")

    client = MarketClient()
    frames: List[pd.DataFrame] = []

    for com in COMMODITIES:
        try:
            df = client.fetch_prices(commodity=com, start=START_DATE, end=END_DATE)
        except TypeError as exc:
            logger.error("Error calling fetch_prices for commodity '%s': %s", com, exc)
            continue

        if df is None or df.empty:
            logger.warning("No market data for commodity '%s'", com)
            continue

        df = df.copy()
        if "commodity" not in df.columns:
            df.insert(1, "commodity", com)
        frames.append(df)

    if not frames:
        logger.warning("No market prices retrieved for any commodity.")
        return

    all_prices = pd.concat(frames, axis=0, ignore_index=True)
    out_dir = Path("data/silver/market")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "market_prices.csv"
    all_prices.to_csv(out_file, index=False)
    logger.info("Saved market prices to %s with shape %s", out_file, all_prices.shape)

# ---------------------------------------------------------------------------
# 3) AGRI
# ---------------------------------------------------------------------------

def build_agri_indicators() -> None:
    logger.info("Building agricultural indicators dataset...")

    client = AgriClient()
    frames: List[pd.DataFrame] = []

    for rid in REGIONS:
        try:
            df = client.fetch_production_stocks(region_id=rid, start=START_DATE, end=END_DATE)
        except TypeError as exc:
            logger.error("Error calling fetch_production_stocks for region '%s': %s", rid, exc)
            continue

        if df is None or df.empty:
            logger.warning("No agri data for region '%s'", rid)
            continue

        df = df.copy()
        if "region_id" not in df.columns:
            df["region_id"] = rid
        frames.append(df)

    if not frames:
        logger.warning("No agricultural indicators retrieved for any region.")
        return

    all_agri = pd.concat(frames, axis=0, ignore_index=True)
    out_dir = Path("data/silver/agri")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "agri_indicators.csv"
    all_agri.to_csv(out_file, index=False)
    logger.info("Saved agri indicators to %s with shape %s", out_file, all_agri.shape)

# ---------------------------------------------------------------------------
# 4) MERGE SILVER DATA
# ---------------------------------------------------------------------------

def merge_silver_data() -> None:
    base = Path("data/silver")

    def safe_read(path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    weather = safe_read(base / "weather" / "weather_anomalies.csv")
    market = safe_read(base / "market" / "market_prices.csv")
    agri = safe_read(base / "agri" / "agri_indicators.csv")

    if weather.empty and market.empty and agri.empty:
        print("No data available to merge.")
        return

    df = weather
    if not agri.empty:
        df = df.merge(agri, on=["date", "region_id"], how="outer")
    if not market.empty:
        df = df.merge(market, on="date", how="outer")

    out_path = base / "silver_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Merged silver_data.csv created at {out_path} with shape {df.shape}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Starting full data pull (Layer A)...")

    build_weather_anomalies()
    build_market_prices()
    build_agri_indicators()
    merge_silver_data()

    logger.info("Full data pull completed.")

if __name__ == "__main__":
    main()
