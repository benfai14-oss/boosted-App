"""
Master pull script.

Single entry point to fetch and persist all upstream datasets
(weather, market data, agricultural indicators) into local files
under `data/silver/` for downstream layers to consume.

Usage:
    python -m scripts.pull_all
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from ingestion.clients import WeatherClient, MarketClient, AgriClient, WeatherClient as WC  # noqa: F401
from ingestion.clients.weather_client import RegionQuery
from scripts.common import write_table, log_event

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global config for Layer A
# ---------------------------------------------------------------------------

# Régions suivies dans tout le projet (cohérent avec WeatherClient & AgriClient)
REGIONS: List[str] = ["FR", "US", "BR", "AR", "CN", "UA"]

# Commodities supportées par MarketClient._SYMBOL_MAP
COMMODITIES: List[str] = ["wheat", "corn", "soybean", "cocoa", "coffee", "sugar"]

# Fenêtre temporelle commune pour l’ingestion
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"


# ---------------------------------------------------------------------------
# 1) WEATHER: anomalies hebdo par région
# ---------------------------------------------------------------------------

def build_weather_anomalies() -> None:
    """
    Fetch weather data via WeatherClient, compute weekly anomalies,
    and persist a clean table for downstream layers.

    Output:
        data/silver/weather/weather_anomalies.parquet (ou .csv selon env)
    """
    logger.info("Building weather anomalies dataset...")

    client = WeatherClient()

    queries = [
        RegionQuery(region_id=rid, start=START_DATE, end=END_DATE)
        for rid in REGIONS
    ]

    df = client.fetch_weather_anomalies(
        commodity="wheat",  # placeholder pour pondérations futures
        queries=queries,
    )

    if df.empty:
        logger.warning("Weather anomalies dataframe is empty. No file written.")
        log_event(
            layer="ingestion",
            action="weather_empty",
            payload={"reason": "no_data", "regions": REGIONS},
        )
        return

    out_path = "data/silver/weather/weather_anomalies.parquet"
    write_table(df, out_path)

    logger.info(
        "Saved weather anomalies to %s with shape %s",
        out_path,
        df.shape,
    )
    log_event(
        layer="ingestion",
        action="weather_written",
        payload={"path": out_path, "rows": int(df.shape[0])},
    )


# ---------------------------------------------------------------------------
# 2) MARKET: prix & vol réalisés par commodity
# ---------------------------------------------------------------------------

def build_market_prices() -> None:
    """
    Fetch weekly prices & realised vol for all configured commodities
    via MarketClient and persist a stacked table.

    Output:
        data/silver/market/market_prices.parquet
        (colonnes: date, commodity, price_spot, price_front_fut, realized_vol_30d)
    """
    logger.info("Building market prices dataset...")

    client = MarketClient()
    frames = []

    for com in COMMODITIES:
        df = client.fetch_prices(commodity=com, start=START_DATE, end=END_DATE)
        if df is None or df.empty:
            logger.warning("No market data for commodity '%s'", com)
            continue

        df = df.copy()
        # On ajoute la colonne 'commodity' pour empiler toutes les séries
        df.insert(1, "commodity", com)
        frames.append(df)

    if not frames:
        logger.warning("No market prices retrieved for any commodity.")
        log_event(
            layer="ingestion",
            action="market_empty",
            payload={"commodities": COMMODITIES},
        )
        return

    all_prices = pd.concat(frames, axis=0, ignore_index=True)

    out_path = "data/silver/market/market_prices.parquet"
    write_table(all_prices, out_path)

    logger.info(
        "Saved market prices to %s with shape %s",
        out_path,
        all_prices.shape,
    )
    log_event(
        layer="ingestion",
        action="market_written",
        payload={"path": out_path, "rows": int(all_prices.shape[0])},
    )


# ---------------------------------------------------------------------------
# 3) AGRI: production & stocks proxies par région
# ---------------------------------------------------------------------------

def build_agri_indicators() -> None:
    """
    Fetch agricultural production & stock proxies via AgriClient
    for all configured regions.

    Output:
        data/silver/agri/agri_indicators.parquet
        (colonnes: date, region_id, prod_estimate, stocks)
    """
    logger.info("Building agricultural indicators dataset...")

    client = AgriClient()
    frames = []

    for rid in REGIONS:
        df = client.fetch_production_stocks(
            region_id=rid,
            start=START_DATE,
            end=END_DATE,
        )
        if df is None or df.empty:
            logger.warning("No agri data for region '%s'", rid)
            continue

        df = df.copy()
        df["region_id"] = rid
        frames.append(df)

    if not frames:
        logger.warning("No agricultural indicators retrieved for any region.")
        log_event(
            layer="ingestion",
            action="agri_empty",
            payload={"regions": REGIONS},
        )
        return

    all_agri = pd.concat(frames, axis=0, ignore_index=True)

    out_path = "data/silver/agri/agri_indicators.parquet"
    write_table(all_agri, out_path)

    logger.info(
        "Saved agri indicators to %s with shape %s",
        out_path,
        all_agri.shape,
    )
    log_event(
        layer="ingestion",
        action="agri_written",
        payload={"path": out_path, "rows": int(all_agri.shape[0])},
    )


# ---------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate all Layer A pulls.

    Chaque build_xxx():
      - appelle un client d'ingestion (WeatherClient, MarketClient, AgriClient)
      - produit une table propre
      - l'enregistre sous data/silver/...
    """
    logger.info("Starting full data pull (Layer A)...")

    build_weather_anomalies()
    build_market_prices()
    build_agri_indicators()

    logger.info("Full data pull completed.")


if __name__ == "__main__":
    main()
