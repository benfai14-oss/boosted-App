"""
Master pull script (Layer A orchestration).

Single entry point to fetch and persist all upstream datasets:
- Weather anomalies by region
- Market prices & realised volatility by commodity
- Agricultural production / stocks proxies by region

Features:
- Automatically determines date range (past Monday over the last 15 years)
- Rescales synthetic market data if unrealistic
- Merges into unified silver layer

Usage:
    python -m scripts.pull_all --commodity wheat [--force]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List
import pandas as pd
from datetime import datetime, timedelta
import argparse

from ingestion.clients.weather_client import WeatherClient, RegionQuery
from ingestion.clients.market_client import MarketClient
from ingestion.clients.agri_client import AgriClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static config
# ---------------------------------------------------------------------------

REGIONS: List[str] = ["FR", "US", "BR", "AR", "CN", "UA"]
ALL_COMMODITIES: List[str] = ["wheat", "corn", "soybean", "cocoa", "coffee", "sugar"]

# ---------------------------------------------------------------------------
# Date window (15 years rolling, aligned to Monday)
# ---------------------------------------------------------------------------

TODAY = datetime.today().date()
# 🕒 align TODAY to the previous Monday
TODAY = TODAY - timedelta(days=TODAY.weekday())
START_WINDOW = TODAY - timedelta(days=15 * 365)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_future(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Remove rows beyond today's date (prevents 'future Monday' issue)."""
    if date_col not in df.columns:
        return df
    df[date_col] = pd.to_datetime(df[date_col])
    return df[df[date_col] <= pd.Timestamp(TODAY)]


def incremental_update(existing_df: pd.DataFrame, new_df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Merge existing and new data, drop duplicates, and trim to 15-year window."""
    if existing_df.empty:
        combined = new_df
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)

    if date_col not in combined.columns:
        raise KeyError(f"Expected a '{date_col}' column in the data.")

    combined[date_col] = pd.to_datetime(combined[date_col])
    combined = combined.drop_duplicates(subset=[date_col] + [c for c in combined.columns if c != date_col])
    combined = combined.sort_values(date_col)
    combined = combined[combined[date_col] >= pd.Timestamp(START_WINDOW)]
    combined = _truncate_future(combined)
    return combined

# ---------------------------------------------------------------------------
# 1) WEATHER
# ---------------------------------------------------------------------------

def build_weather_anomalies(*, commodity: str, force: bool = False) -> None:
    logger.info(f"Building weather anomalies dataset (rolling 15y) for {commodity} ...")

    out_dir = Path("data/silver/weather")
    out_file = out_dir / "weather_anomalies.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not out_file.exists() or force:
        existing = pd.DataFrame()
        start_new = START_WINDOW.strftime("%Y-%m-%d")
    else:
        existing = pd.read_csv(out_file)
        last_date = pd.to_datetime(existing["date"]).max() if "date" in existing else None
        start_new = (last_date + timedelta(days=1)).strftime("%Y-%m-%d") if last_date else START_WINDOW.strftime("%Y-%m-%d")

    client = WeatherClient()
    queries = [RegionQuery(region_id=r, start=start_new, end=TODAY.strftime("%Y-%m-%d")) for r in REGIONS]
    df_new = client.fetch_weather_anomalies(commodity, queries)
    if df_new.empty:
        logger.warning("No new weather data.")
        return

    updated = incremental_update(existing, df_new)
    updated.to_csv(out_file, index=False)
    logger.info(f"Weather anomalies updated → {out_file} (shape={updated.shape})")

# ---------------------------------------------------------------------------
# 2) MARKET
# ---------------------------------------------------------------------------

def build_market_prices(*, commodity: str, force: bool = False) -> None:
    logger.info(f"Building market prices dataset (rolling 15y) for {commodity} ...")

    out_dir = Path("data/silver/market")
    out_file = out_dir / "market_prices.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not out_file.exists() or force:
        existing = pd.DataFrame()
        start_new = START_WINDOW.strftime("%Y-%m-%d")
    else:
        existing = pd.read_csv(out_file)
        last_date = pd.to_datetime(existing["date"]).max() if "date" in existing else None
        start_new = (last_date + timedelta(days=1)).strftime("%Y-%m-%d") if last_date else START_WINDOW.strftime("%Y-%m-%d")

    client = MarketClient()
    try:
        df_new = client.fetch_prices(commodity, start_new, TODAY.strftime("%Y-%m-%d"))
    except Exception as exc:
        logger.error(f"Error fetching prices for {commodity}: {exc}")
        return

    if df_new.empty:
        logger.warning(f"No new market data for {commodity}.")
        return

    # ✅ Rescale synthetic price paths if unrealistic
    p = df_new["price_spot"]
    if p.max() > 500 or p.mean() > 250:
        logger.warning(f"⚠️ Detected high synthetic prices for {commodity}, rescaling...")
        scale = 200 / p.median()
        for col in ["price_spot", "price_front_fut"]:
            df_new[col] = df_new[col] * scale

    df_new = df_new.copy()
    if "commodity" not in df_new.columns:
        df_new.insert(1, "commodity", commodity)

    updated = incremental_update(existing, df_new)
    updated.to_csv(out_file, index=False)
    logger.info(f"Market prices updated → {out_file} (shape={updated.shape})")

# ---------------------------------------------------------------------------
# 3) AGRI
# ---------------------------------------------------------------------------

def build_agri_indicators(*, force: bool = False) -> None:
    logger.info("Building agricultural indicators dataset (rolling 15y)...")

    out_dir = Path("data/silver/agri")
    out_file = out_dir / "agri_indicators.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not out_file.exists() or force:
        existing = pd.DataFrame()
        start_new = START_WINDOW.strftime("%Y-%m-%d")
    else:
        existing = pd.read_csv(out_file)
        last_date = pd.to_datetime(existing["date"]).max() if "date" in existing else None
        start_new = (last_date + timedelta(days=1)).strftime("%Y-%m-%d") if last_date else START_WINDOW.strftime("%Y-%m-%d")

    client = AgriClient()
    frames: List[pd.DataFrame] = []

    for rid in REGIONS:
        try:
            df_new = client.fetch_production_stocks(rid, start_new, TODAY.strftime("%Y-%m-%d"))
        except Exception as exc:
            logger.error(f"Error fetching agri data for {rid}: {exc}")
            continue
        if df_new.empty:
            continue
        df_new["region_id"] = rid
        frames.append(df_new)

    if not frames:
        logger.warning("No agricultural data retrieved.")
        return

    new_all = pd.concat(frames, ignore_index=True)
    updated = incremental_update(existing, new_all)
    updated.to_csv(out_file, index=False)
    logger.info(f"Agricultural indicators updated → {out_file} (shape={updated.shape})")

# ---------------------------------------------------------------------------
# 4) MERGE SILVER DATA
# ---------------------------------------------------------------------------

def merge_silver_data() -> None:
    base = Path("data/silver")

    def safe_read(path: Path) -> pd.DataFrame:
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

    df = _truncate_future(df)
    out_path = base / "silver_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Merged silver_data.csv → {out_path} (shape={df.shape})")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rolling 15-year data pull (Layer A).")
    parser.add_argument("--commodity", type=str, required=True, help="Commodity to pull (e.g., 'wheat')")
    parser.add_argument("--force", action="store_true", help="Force full refresh from APIs.")
    args = parser.parse_args()

    if args.commodity not in ALL_COMMODITIES:
        allowed = ", ".join(ALL_COMMODITIES)
        raise ValueError(f"Unknown commodity '{args.commodity}'. Allowed: {allowed}")

    logger.info(f"Starting rolling data pull (15 years) for {args.commodity} ...")

    build_weather_anomalies(commodity=args.commodity, force=args.force)
    build_market_prices(commodity=args.commodity, force=args.force)
    build_agri_indicators(force=args.force)
    merge_silver_data()

    logger.info("✅ Full Layer A data pull completed successfully.")


if __name__ == "__main__":
    main()
