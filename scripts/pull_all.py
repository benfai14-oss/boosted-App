"""
Master pull script.

Single entry point to fetch and persist all upstream datasets
(weather, macro, prices, etc.) into local files under `data/`
for downstream layers to consume.

Usage:
    python -m scripts.pull_all
"""

import logging
from pathlib import Path

from ingestion.clients.weather_client import WeatherClient, RegionQuery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_weather_anomalies() -> None:
    """
    Fetch weather data (via WeatherClient), compute weekly anomalies,
    and save them to a parquet file used by other layers.
    """
    logger.info("Building weather anomalies dataset...")

    # Où on enregistre les données météo pour le reste du projet
    out_dir = Path("data/weather")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "weather_anomalies.parquet"

    # À adapter selon votre sujet / périmètre
    queries = [
        RegionQuery(region_id="FR", start="2020-01-01", end="2024-12-31"),
        RegionQuery(region_id="US", start="2020-01-01", end="2024-12-31"),
        RegionQuery(region_id="BR", start="2020-01-01", end="2024-12-31"),
        RegionQuery(region_id="AR", start="2020-01-01", end="2024-12-31"),
        RegionQuery(region_id="CN", start="2020-01-01", end="2024-12-31"),
        RegionQuery(region_id="UA", start="2020-01-01", end="2024-12-31"),
    ]

    client = WeatherClient()
    df = client.fetch_weather_anomalies(
        commodity="wheat",  # placeholder for future weighting logic
        queries=queries,
    )

    if df.empty:
        logger.warning(
            "Weather anomalies dataframe is empty. No file will be written."
        )
        return

    df.to_parquet(out_file, index=False)
    logger.info(
        "Saved weather anomalies to %s with shape %s",
        out_file,
        df.shape,
    )


def main() -> None:
    """
    Orchestrate all data pulls.

    Each build_xxx() is responsible for:
    - pulling data from its APIs or sources
    - minimal transformation / validation
    - saving a clean dataset in `data/` for downstream use
    """
    logger.info("Starting full data pull...")

    # 1) Weather dataset
    build_weather_anomalies()

    # 2) Add more builders as your project grows:
    # build_macro()
    # build_commodities()
    # build_equities()
    # etc.

    logger.info("Full data pull completed.")


if __name__ == "__main__":
    main()
