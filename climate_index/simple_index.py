from __future__ import annotations
from pathlib import Path
import pandas as pd


def _read_any(base_path: str | Path) -> pd.DataFrame:
    """
    Read a dataset that can exist as either Parquet or CSV.
    - Tries .parquet first
    - Falls back to .csv if Parquet engine not available
    - Returns empty DataFrame if file not found
    """
    base = Path(base_path)

    # Try with explicit extensions first
    if base.exists():
        if base.suffix == ".csv":
            return pd.read_csv(base)
        elif base.suffix == ".parquet":
            try:
                return pd.read_parquet(base)
            except ImportError:
                alt = base.with_suffix(".csv")
                if alt.exists():
                    return pd.read_csv(alt)
                return pd.DataFrame()

    # Otherwise, test both possibilities without extension
    parquet_path = base.with_suffix(".parquet")
    csv_path = base.with_suffix(".csv")

    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except ImportError:
            if csv_path.exists():
                return pd.read_csv(csv_path)
            return pd.DataFrame()

    if csv_path.exists():
        return pd.read_csv(csv_path)

    return pd.DataFrame()


def _load_weather() -> pd.DataFrame:
    return _read_any("data/silver/weather/weather_anomalies")


def _load_market() -> pd.DataFrame:
    return _read_any("data/silver/market/market_prices")


def _load_agri() -> pd.DataFrame:
    return _read_any("data/silver/agri/agri_indicators")


def build_regional_climate_score() -> pd.DataFrame:
    """
    Combine temperature and precipitation anomalies into a regional
    climate stress score.

    climate_score = temp_anom - precip_anom
    (positive = warmer/drier than normal)
    """
    weather = _load_weather()
    if weather.empty:
        return pd.DataFrame(columns=["date", "region_id", "climate_score"])

    df = weather.copy()
    df["climate_score"] = df["temp_anom"] - df["precip_anom"]
    return df[["date", "region_id", "climate_score"]]


def build_global_climate_index() -> pd.DataFrame:
    """
    Compute a simple equal-weighted global index
    from regional climate scores.
    """
    regional = build_regional_climate_score()
    if regional.empty:
        return pd.DataFrame(columns=["date", "global_climate_index"])

    idx = (
        regional.groupby("date", as_index=False)["climate_score"]
        .mean()
        .rename(columns={"climate_score": "global_climate_index"})
    )
    return idx
