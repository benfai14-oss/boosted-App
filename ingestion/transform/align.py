"""Functions to align and merge weather, agricultural and market data.

The ingestion layer fetches three separate tables:

* **Weather anomalies** per region per week (returned by
  :class:`ingestion.clients.WeatherClient`).
* **Agricultural production and stocks** per region per week
  (returned by :class:`ingestion.clients.AgriClient`).
* **Market prices** per week (returned by :class:`ingestion.clients.MarketClient`).

This module defines helper functions to merge these tables into a
single *silver* dataset.  The merge is performed on the date index,
and each region receives the same market prices because futures
contracts are global.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


def align_data(
    weather_df: pd.DataFrame,
    market_df: pd.DataFrame,
    agri_frames: Dict[str, pd.DataFrame],
    regions: List[str],
) -> pd.DataFrame:
    """Align weather, market and agricultural data on a weekly calendar.

    Parameters
    ----------
    weather_df: pandas.DataFrame
        Combined weather anomalies for all regions with columns
        ``['date', 'region_id', 'temp_anom', 'precip_anom', 'ndvi', 'enso']``.
    market_df: pandas.DataFrame
        Market prices with columns ``['date', 'price_spot', 'price_front_fut',
        'realized_vol_30d']``.
    agri_frames: dict
        Mapping from ``region_id`` to a DataFrame with columns
        ``['date', 'prod_estimate', 'stocks']``.
    regions: list
        List of region identifiers to include in the output.  Any
        ``region_id`` not present in ``agri_frames`` will be ignored.

    Returns
    -------
    pandas.DataFrame
        Silver layer with columns ``['date','region_id','region_weight',
        'temp_anom','precip_anom','ndvi','enso','prod_estimate','stocks',
        'price_spot','price_front_fut','realized_vol_30d']``.
    """
    # Prepare market prices: set date as index for join
    mkt = market_df.copy().set_index("date")
    rows = []
    for rid in regions:
        w = weather_df.loc[weather_df["region_id"] == rid].copy()
        if w.empty:
            continue
        # join agri data for this region
        adf = agri_frames.get(rid)
        if adf is None or adf.empty:
            continue
        temp = w.set_index("date").join(adf.set_index("date"), how="left")
        # join market (broadcasting same values to all rows)
        temp = temp.join(mkt, how="left")
        temp["region_id"] = rid
        temp["region_weight"] = None  # to be filled by index calculator
        rows.append(temp.reset_index())
    if not rows:
        return pd.DataFrame(columns=[
            "date", "region_id", "region_weight", "temp_anom", "precip_anom",
            "ndvi", "enso", "prod_estimate", "stocks", "price_spot",
            "price_front_fut", "realized_vol_30d"
        ])
    merged = pd.concat(rows, axis=0, ignore_index=True)
    # reorder columns
    return merged[[
        "date", "region_id", "region_weight", "temp_anom", "precip_anom",
        "ndvi", "enso", "prod_estimate", "stocks", "price_spot",
        "price_front_fut", "realized_vol_30d"
    ]]