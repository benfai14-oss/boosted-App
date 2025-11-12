
from __future__ import annotations

"""
AgriClient — synthetic agricultural supply generator.

What it does
------------
- Produces weekly (Friday) estimates for:
  * prod_estimate : level-like production estimate (arbitrary units)
  * stocks        : stock level (arbitrary units)

- Deterministic outputs (no API): numbers are reproducible thanks to a seed
  based on (commodity, region_id). This makes CI and re-runs stable.

- Frequency & merge-compatibility:
  Returns a DataFrame with columns:
    ['date', 'region_id', 'prod_estimate', 'stocks', 'data_source']
  where 'date' is a weekly timestamp (W-FRI).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SupplyQuery:
    region_id: str
    start: str  # "YYYY-MM-DD"
    end: str    # "YYYY-MM-DD"


class AgriClient:
    """
    Synthetic agricultural supply client.

    Generates reproducible weekly production & stock series per region,
    suitable for joining with weather anomalies on ['date', 'region_id'].
    """

    # Keep regions aligned with WeatherClient to simplify joins
    _REGIONS = ["FR", "US", "BR", "AR", "CN", "UA"]

    def __init__(self) -> None:
        pass

    @staticmethod
    def _seed(commodity: str, region_id: str) -> int:
        # Stable small positive int from the tuple; mask to avoid negative
        return (hash((commodity.lower(), region_id)) & 0x7FFFFFFF) % 1_000_000

    def _make_daily_supply(
        self,
        commodity: str,
        region_id: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Build deterministic DAILY supply signals then upsample to weekly.

        Model (simple but realistic enough):
        - prod_base depends on region & commodity hash
        - seasonality: annual sinus (amplitude ~10-20%)
        - mild trend and noise
        - stocks follow a smoothed version of production with accumulation & decay
        """
        idx = pd.date_range(start, end, freq="D", tz="UTC")
        if len(idx) == 0:
            return pd.DataFrame(columns=["date", "prod_estimate", "stocks"]).set_index(
                pd.DatetimeIndex([], tz="UTC")
            )

        seed = self._seed(commodity, region_id)
        rng = np.random.default_rng(seed)

        # Region coefficient (keeps magnitudes distinct but stable)
        region_factor = 0.8 + (abs(hash(region_id)) % 300) / 1000.0  # ≈ 0.8 → 1.1

        # Commodity base level variation
        com_factor = 1.0 + (abs(hash(commodity.lower())) % 500) / 1000.0  # ≈ 1.0 → 1.5

        prod_base = 100.0 * region_factor * com_factor  # arbitrary units

        # Annual seasonality over days
        t = np.arange(len(idx))
        phase = (abs(hash(region_id + commodity)) % 360) * np.pi / 180.0
        seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * t / 365.25 + phase)

        # Mild linear trend (some regions growing/declining a touch)
        trend = 1.0 + (rng.normal(0.0, 0.0002) * t)  # tiny drift

        # Noise
        eps = rng.normal(0.0, 0.03, size=len(idx))

        prod = prod_base * seasonal * trend * (1.0 + eps)
        prod = np.maximum(prod, 0.0)

        # Stocks: AR(1)-ish accumulation with decay
        decay = 0.98 + rng.normal(0.0, 0.002)  # near 1
        inflow = 0.20  # fraction of production flowing into stocks
        stocks = np.zeros(len(idx))
        stocks_level0 = prod_base * 3.0 * (0.9 + rng.normal(0.0, 0.02))
        stocks[0] = max(stocks_level0, 0.0)
        for i in range(1, len(idx)):
            stocks[i] = max(decay * stocks[i - 1] + inflow * prod[i] + rng.normal(0.0, prod_base * 0.01), 0.0)

        df = pd.DataFrame(
            {"prod_estimate": prod, "stocks": stocks},
            index=idx,
        )
        df.index.name = "date"
        return df

    def fetch_supply(
        self,
        commodity: str,
        start: str,
        end: str,
        regions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Public API used by scripts.pull_all:
        Returns weekly Friday data with columns:
        ['date','region_id','prod_estimate','stocks','data_source']
        """
        regions = regions or self._REGIONS

        frames = []
        for region_id in regions:
            daily = self._make_daily_supply(commodity, region_id, start, end)

            if daily.empty:
                continue

            # Weekly aggregation (Friday). Use 'mean' which is smoother for levels.
            weekly = daily.resample("W-FRI").mean()
            weekly["region_id"] = region_id
            frames.append(weekly[["region_id", "prod_estimate", "stocks"]])

        if not frames:
            return pd.DataFrame(
                columns=["date", "region_id", "prod_estimate", "stocks", "data_source"]
            )

        out = pd.concat(frames, axis=0)
        out.index.name = "date"
        out = out.reset_index()

        # Final touches
        out["data_source"] = "synthetic"
        # Ensure tz-naive or consistently tz-aware; pick UTC-naive for CSV merge ease
        out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert(None)

        return out


__all__ = ["AgriClient", "SupplyQuery"]
