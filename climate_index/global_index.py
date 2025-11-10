"""
climate_index.global_index
-------------------------

This module orchestrates the construction of a global climate risk
index by combining regional risk scores.  It relies on the functions
provided in :mod:`climate_index.regional_score` to compute the
per‑region contributions and then aggregates them using region
weights specified in a configuration file. 

Callers supply a data frame of aligned observations (the ``silver``
data), a commodity identifier and a configuration mapping.  The
function returns a new data frame with one row per date and the
computed global risk index as well as per‑region contributions.


"""

from __future__ import annotations

from typing import Dict, List, Any
import pandas as pd
import yaml
from .regional_score import compute_regional_score


def compute_global_index(
    silver_df: pd.DataFrame,
    commodity: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Compute the global climate risk index for a given commodity.

    Parameters
    ----------
    silver_df : pandas.DataFrame
        An aligned data frame indexed by ``date`` and containing at
        least the columns ``region_id`` and the anomaly columns used
        by the regional score (e.g. ``temp_anom``, ``precip_anom``)
        This corresponds to the
        ``silver`` layer produced by the ingestion pipeline.
    commodity : str
        The commodity identifier. 
    config : dict
        A mapping (parsed from YAML or constructed manually) where
        ``config[commodity]["regions"]`` is a list of region
        definitions.  Each region definition is a dict with keys:

        - ``id``: the region identifier
        - ``weight``: the importance weight of the region in the
          global index (e.g. share of production)
        - ``factors``: a mapping from column names to factor weights
          used by :func:`compute_regional_score`

    Returns
    -------
    pandas.DataFrame
        A data frame indexed by ``date`` with the following columns:

        - ``global_risk_0_100``: the aggregated global risk index
        - ``contrib_{region_id}``: contribution of each region to the
          global index (scaled 0..100)


    """

    if commodity not in config:
        raise KeyError(f"Commodity '{commodity}' not found in configuration")
    region_cfgs = config[commodity].get("regions", [])
    if not region_cfgs:
        raise ValueError(f"No regions configured for commodity '{commodity}'")

    # Prepare list to hold per‑region contributions
    contrib_frames: List[pd.DataFrame] = []
    # Loop through each region defined in the config
    for region in region_cfgs:
        region_id = region.get("id")
        weight = region.get("weight", 0.0)
        factors = region.get("factors", {})
        # Extract subset of data for the region
        region_df = silver_df[silver_df["region_id"] == region_id].copy()
        if region_df.empty:
            # Skip if no data for this region
            continue
        # Compute regional score
        region_df["_regional_score"] = compute_regional_score(region_df, factors)

        region_df = (region_df.groupby("date", as_index=False)["_regional_score"].mean() ) 
        # Multiply by the region weight to obtain contribution
        region_df[f"contrib_{region_id}"] = region_df["_regional_score"] * weight
        contrib_frames.append(region_df[["date", f"contrib_{region_id}"]])

    if not contrib_frames:
        raise ValueError("No regional contributions computed – check configuration and data")
    

    # Merge contributions on date
    contrib_merged = contrib_frames[0]
    for cdf in contrib_frames[1:]:
        contrib_merged = pd.merge(contrib_merged, cdf, on="date", how="outer")
    contrib_merged.set_index("date", inplace=True)
    contrib_merged = contrib_merged.sort_index().fillna(0.0)

    # Sum across regions to obtain global risk
    contrib_merged["global_risk_0_100"] = contrib_merged.sum(axis=1)
    # Reorder columns: global index first
    cols = ["global_risk_0_100"] + [c for c in contrib_merged.columns if c.startswith("contrib_")]
    return contrib_merged[cols]
