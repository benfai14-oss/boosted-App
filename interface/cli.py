"""
Command-line interface (CLI) for data ingestion, climate index computation,
market modeling (ARIMAX), and dynamic hedging.

Usage examples:
    python -m interface.cli ingest --commodity wheat --regions FR,US,BR --start 2024-01-01 --end 2025-10-01
    python -m interface.cli climate-index --commodity wheat
    python -m interface.cli market-model --commodity wheat
    python -m interface.cli hedging-dynamic --commodity wheat
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from climate_index import global_index
from market_models import arimax
import numpy as np


# =========================================================
# 1. INGEST COMMAND
# =========================================================
def run_ingest(args: argparse.Namespace) -> None:
    """Run the ingestion script via subprocess (delegates to scripts.pull_all)."""
    cmd = [
        "python", "-m", "scripts.pull_all",
        "--commodity", args.commodity,
        "--regions", args.regions,
        "--start", args.start,
        "--end", args.end,
    ]
    if args.output:
        cmd.extend(["--output", args.output])

    subprocess.run(cmd, check=True)
    print(f"Ingestion completed. Data saved to {args.output or 'data/silver'}.")


# =========================================================
# 2. CLIMATE INDEX COMMAND
# =========================================================
def run_climate_index(args: argparse.Namespace) -> None:
    """Compute the global climate risk index using silver data."""
    print(f"Computing climate index for {args.commodity}...")

    silver_path = Path("data/silver/silver_data.csv")
    if not silver_path.exists():
        raise FileNotFoundError(f"Silver data not found: {silver_path}")
    silver_df = pd.read_csv(silver_path)

    config_path = Path("config/commodities.yaml")
    if not config_path.exists():
        alt_path = Path("configs/commodities.yaml")
        if alt_path.exists():
            config_path = alt_path
        else:
            raise FileNotFoundError("Config file not found in either 'config/' or 'configs/'.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    df_index = global_index.compute_global_index(
        silver_df=silver_df,
        commodity=args.commodity,
        config=config,
    )

    output_path = Path(args.output or f"data/silver/climate_index_{args.commodity}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_index.to_csv(output_path)

    print(f"Climate index computed successfully and saved to {output_path}")
    print("\nPreview of results:")
    print(df_index.head())


# =========================================================
# 3. MARKET MODEL (ARIMAX) COMMAND
# =========================================================
def run_market_model(args: argparse.Namespace) -> None:
    """Fit an ARIMAX model using market prices and climate index for a commodity."""
    commodity = args.commodity.lower()
    print(f"Fitting ARIMAX model for {commodity}...")

    market_path = Path("data/silver/market/market_prices.csv")
    climate_path = Path(f"data/silver/climate_index_{commodity}.csv")

    if not market_path.exists():
        raise FileNotFoundError(f"Market data not found: {market_path}")
    if not climate_path.exists():
        raise FileNotFoundError(f"Climate index file not found: {climate_path}")

    market_df = pd.read_csv(market_path)
    climate_df = pd.read_csv(climate_path)

    if "commodity" in market_df.columns:
        market_df = market_df[market_df["commodity"].str.lower() == commodity]

    for df in (market_df, climate_df):
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)

    df = pd.merge(market_df, climate_df[["date", "global_risk_0_100"]], on="date", how="inner")
    if df.empty:
        raise ValueError("Merged dataset is empty – check date alignment or commodity name.")

    df["y"] = (df["price_spot"] / df["price_spot"].shift(1)).apply(
        lambda x: 0 if pd.isna(x) or x <= 0 else np.log(x)
    )

    params = arimax.fit_arimax(df, p=2, q=2, include_seasonal=True)
    print("Model fitted successfully.")
    print("Coefficients:")
    for k, v in params["coeffs"].items():
        print(f"  {k}: {v:.4f}")

    out_path = Path(f"data/models/arimax_{commodity}.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(params, f)

    print(f"Model parameters saved to {out_path}")


# =========================================================
# 4. HEDGING DYNAMIC COMMAND
# =========================================================
def run_hedging_dynamic(args: argparse.Namespace) -> None:
    """Run the advanced dynamic hedging strategy using ARIMAX results and climate risk data."""
    from hedging.advanced import dynamic_hedging_from_pipeline

    print(f"Running dynamic hedging for {args.commodity}...\n")
    try:
        hedge_df, report = dynamic_hedging_from_pipeline(
            commodity=args.commodity,
            horizon=args.horizon,
            exposure=args.exposure,
            role=args.role,
        )
    except Exception as e:
        print(f"[ERROR] Dynamic hedging failed: {e}")
        return

    print("\n=== Hedging Completed ===")
    print(report)
    print("\nResults saved under data/silver/hedging_<commodity>.csv")


# =========================================================
# MAIN ENTRYPOINT
# =========================================================
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Global Climate Hedging CLI (Ingestion + Climate Index + Market Model + Dynamic Hedging)")
    sub = parser.add_subparsers(dest="command")

    # --- Ingest command ---
    ingest_parser = sub.add_parser("ingest", help="Download raw data into the silver layer")
    ingest_parser.add_argument("--commodity", required=True, help="Commodity name (e.g. wheat)")
    ingest_parser.add_argument("--regions", required=True, help="Comma-separated region identifiers")
    ingest_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ingest_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    ingest_parser.add_argument("--output", help="Output path (default: data/silver)")
    ingest_parser.set_defaults(func=run_ingest)

    # --- Climate index command ---
    ci_parser = sub.add_parser("climate-index", help="Compute the global climate risk index")
    ci_parser.add_argument("--commodity", required=True, help="Commodity name (e.g. wheat)")
    ci_parser.add_argument("--output", help="Optional output CSV path")
    ci_parser.set_defaults(func=run_climate_index)

    # --- Market model command ---
    mm_parser = sub.add_parser("market-model", help="Fit ARIMAX model using market and climate data")
    mm_parser.add_argument("--commodity", required=True, help="Commodity name (e.g. wheat)")
    mm_parser.set_defaults(func=run_market_model)

    # --- Hedging dynamic command ---
    hedge_parser = sub.add_parser("hedging-dynamic", help="Run dynamic hedging using ARIMAX and risk data")
    hedge_parser.add_argument("--commodity", required=True, help="Commodity name (e.g. wheat)")
    hedge_parser.add_argument("--horizon", type=int, default=8, help="Forecast horizon (weeks)")
    hedge_parser.add_argument("--role", choices=["importer", "exporter"], default="importer", help="Hedging role")
    hedge_parser.add_argument("--exposure", type=float, default=1000.0, help="Exposure quantity to hedge")
    hedge_parser.set_defaults(func=run_hedging_dynamic)

    parsed = parser.parse_args(argv)
    if not hasattr(parsed, "func"):
        parser.print_help()
        return
    parsed.func(parsed)


if __name__ == "__main__":
    main()
