"""
Unified CLI for the Global Climate Hedging Project.

Pipeline steps:
1. Ingestion (data/silver)
2. Climate Index (data/gold)
3. ARIMAX Forecast (data/gold)
4. Hedging
5. Report 

Usage examples:
---------------
Run full pipeline:
python -m interface.cli full-run --commodity wheat --profile balanced --role importer --exposure 10000

Run separately: 
python -m interface.cli ingest --commodity wheat --force
python -m interface.cli climate-index --commodity wheat
python -m interface.cli market-model --commodity wheat
python -m interface.cli hedge --commodity wheat --profile balanced --role importer --exposure 10000
python -m interface.cli report --commodity wheat

"""

from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import pandas as pd
import json
import yaml
import numpy as np

# === Internal modules ===
from climate_index import global_index
from market_models import arimax
from hedging.advanced import recommend_hedge_from_arimax
from visualization import report as report_module


# =========================================================
# INGEST
# =========================================================
def run_ingest(args: argparse.Namespace) -> None:
    print(f"[1/5] Ingesting data for {args.commodity} (automatic 15-year window)...")

    cmd = ["python", "-m", "scripts.pull_all", "--commodity", args.commodity]
    if args.force:
        cmd.append("--force")

    subprocess.run(cmd, check=True)
    print("Ingestion complete.\n")



# =========================================================
# CLIMATE INDEX
# =========================================================
def run_climate_index(args: argparse.Namespace) -> None:
    print(f"[2/5] Computing climate index for {args.commodity}...")

    silver_path = Path("data/silver/silver_data.csv")
    if not silver_path.exists():
        raise FileNotFoundError(f"Missing: {silver_path}")

    silver_df = pd.read_csv(silver_path)

    config_path = Path("config/commodities.yaml")
    if not config_path.exists():
        config_path = Path("configs/commodities.yaml")
    if not config_path.exists():
        raise FileNotFoundError("commodities.yaml not found.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    df_index = global_index.compute_global_index(
        silver_df=silver_df, commodity=args.commodity, config=config
    )

    df_index = df_index.reset_index()

    out_path = Path(f"data/gold/{args.commodity}_global_index.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_index.to_json(out_path, orient="records", indent=2)
    print(f"Climate index saved to {out_path}\n")


# =========================================================
# ARIMAX FORECAST
# =========================================================
def run_market_model(args: argparse.Namespace) -> None:
    print(f"[3/5] Running ARIMAX model for {args.commodity}...")

    market_path = Path("data/silver/market/market_prices.csv")
    climate_path = Path(f"data/gold/{args.commodity}_global_index.json")

    if not market_path.exists():
        raise FileNotFoundError(f"Market data missing: {market_path}")
    if not climate_path.exists():
        raise FileNotFoundError(f"Climate index missing: {climate_path}")

    market_df = pd.read_csv(market_path)
    climate_df = pd.read_json(climate_path)

    if "date" not in climate_df.columns:
        if climate_df.index.name == "date" or isinstance(climate_df.index, pd.DatetimeIndex):
            climate_df = climate_df.reset_index()
        else:
            print("⚠️ No 'date' column in climate index JSON, creating from index...")
            climate_df = climate_df.reset_index().rename(columns={"index": "date"})

    if "commodity" in market_df.columns:
        market_df = market_df[market_df["commodity"].str.lower() == args.commodity.lower()]

    for df_name, df in [("market_df", market_df), ("climate_df", climate_df)]:
        if "date" not in df.columns:
            raise KeyError(f"'date' column missing in {df_name} columns: {df.columns.tolist()}")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values("date", inplace=True)

    # ✅ Fix: exclude any data beyond today's date (avoid "next Monday" rows)
    today_cutoff = pd.Timestamp.today().normalize()
    market_df = market_df[market_df["date"] <= today_cutoff]
    climate_df = climate_df[climate_df["date"] <= today_cutoff]

    merged = pd.merge(
        market_df,
        climate_df[["date", "global_risk_0_100"]],
        on="date",
        how="inner"
    )
    if merged.empty:
        raise ValueError("Merged dataset is empty — check date alignment or commodity name.")

    merged["y"] = np.log(merged["price_spot"]).diff()

    params = arimax.fit_arimax(merged, p=2, q=2, include_seasonal=True)

    last_price = merged["price_spot"].iloc[-1]
    last_row = merged.iloc[-1]
    risk_future = [r / 100 for r in merged["global_risk_0_100"].tail(10).to_list()]
    forecasts, prices = arimax.forecast_arimax(
        last_price=last_price,
        last_row=last_row,
        params=params,
        risk_future=risk_future,
        horizon=8,
    )

    dates_future = pd.date_range(start=merged["date"].iloc[-1], periods=9, freq="W")[1:]
    out_df = pd.DataFrame({
        "date": dates_future,
        "price_forecast": prices,
        "y_forecast": forecasts,
    })

    out_path = Path(f"data/gold/{args.commodity}_forecast.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_json(out_path, orient="records", indent=2)
    print(f"Forecasts saved to {out_path}\n")


# =========================================================
# HEDGING
# =========================================================
def run_hedge(args: argparse.Namespace) -> dict:
    print(f"[4/5] Running hedging for {args.commodity} ({args.profile}, {args.role})...")

    forecast_path = f"data/gold/{args.commodity}_forecast.json"
    risk_path = f"data/gold/{args.commodity}_global_index.json"

    result = recommend_hedge_from_arimax(
        forecast_path=forecast_path,
        risk_index_path=risk_path,
        profile=args.profile,
        role=args.role,
        exposure=args.exposure,
    )

    print("Hedge recommendation complete.")
    print(f"Hedge saved at: {forecast_path.replace('_forecast.json', '_hedge_rec.json')}\n")
    return result


# =========================================================
# REPORT
# =========================================================
def run_report(args: argparse.Namespace) -> None:
    print(f"[5/5] Generating report for {args.commodity}...\n")

    forecast_path = f"data/gold/{args.commodity}_forecast.json"
    risk_path = f"data/gold/{args.commodity}_global_index.json"
    hedge_path = f"data/gold/{args.commodity}_hedge_rec.json"
    output_path = f"reports/{args.commodity}_report.pdf"

    Path("reports").mkdir(parents=True, exist_ok=True)

    report_module.generate_pdf_report_from_arimax(
        forecast_path=forecast_path,
        risk_path=risk_path,
        hedge_path=hedge_path,
        output_path=output_path,
        title=f"{args.commodity.capitalize()} Climate Hedging Report"
    )
    print(f"Report generated: {output_path}\n")


# =========================================================
# MAIN ENTRYPOINT
# =========================================================
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Global Climate Hedging CLI (Full pipeline)")
    sub = parser.add_subparsers(dest="command")

    # --- Full pipeline ---
    full = sub.add_parser("full-run", help="Run full pipeline end-to-end")
    full.add_argument("--commodity", required=True)
    full.add_argument("--profile", default="balanced")
    full.add_argument("--role", default="importer")
    full.add_argument("--exposure", type=float, default=10000.0)
    full.add_argument("--force", action="store_true", help="Force full data reload in ingestion step.")

    def full_run(args):
        run_ingest(args)
        run_climate_index(args)
        run_market_model(args)
        run_hedge(args)
        run_report(args)

    full.set_defaults(func=full_run)

    # --- Individual steps ---
    for name, func in {
        "ingest": run_ingest,
        "climate-index": run_climate_index,
        "market-model": run_market_model,
        "hedge": run_hedge,
        "report": run_report,
    }.items():
        p = sub.add_parser(name)
        p.add_argument("--commodity", required=True)
        if name == "ingest":
            p.add_argument("--force", action="store_true", help="Force full data reload from APIs.")
        if name in {"hedge", "full-run"}:
            p.add_argument("--profile", default="balanced")
            p.add_argument("--role", default="importer")
            p.add_argument("--exposure", type=float, default=10000.0)
        p.set_defaults(func=func)

    parsed = parser.parse_args(argv)
    if not hasattr(parsed, "func"):
        parser.print_help()
        return
    parsed.func(parsed)


if __name__ == "__main__":
    main()
