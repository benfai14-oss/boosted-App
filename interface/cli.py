import os

"""
Unified CLI for the Global Climate Hedging Project – Streamlit edition.

Pipeline steps:
1) ingest         -> data/silver/*
2) climate-index  -> data/gold/<commodity>_global_index.json
3) market-model   -> data/gold/<commodity>_forecast.json
4) hedge          -> data/gold/<commodity>_hedge_rec.json
5) report         -> launches a beautiful Streamlit dashboard (interactive)

pip install -r requirements.txt
Full run:
python -m interface.cli full-run --commodity wheat --profile balanced --role importer --exposure 10000
Step by step:
# 1) Ingest (si ton pull_all marche dans l’environnement)
python -m interface.cli ingest --commodity wheat
# 2) Index climat
python -m interface.cli climate-index --commodity wheat
# 3) Modèle de marché (avec bandes)
python -m interface.cli market-model --commodity wheat
# 4) Hedging
python -m interface.cli hedge --commodity wheat --profile balanced --role importer --exposure 10000
# 5) Streamlit
python -m interface.cli report --commodity wheat
"""

#from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import yaml
import json

# === Internal modules (adapt to your repo structure if needed) ===
from climate_index import global_index
from market_models import arimax
# If hedging.advanced exists in your repo:
try:
    from hedging.advanced import recommend_hedge_from_arimax
except Exception:
    recommend_hedge_from_arimax = None


# =========================================================
# INGEST
# =========================================================
def run_ingest(args: argparse.Namespace) -> None:
    print(f"[1/5] Ingesting data for {args.commodity}...")
    cmd = ["python", "-m", "scripts.pull_all", "--commodity", args.commodity]
    if getattr(args, "force", False):
        cmd.append("--force")
    # Ne PAS passer --regions / --start / --end, le script ne les supporte pas
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("[ERROR] Ingest failed with exit code", e.returncode)
        print("Tip: ensure dependencies are installed (e.g. `pip install yfinance`) or run with synthetic data.")
        raise
    print("Ingestion complete.\n")


# =========================================================
# CLIMATE INDEX
# =========================================================
def run_climate_index(args: argparse.Namespace) -> None:
    print(f"[2/5] Computing climate index for {args.commodity}...")

    # Silver input (adapt path if your repo differs)
    silver_path = Path("data/silver/silver_data.csv")
    if not silver_path.exists():
        raise FileNotFoundError(f"Missing silver data: {silver_path}")

    silver_df = pd.read_csv(silver_path)

    # Config location (try both common paths)
    config_path = Path("config/commodities.yaml")
    if not config_path.exists():
        config_path = Path("configs/commodities.yaml")
    if not config_path.exists():
        raise FileNotFoundError("commodities.yaml not found (looked in config/ and configs/).")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    df_index = global_index.compute_global_index(
        silver_df=silver_df, commodity=args.commodity, config=config
    )

    # Ensure there's a 'date' column
    if "date" not in df_index.columns:
        df_index = df_index.reset_index().rename(columns={"index": "date"})

    out_path = Path(f"data/gold/{args.commodity}_global_index.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_index.to_json(out_path, orient="records", indent=2, date_format="iso")
    print(f"Climate index saved to {out_path}\n")


# =========================================================
# ARIMAX FORECAST (extended ARIMAX)
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

    # Filtre sur la commodity si présent
    if "commodity" in market_df.columns:
        market_df = market_df[market_df["commodity"].str.lower() == args.commodity.lower()]

    # Nettoyage date
    for df_name, df in [("market_df", market_df), ("climate_df", climate_df)]:
        if "date" not in df.columns:
            df.reset_index(inplace=True)
            df.rename(columns={"index": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values("date", inplace=True)

    # Merge marché + risque
    merged = pd.merge(
        market_df,
        climate_df[["date", "global_risk_0_100"]],
        on="date",
        how="inner"
    )
    if merged.empty:
        raise ValueError("Merged dataset is empty — check date alignment or commodity name.")

    # Choix colonne de prix
    price_col = "price_spot" if "price_spot" in merged.columns else "price_front_fut"
    if price_col not in merged.columns:
        raise KeyError(
            f"Neither 'price_spot' nor 'price_front_fut' found in market data. "
            f"Available columns: {merged.columns.tolist()}"
        )

    # Retour log hebdo
    merged["y"] = np.log(merged[price_col]).diff()

    # Fit ARIMAX étendu (auto-features + saisonnalité)
    params = arimax.fit_arimax(
        merged,
        feature_cols=None,
        p=2,
        q=2,
        include_seasonal=True,
        season_period=52,
        extra_exog=None,
    )

    resid_var = float(params.get("resid_var", 0.0))

    # Forecast 8 steps (8 semaines)
    horizon = 8
    last_price = float(merged[price_col].iloc[-1])
    last_row = merged.iloc[-1]

    # Risk future: on prolonge le dernier niveau (normalisé 0–1)
    risk_future = (merged["global_risk_0_100"].tail(10) / 100.0).tolist()
    if len(risk_future) < horizon:
        risk_future = risk_future + [risk_future[-1]] * (horizon - len(risk_future))

    forecasts, prices = arimax.forecast_arimax(
        last_price=last_price,
        last_row=last_row,
        params=params,
        risk_future=risk_future,
        horizon=horizon,
    )

    # Bandes d’incertitude approximatives sur le log-return cumulé
    forecasts = np.asarray(forecasts, dtype=float)
    cum_ret = np.cumsum(forecasts)

    if resid_var > 0:
        steps = np.arange(1, horizon + 1, dtype=float)
        sigma = np.sqrt(resid_var * steps)          # écart-type sur somme des retours
        lo_ret = cum_ret - 1.96 * sigma
        hi_ret = cum_ret + 1.96 * sigma
        lo_price = last_price * np.exp(lo_ret)
        hi_price = last_price * np.exp(hi_ret)
    else:
        lo_price = hi_price = np.asarray(prices, dtype=float)

    dates_future = pd.date_range(start=merged["date"].iloc[-1], periods=horizon + 1, freq="W")[1:]
    out_df = pd.DataFrame({
        "date": dates_future,
        "price_forecast": prices,
        "y_forecast": forecasts,
        "price_lo95": lo_price,
        "price_hi95": hi_price,
    })

    out_path = Path(f"data/gold/{args.commodity}_forecast.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_json(out_path, orient="records", indent=2, date_format="iso")
    print(f"Forecasts (with bands) saved to {out_path}\n")


# =========================================================
# HEDGING (from ARIMAX + Climate Index)
# =========================================================
def run_hedge(args: argparse.Namespace) -> None:
    print(f"[4/5] Running hedging for {args.commodity} ({args.profile}, {args.role})...")

    forecast_path = Path(f"data/gold/{args.commodity}_forecast.json")
    risk_path = Path(f"data/gold/{args.commodity}_global_index.json")
    out_path = Path(f"data/gold/{args.commodity}_hedge_rec.json")

    if recommend_hedge_from_arimax is None:
        print("⚠️ hedging.advanced.recommend_hedge_from_arimax not available. Creating a placeholder output.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        placeholder = pd.DataFrame([{
            "commodity": args.commodity,
            "profile": args.profile,
            "role": args.role,
            "exposure": float(args.exposure),
            "hedge_notional": float(args.exposure) * 0.6,
            "instrument": "Futures",
            "notes": "Placeholder hedge because hedging.advanced is not available."
        }])
        placeholder.to_json(out_path, orient="records", indent=2)
        print(f"Placeholder hedge saved to {out_path}\n")
        return

    result = recommend_hedge_from_arimax(
        forecast_path=str(forecast_path),
        risk_index_path=str(risk_path),
        profile=args.profile,
        role=args.role,
        exposure=args.exposure,
    )

    # --- Normalisation robuste en DataFrame ---
    def _to_frame(obj):
        import pandas as _pd
        if isinstance(obj, _pd.DataFrame):
            return obj
        if isinstance(obj, dict):
            return _pd.DataFrame([obj])             # <-- dict -> une ligne
        if isinstance(obj, list):
            if len(obj) == 0:
                return _pd.DataFrame()
            if isinstance(obj[0], dict):
                return _pd.DataFrame(obj)           # <-- liste de dicts
            return _pd.DataFrame({"value": obj})    # <-- liste de scalaires
        return _pd.DataFrame([{"result": obj}])     # <-- tout autre type

    out_df = _to_frame(result)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_json(out_path, orient="records", indent=2)
    print(f"Hedge recommendation saved to {out_path}\n")


# =========================================================
# REPORT (launch separate Streamlit app)
# =========================================================
def run_report(args: argparse.Namespace) -> None:
    """
    Launch the Streamlit dashboard defined in interface/streamlit_app.py.

    The commodity is passed via an environment variable HEDGE_COMMODITY.
    """
    env = os.environ.copy()
    env["HEDGE_COMMODITY"] = args.commodity

    try:
        print("Launching Streamlit… If the browser does not open, use the URL printed below.")
        subprocess.run(
            ["streamlit", "run", "interface/streamlit_app.py"],
            check=True,
            env=env,
        )
    except FileNotFoundError:
        print("\n[ERROR] Streamlit is not installed. Install it with:\n  pip install streamlit altair\n")
        raise
    except subprocess.CalledProcessError as e:
        print("Streamlit exited with an error code:", e.returncode)
        raise

# =========================================================
# MAIN ENTRYPOINT
# =========================================================
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Global Climate Hedging CLI (Streamlit report)")
    sub = parser.add_subparsers(dest="command")

    # Full pipeline
    full = sub.add_parser("full-run", help="Run full pipeline end-to-end")
    full.add_argument("--commodity", required=True)
    full.add_argument("--force", action="store_true", help="force re-download in ingest if supported")
    full.add_argument("--profile", default="balanced")
    full.add_argument("--role", default="importer")
    full.add_argument("--exposure", type=float, default=10000.0)

    def full_run(args):
        run_ingest(args)
        run_climate_index(args)
        run_market_model(args)
        run_hedge(args)
        run_report(args)

    full.set_defaults(func=full_run)

    # Individual steps
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
            p.add_argument("--force", action="store_true", help="force re-download in ingest if supported")
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
