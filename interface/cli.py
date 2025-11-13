"""
Unified CLI for the Global Climate Hedging Project – Streamlit edition.

Pipeline steps:
1) ingest         -> data/silver/*
2) climate-index  -> data/gold/<commodity>_global_index.json
3) market-model   -> data/gold/<commodity>_forecast.json
4) hedge          -> data/gold/<commodity>_hedge_rec.json
5) report         -> launches a beautiful Streamlit dashboard (interactive)

Usage examples:
  python -m interface.cli full-run --commodity wheat --regions europe --start 2023-01-01 --end 2024-01-01 --profile balanced --role importer --exposure 10000
  python -m interface.cli report    --commodity wheat
"""

from __future__ import annotations

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

    # Expect market prices + climate index
    market_path = Path("data/silver/market/market_prices.csv")
    climate_path = Path(f"data/gold/{args.commodity}_global_index.json")

    if not market_path.exists():
        raise FileNotFoundError(f"Market data missing: {market_path}")
    if not climate_path.exists():
        raise FileNotFoundError(f"Climate index missing: {climate_path}")

    market_df = pd.read_csv(market_path)
    climate_df = pd.read_json(climate_path)

    # Fix date columns and filter commodity if present
    if "commodity" in market_df.columns:
        market_df = market_df[market_df["commodity"].str.lower() == args.commodity.lower()]

    for df_name, df in [("market_df", market_df), ("climate_df", climate_df)]:
        if "date" not in df.columns:
            df.reset_index(inplace=True)
            df.rename(columns={"index": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values("date", inplace=True)

    merged = pd.merge(
        market_df,
        climate_df[["date", "global_risk_0_100"]],
        on="date",
        how="inner"
    )
    if merged.empty:
        raise ValueError("Merged dataset is empty — check date alignment or commodity name.")

    # Build weekly log-return y
    price_col = "price_spot" if "price_spot" in merged.columns else "price_front_fut"
    if price_col not in merged.columns:
        raise KeyError(f"Neither 'price_spot' nor 'price_front_fut' found in market data. Found: {merged.columns.tolist()}")

    merged["y"] = np.log(merged[price_col]).diff()

    # Fit extended ARIMAX with seasonality (weekly -> 52)
    params = arimax.fit_arimax(
        merged,
        feature_cols=None,            # auto-build features and risk lags
        p=2, q=2,
        include_seasonal=True,
        season_period=52,
        extra_exog=None,
    )

    # Forecast 8 steps ahead (≈ 8 weeks)
    last_price = float(merged[price_col].iloc[-1])
    last_row = merged.iloc[-1]
    risk_future = (merged["global_risk_0_100"].tail(10) / 100.0).tolist()
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
    out_df.to_json(out_path, orient="records", indent=2, date_format="iso")
    print(f"Forecasts saved to {out_path}\n")


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_json(out_path, orient="records", indent=2)
    print(f"Hedge recommendation saved to {out_path}\n")


# =========================================================
# REPORT (Streamlit App)
# =========================================================
def run_report(args: argparse.Namespace) -> None:
    """
    Launch an interactive Streamlit report instead of a static PDF.

    It creates a temporary Streamlit app that reads:
      - data/gold/{commodity}_forecast.json
      - data/gold/{commodity}_global_index.json
      - data/gold/{commodity}_hedge_rec.json
    and renders a rich dashboard with KPIs, charts, scenario toggles,
    raw tables and download buttons.
    """
    forecast_path = f"data/gold/{args.commodity}_forecast.json"
    risk_path = f"data/gold/{args.commodity}_global_index.json"
    hedge_path = f"data/gold/{args.commodity}_hedge_rec.json"

    app_code = f"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Global Climate Hedging – Report", layout="wide")

FORECAST_PATH = Path(r"{forecast_path}")
RISK_PATH     = Path(r"{risk_path}")
HEDGE_PATH    = Path(r"{hedge_path}")
COMMODITY     = "{args.commodity}"

def _safe_load_json(path: Path):
    if path.exists():
        try:
            return pd.read_json(path)
        except Exception:
            try:
                with open(path, "r") as f:
                    return pd.DataFrame(json.load(f))
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()

fc_df   = _safe_load_json(FORECAST_PATH)
risk_df = _safe_load_json(RISK_PATH)
hedge_df = _safe_load_json(HEDGE_PATH)

for df in (fc_df, risk_df):
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values("date", inplace=True)

st.title(f"{{COMMODITY.capitalize()}} – Climate Hedging Dashboard")

# === Sidebar ===
st.sidebar.header("Controls")
horiz_choice = st.sidebar.multiselect(
    "Show horizons (steps)", options=[2,4,8,12], default=[8],
    help="Number of weekly forecast steps to display."
)
show_hedge = st.sidebar.checkbox("Show hedge details", value=True)
theme = st.sidebar.radio("Theme", options=["Light","Dark"], index=0)
if theme == "Dark":
    st.markdown("<style> .stApp {{ background-color: #0e1117; color: #fafafa; }} </style>", unsafe_allow_html=True)

# === KPIs (robust guards) ===
col1, col2, col3, col4 = st.columns(4)

_last_price = None
if not fc_df.empty and "price_forecast" in fc_df.columns and len(fc_df["price_forecast"]) > 0:
    try:
        _last_price = float(fc_df["price_forecast"].iloc[-1])
    except Exception:
        _last_price = None

_risk_level = None
if not risk_df.empty:
    if "risk" in risk_df.columns and len(risk_df["risk"]) > 0:
        _risk_level = float(risk_df["risk"].iloc[-1]) * 100.0
    elif "global_risk_0_100" in risk_df.columns and len(risk_df["global_risk_0_100"]) > 0:
        _risk_level = float(risk_df["global_risk_0_100"].iloc[-1])

_hedge_notional = None
if not hedge_df.empty and "hedge_notional" in hedge_df.columns and len(hedge_df["hedge_notional"]) > 0:
    try:
        _hedge_notional = float(hedge_df["hedge_notional"].sum())
    except Exception:
        _hedge_notional = None

col1.metric("Latest Forecast Price", f"{{_last_price:,.2f}}" if _last_price is not None else "–")
col2.metric("Current Risk Index", f"{{_risk_level:,.0f}}" if _risk_level is not None else "–", help="0–100 scale")
col3.metric("Hedge Notional", f"{{_hedge_notional:,.0f}}" if _hedge_notional is not None else "–")
col4.metric("Forecast Steps", f"{{len(fc_df)}}" if not fc_df.empty else "–")

# === Price Forecast ===
st.subheader("Price forecast trajectory")
if fc_df.empty or "date" not in fc_df.columns or "price_forecast" not in fc_df.columns:
    st.info("No forecast data found. Run the market-model step first.")
else:
    st.line_chart(fc_df.set_index("date")["price_forecast"], height=280)

# === Climate Risk ===
st.subheader("Climate risk index (recent)")
if risk_df.empty or "date" not in risk_df.columns:
    st.info("No climate index found. Run the climate-index step first.")
else:
    series = None
    if "risk" in risk_df.columns:
        series = risk_df[["date","risk"]].set_index("date")
    elif "global_risk_0_100" in risk_df.columns:
        series = risk_df[["date","global_risk_0_100"]].set_index("date")
    if series is not None and not series.empty:
        st.line_chart(series.tail(104), height=200)
    else:
        st.info("Risk series not available in expected columns.")

# === Scenario (simple) ===
st.subheader("Scenario exploration")
_base = None
if not risk_df.empty:
    if "risk" in risk_df.columns and len(risk_df["risk"])>0:
        _base = float(risk_df["risk"].iloc[-1])
    elif "global_risk_0_100" in risk_df.columns and len(risk_df["global_risk_0_100"])>0:
        _base = float(risk_df["global_risk_0_100"].iloc[-1]) / 100.0
colA, colB, colC = st.columns(3)
with colA:
    up = st.slider("Stress up (+%)", 0, 100, 20, step=5, disabled=_base is None)
with colB:
    dn = st.slider("Stress down (-%)", 0, 100, 20, step=5, disabled=_base is None)
with colC:
    steps = st.selectbox("Steps", options=horiz_choice or [8], index=0)

st.caption("Scenarios apply a flat scaling to the last risk level for the chosen number of steps. (Display only)")

# === Hedge details ===
if show_hedge:
    st.subheader("Hedging recommendation")
    if hedge_df.empty:
        st.info("No hedge recommendation found. Run the hedging step first.")
    else:
        st.dataframe(hedge_df, use_container_width=True)
        st.download_button(
            label="Download hedge recommendation (CSV)",
            data=hedge_df.to_csv(index=False),
            file_name=f"{{COMMODITY}}_hedge_rec.csv",
            mime="text/csv",
        )

# === Raw Data Tabs ===
with st.expander("Raw data tables"):
    t1, t2, t3 = st.tabs(["Forecast", "Risk Index", "Hedge Rec"])
    with t1:
        st.dataframe(fc_df if not fc_df.empty else pd.DataFrame(), use_container_width=True)
    with t2:
        st.dataframe(risk_df if not risk_df.empty else pd.DataFrame(), use_container_width=True)
    with t3:
        st.dataframe(hedge_df if not hedge_df.empty else pd.DataFrame(), use_container_width=True)

st.caption("© Global Climate Hedging – Interactive Streamlit Report")
"""
    tmp_app_path = Path("interface/_report_app_tmp.py")
    tmp_app_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_app_path.write_text(app_code, encoding="utf-8")

    try:
        print("Launching Streamlit… If the browser does not open, use the URL printed below.")
        subprocess.run(["streamlit", "run", str(tmp_app_path)], check=True)
    except FileNotFoundError:
        print("\n[ERROR] Streamlit is not installed. Install it with:\n  pip install streamlit\n")
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
