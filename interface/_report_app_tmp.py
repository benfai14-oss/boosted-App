
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Global Climate Hedging – Report", layout="wide")

FORECAST_PATH = Path(r"data/gold/wheat_forecast.json")
RISK_PATH     = Path(r"data/gold/wheat_global_index.json")
HEDGE_PATH    = Path(r"data/gold/wheat_hedge_rec.json")
COMMODITY     = "wheat"

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

st.title(f"{COMMODITY.capitalize()} – Climate Hedging Dashboard")

# =========================
# KPIs
# =========================
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

col1.metric("Latest Forecast Price", f"{_last_price:,.2f}" if _last_price is not None else "–")
col2.metric("Current Risk Index", f"{_risk_level:,.0f}" if _risk_level is not None else "–")
col3.metric("Hedge Notional", f"{_hedge_notional:,.0f}" if _hedge_notional is not None else "–")
col4.metric("Forecast Steps", f"{len(fc_df)}" if not fc_df.empty else "–")

# =========================
# Graphique : Price forecast + bandes 95%
# =========================
st.subheader("Price forecast with 95% confidence band")

if fc_df.empty or "date" not in fc_df.columns or "price_forecast" not in fc_df.columns:
    st.info("No forecast data found. Run the market-model step first.")
else:
    try:
        import altair as alt

        base = alt.Chart(fc_df).encode(
            x=alt.X("date:T", title="Date")
        )

        if "price_lo95" in fc_df.columns and "price_hi95" in fc_df.columns:
            band = base.mark_area(opacity=0.2).encode(
                y=alt.Y("price_lo95:Q", title="Price"),
                y2="price_hi95:Q"
            )
            line = base.mark_line(color="#1f77b4").encode(
                y="price_forecast:Q"
            )
            chart = band + line
        else:
            chart = base.mark_line(color="#1f77b4").encode(
                y=alt.Y("price_forecast:Q", title="Price")
            )

        st.altair_chart(chart.properties(height=300).interactive(), use_container_width=True)
    except Exception:
        # fallback très simple si altair n'est pas dispo
        st.line_chart(fc_df.set_index("date")[["price_forecast"]], height=300)

# =========================
# Graphique : Climate risk index
# =========================
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
        st.line_chart(series.tail(104), height=220)
    else:
        st.info("Risk series not available in expected columns.")

# =========================
# Hedging recommendation
# =========================
st.subheader("Hedging recommendation")

if hedge_df.empty:
    st.info("No hedge recommendation found. Run the hedging step first.")
else:
    st.dataframe(hedge_df, use_container_width=True)
    st.download_button(
        label="Download hedge recommendation (CSV)",
        data=hedge_df.to_csv(index=False),
        file_name=f"{COMMODITY}_hedge_rec.csv",
        mime="text/csv",
    )

# =========================
# Raw data tables
# =========================
with st.expander("Raw data tables"):
    t1, t2, t3 = st.tabs(["Forecast", "Risk Index", "Hedge Rec"])
    with t1:
        st.dataframe(fc_df if not fc_df.empty else pd.DataFrame(), use_container_width=True)
    with t2:
        st.dataframe(risk_df if not risk_df.empty else pd.DataFrame(), use_container_width=True)
    with t3:
        st.dataframe(hedge_df if not hedge_df.empty else pd.DataFrame(), use_container_width=True)

st.caption("© Global Climate Hedging – Streamlit Report")
