
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Global Climate Hedging – Report", layout="wide")

FORECAST_PATH = Path(r"data/gold/wheat_forecast.json")
RISK_PATH     = Path(r"data/gold/wheat_global_index.json")
HEDGE_PATH    = Path(r"data/gold/wheat_hedge_rec.json")
MARKET_PATH   = Path(r"data/silver/market/market_prices.csv")
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

def _safe_load_csv(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# --- Load data ---
fc_df     = _safe_load_json(FORECAST_PATH)
risk_df   = _safe_load_json(RISK_PATH)
hedge_df  = _safe_load_json(HEDGE_PATH)
market_df = _safe_load_csv(MARKET_PATH)

# Normalise dates
for df in (fc_df, risk_df, market_df):
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values("date", inplace=True)

# Filter commodity in market history if present
if not market_df.empty and "commodity" in market_df.columns:
    market_df = market_df[market_df["commodity"].str.lower() == COMMODITY.lower()]

st.title(f"{COMMODITY.capitalize()} – Climate Hedging Dashboard")

def _apply_window(df, window):
    if df.empty or "date" not in df.columns:
        return df
    if window == "All":
        return df
    try:
        months = int(window.replace("M", ""))
        last_date = df["date"].max()
        cutoff = last_date - pd.DateOffset(months=months)
        return df[df["date"] >= cutoff]
    except Exception:
        return df

# =========================
# KPIs (3 tiles)
# =========================
col1, col2, col3 = st.columns(3)

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

col1.metric("Latest forecast price", f"{_last_price:,.2f}" if _last_price is not None else "–")
col2.metric("Current risk index", f"{_risk_level:,.0f}" if _risk_level is not None else "–")
col3.metric("Hedge notional", f"{_hedge_notional:,.0f}" if _hedge_notional is not None else "–")

# =========================
# TOP ROW: Hedging (left) + Forecast charts (right)
# =========================
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Hedging recommendation")
    if hedge_df.empty:
        st.info("No hedge recommendation found. Run the hedging step first.")
    else:
        # Horizontal listing: one row per recommendation, columns as fields
        st.dataframe(hedge_df.reset_index(drop=True), use_container_width=True)
        st.download_button(
            label="Download hedge recommendation (CSV)",
            data=hedge_df.to_csv(index=False),
            file_name=f"{COMMODITY}_hedge_rec.csv",
            mime="text/csv",
        )

with right_col:
    # --- Forecast level + bands, with zoomed y-scale ---
    st.subheader("Price forecast with 95% confidence band")
    if fc_df.empty or "date" not in fc_df.columns or "price_forecast" not in fc_df.columns:
        st.info("No forecast data found. Run the market-model step first.")
    else:
        try:
            import altair as alt

            fc_plot = fc_df.dropna(subset=["price_forecast"]).copy()
            # Compute a tight y-range around forecast + bands
            y_arrays = [fc_plot["price_forecast"].values]
            if "price_lo95" in fc_plot.columns and "price_hi95" in fc_plot.columns:
                y_arrays.append(fc_plot["price_lo95"].values)
                y_arrays.append(fc_plot["price_hi95"].values)
            y_min = float(min(arr.min() for arr in y_arrays))
            y_max = float(max(arr.max() for arr in y_arrays))
            margin = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
            y_domain = [y_min - margin, y_max + margin]

            base_fc = alt.Chart(fc_plot).encode(
                x=alt.X("date:T", title="Date")
            )

            if "price_lo95" in fc_plot.columns and "price_hi95" in fc_plot.columns:
                band = base_fc.mark_area(opacity=0.2, color="#1f77b4").encode(
                    y=alt.Y("price_lo95:Q", title="Price", scale=alt.Scale(domain=y_domain)),
                    y2="price_hi95:Q"
                )
                line = base_fc.mark_line(color="#1f77b4").encode(
                    y=alt.Y("price_forecast:Q", scale=alt.Scale(domain=y_domain))
                )
                chart_fc = band + line
            else:
                chart_fc = base_fc.mark_line(color="#1f77b4").encode(
                    y=alt.Y("price_forecast:Q", title="Price", scale=alt.Scale(domain=y_domain))
                )

            st.altair_chart(chart_fc.properties(height=230).interactive(), use_container_width=True)
        except Exception:
            st.line_chart(fc_df.set_index("date")[["price_forecast"]], height=230)

    # --- Forecast % change + bands, with zoomed y-scale ---
    st.subheader("Forecasted price change (%) with 95% band")
    if fc_df.empty or "date" not in fc_df.columns or "price_forecast" not in fc_df.columns or _last_price is None or _last_price == 0:
        st.info("Cannot compute percentage forecast (missing last price or forecast).")
    else:
        fc_pct = fc_df.copy()
        base_price = _last_price
        fc_pct["ret_pct"] = (fc_pct["price_forecast"] / base_price - 1.0) * 100.0
        if "price_lo95" in fc_pct.columns and "price_hi95" in fc_pct.columns:
            fc_pct["ret_lo95"] = (fc_pct["price_lo95"] / base_price - 1.0) * 100.0
            fc_pct["ret_hi95"] = (fc_pct["price_hi95"] / base_price - 1.0) * 100.0

        try:
            import altair as alt

            fc_pct_plot = fc_pct.dropna(subset=["ret_pct"]).copy()
            y_arrays = [fc_pct_plot["ret_pct"].values]
            if "ret_lo95" in fc_pct_plot.columns and "ret_hi95" in fc_pct_plot.columns:
                y_arrays.append(fc_pct_plot["ret_lo95"].values)
                y_arrays.append(fc_pct_plot["ret_hi95"].values)
            y_min = float(min(arr.min() for arr in y_arrays))
            y_max = float(max(arr.max() for arr in y_arrays))
            margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.5
            y_domain = [y_min - margin, y_max + margin]

            base_pct = alt.Chart(fc_pct_plot).encode(
                x=alt.X("date:T", title="Date")
            )

            if "ret_lo95" in fc_pct_plot.columns and "ret_hi95" in fc_pct_plot.columns:
                band_pct = base_pct.mark_area(opacity=0.2, color="#2ca02c").encode(
                    y=alt.Y("ret_lo95:Q", title="Price change (%)", scale=alt.Scale(domain=y_domain)),
                    y2="ret_hi95:Q"
                )
                line_pct = base_pct.mark_line(color="#2ca02c").encode(
                    y=alt.Y("ret_pct:Q", scale=alt.Scale(domain=y_domain))
                )
                chart_pct = band_pct + line_pct
            else:
                chart_pct = base_pct.mark_line(color="#2ca02c").encode(
                    y=alt.Y("ret_pct:Q", title="Price change (%)", scale=alt.Scale(domain=y_domain))
                )

            st.altair_chart(chart_pct.properties(height=230).interactive(), use_container_width=True)
        except Exception:
            st.line_chart(fc_pct.set_index("date")[["ret_pct"]], height=230)

# =========================
# MIDDLE: History charts (with zoom controls under each chart)
# =========================

# --- Price history ---
st.subheader("Price history")

if market_df.empty or "date" not in market_df.columns:
    st.info("No market history found (data/silver/market/market_prices.csv).")
else:
    price_col = None
    for c in ["price_spot", "price_front_fut", "settlement", "close"]:
        if c in market_df.columns:
            price_col = c
            break

    if price_col is None:
        st.info("No usable price column found in market data.")
    else:
        # Zoom selector just under the chart
        price_window = st.selectbox(
            "Price history window",
            options=["12M", "24M", "All"],
            index=0,
            help="Time window used for the price history chart."
        )

        hist_price = _apply_window(market_df.copy(), price_window)
        try:
            import altair as alt
            chart_hist = (
                alt.Chart(hist_price)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y(price_col + ":Q", title="Price"),
                    tooltip=["date:T", price_col + ":Q"],
                )
                .properties(height=250)
                .interactive()
            )
            st.altair_chart(chart_hist, use_container_width=True)
        except Exception:
            st.line_chart(hist_price.set_index("date")[[price_col]], height=250)

# --- Climate risk history ---
st.subheader("Climate risk index (history)")

if risk_df.empty or "date" not in risk_df.columns:
    st.info("No climate index found. Run the climate-index step first.")
else:
    series_df = None
    if "risk" in risk_df.columns:
        series_df = risk_df[["date", "risk"]].rename(columns={"risk": "risk_value"})
    elif "global_risk_0_100" in risk_df.columns:
        series_df = risk_df[["date", "global_risk_0_100"]].rename(columns={"global_risk_0_100": "risk_value"})

    if series_df is not None and not series_df.empty:
        risk_window = st.selectbox(
            "Risk index history window",
            options=["12M", "24M", "All"],
            index=0,
            help="Time window used for the climate risk history chart."
        )

        hist_risk = _apply_window(series_df.copy(), risk_window)
        try:
            import altair as alt
            chart_risk = (
                alt.Chart(hist_risk)
                .mark_line(color="#d62728")
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("risk_value:Q", title="Risk index"),
                    tooltip=["date:T", "risk_value:Q"],
                )
                .properties(height=230)
                .interactive()
            )
            st.altair_chart(chart_risk, use_container_width=True)
        except Exception:
            st.line_chart(hist_risk.set_index("date")[["risk_value"]], height=230)
    else:
        st.info("Risk series not available in expected columns.")

# =========================
# BOTTOM: Raw data tables
# =========================
with st.expander("Raw data tables"):
    t1, t2, t3 = st.tabs(["Forecast", "Risk index", "Hedge recommendation"])
    with t1:
        st.dataframe(fc_df if not fc_df.empty else pd.DataFrame(), use_container_width=True)
    with t2:
        st.dataframe(risk_df if not risk_df.empty else pd.DataFrame(), use_container_width=True)
    with t3:
        st.dataframe(hedge_df if not hedge_df.empty else pd.DataFrame(), use_container_width=True)

st.caption("© Global Climate Hedging – Streamlit Report")
