import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

try:
    import altair as alt
except ImportError:
    alt = None

# --- Commodity passed from CLI via env ---
COMMODITY = os.getenv("HEDGE_COMMODITY", "wheat")


# =========================
# Helpers
# =========================
def _safe_load_json(path: Path) -> pd.DataFrame:
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


def _safe_load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _apply_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
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
# Load data
# =========================
FORECAST_PATH = Path(f"data/gold/{COMMODITY}_forecast.json")
RISK_PATH = Path(f"data/gold/{COMMODITY}_global_index.json")
HEDGE_PATH = Path(f"data/gold/{COMMODITY}_hedge_rec.json")
MARKET_PATH = Path("data/silver/market/market_prices.csv")

fc_df = _safe_load_json(FORECAST_PATH)
risk_df = _safe_load_json(RISK_PATH)
hedge_df = _safe_load_json(HEDGE_PATH)
market_df = _safe_load_csv(MARKET_PATH)

for df in (fc_df, risk_df, market_df):
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values("date", inplace=True)

if not market_df.empty and "commodity" in market_df.columns:
    market_df = market_df[market_df["commodity"].str.lower() == COMMODITY.lower()]

# =========================
# Layout & title
# =========================
st.set_page_config(page_title="Global Climate Hedging – Report", layout="wide")
st.title(f"{COMMODITY.capitalize()} – Climate Hedging Dashboard")


# =========================
# KPIs
# =========================
col1, col2, col3, col4, col5 = st.columns(5)

# ---- Latest forecast price ----
_last_price = None
if not fc_df.empty and "price_forecast" in fc_df.columns and len(fc_df["price_forecast"]) > 0:
    try:
        _last_price = float(fc_df["price_forecast"].iloc[-1])
    except Exception:
        _last_price = None

# ---- Current risk score (one decimal) ----
_risk_level = None
if not risk_df.empty:
    if "risk" in risk_df.columns and len(risk_df["risk"]) > 0:
        _risk_level = float(risk_df["risk"].iloc[-1]) * 100.0
    elif "global_risk_0_100" in risk_df.columns and len(risk_df["global_risk_0_100"]) > 0:
        _risk_level = float(risk_df["global_risk_0_100"].iloc[-1])

# ---- Hedge notional ----
_hedge_notional = None
if not hedge_df.empty and "hedge_notional" in hedge_df.columns and len(hedge_df["hedge_notional"]) > 0:
    try:
        _hedge_notional = float(hedge_df["hedge_notional"].sum())
    except Exception:
        _hedge_notional = None

# ---- Hedge ratio (0–1) -> % ----
_hedge_ratio = None
if not hedge_df.empty and "hedge_ratio" in hedge_df.columns and len(hedge_df["hedge_ratio"]) > 0:
    try:
        _hedge_ratio = float(hedge_df["hedge_ratio"].iloc[0])
    except Exception:
        _hedge_ratio = None

# ---- ARIMAX scenario + forecast change (%) ----
_arimax_scenario = None
_arimax_change_pct = None

if not hedge_df.empty and "scenario" in hedge_df.columns and len(hedge_df["scenario"]) > 0:
    _arimax_scenario = str(hedge_df["scenario"].iloc[0])

if not fc_df.empty and "price_forecast" in fc_df.columns and len(fc_df["price_forecast"]) >= 2:
    try:
        first_fc = float(fc_df["price_forecast"].iloc[0])
        last_fc = float(fc_df["price_forecast"].iloc[-1])
        if first_fc != 0:
            _arimax_change_pct = (last_fc / first_fc - 1.0) * 100.0
    except Exception:
        _arimax_change_pct = None

# ---- Display KPIs ----
col1.metric(
    "Latest forecast price",
    f"{_last_price:,.2f}" if _last_price is not None else "–",
)

col2.metric(
    "Current risk score",
    f"{_risk_level:,.1f}" if _risk_level is not None else "–",
)

col3.metric(
    "Hedge notional",
    f"{_hedge_notional:,.0f}" if _hedge_notional is not None else "–",
)

col4.metric(
    "Hedge ratio",
    f"{_hedge_ratio * 100:.0f}%" if _hedge_ratio is not None else "–",
)

if _arimax_scenario is not None:
    col5.metric(
        "ARIMAX scenario",
        _arimax_scenario.capitalize(),
        f"{_arimax_change_pct:+.1f}%" if _arimax_change_pct is not None else "",
    )
else:
    col5.metric("ARIMAX scenario", "–")


# =========================
# TOP ROW: Hedge (left) + Forecast charts (right)
# =========================
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Hedging recommendation")

    if hedge_df.empty:
        st.info("No hedge recommendation found. Run the hedging step first.")
    else:
        # Take the first row as the main recommendation
        row = hedge_df.iloc[0].to_dict()

        # Vertical key/value display
        fields = [
            ("Commodity", "commodity"),
            ("Profile", "profile"),
            ("Role", "role"),
            ("Exposure", "exposure"),
            ("Hedge notional", "hedge_notional"),
            ("Instrument", "instrument"),
            ("Maturity", "instrument_maturity"),
            ("Strike", "instrument_strike"),
        ]

        for label, key in fields:
            value = row.get(key, "–")
            c1, c2 = st.columns([1, 2])
            c1.write(f"**{label}**")
            c2.write(str(value))

        # Justification paragraph (business explanation)
        justification = row.get("justification") or row.get("summary")
        if justification:
            st.markdown("**Rationale**")
            st.write(justification)

        # If there are multiple legs, mention it
        if len(hedge_df) > 1:
            st.caption(f"{len(hedge_df)} hedge legs in total (showing primary leg above).")

        st.download_button(
            label="Download full hedge recommendation (CSV)",
            data=hedge_df.to_csv(index=False),
            file_name=f"{COMMODITY}_hedge_rec.csv",
            mime="text/csv",
        )

with right_col:
    st.subheader("Price forecast with 95% confidence band")
    if fc_df.empty or "date" not in fc_df.columns or "price_forecast" not in fc_df.columns:
        st.info("No forecast data found. Run the market-model step first.")
    else:
        if alt is not None:
            # --- Compute tight y-axis domain based on forecasts (and bands if available) ---
            y_values = []

            if "price_forecast" in fc_df.columns:
                y_values.extend(fc_df["price_forecast"].dropna().tolist())
            if "price_lo95" in fc_df.columns:
                y_values.extend(fc_df["price_lo95"].dropna().tolist())
            if "price_hi95" in fc_df.columns:
                y_values.extend(fc_df["price_hi95"].dropna().tolist())

            if y_values:
                y_min = min(y_values)
                y_max = max(y_values)
                if y_max == y_min:
                    # éviter une échelle dégénérée
                    y_min -= 1
                    y_max += 1
                pad = (y_max - y_min) * 0.05  # 5% margin
                dom_min = y_min - pad
                dom_max = y_max + pad
            else:
                dom_min = None
                dom_max = None

            base_fc = alt.Chart(fc_df).encode(
                x=alt.X("date:T", title="Date"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("price_forecast:Q", title="Forecast price", format=",.2f"),
                    alt.Tooltip("price_lo95:Q", title="Lower 95%", format=",.2f"),
                    alt.Tooltip("price_hi95:Q", title="Upper 95%", format=",.2f"),
                ],
            )

            if "price_lo95" in fc_df.columns and "price_hi95" in fc_df.columns:
                band = base_fc.mark_area(opacity=0.2, color="#1f77b4").encode(
                    y=alt.Y(
                        "price_lo95:Q",
                        title="Price",
                        scale=alt.Scale(domain=[dom_min, dom_max]) if dom_min is not None else alt.Undefined,
                    ),
                    y2="price_hi95:Q",
                )
                line = base_fc.mark_line(color="#1f77b4").encode(
                    y=alt.Y(
                        "price_forecast:Q",
                        scale=alt.Scale(domain=[dom_min, dom_max]) if dom_min is not None else alt.Undefined,
                    )
                )
                chart_fc = band + line
            else:
                chart_fc = base_fc.mark_line(color="#1f77b4").encode(
                    y=alt.Y(
                        "price_forecast:Q",
                        title="Price",
                        scale=alt.Scale(domain=[dom_min, dom_max]) if dom_min is not None else alt.Undefined,
                    )
                )

            st.altair_chart(chart_fc.properties(height=230).interactive(), use_container_width=True)
        else:
            st.line_chart(fc_df.set_index("date")[["price_forecast"]], height=230)

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

        if alt is not None:
            # --- Compute tight y-axis domain for percentage changes ---
            y_values_pct = []

            if "ret_pct" in fc_pct.columns:
                y_values_pct.extend(fc_pct["ret_pct"].dropna().tolist())
            if "ret_lo95" in fc_pct.columns:
                y_values_pct.extend(fc_pct["ret_lo95"].dropna().tolist())
            if "ret_hi95" in fc_pct.columns:
                y_values_pct.extend(fc_pct["ret_hi95"].dropna().tolist())

            if y_values_pct:
                y_min_pct = min(y_values_pct)
                y_max_pct = max(y_values_pct)
                if y_max_pct == y_min_pct:
                    y_min_pct -= 0.1
                    y_max_pct += 0.1
                pad_pct = (y_max_pct - y_min_pct) * 0.1  # 10% margin
                dom_min_pct = y_min_pct - pad_pct
                dom_max_pct = y_max_pct + pad_pct
            else:
                dom_min_pct = None
                dom_max_pct = None

            base_pct = alt.Chart(fc_pct).encode(
                x=alt.X("date:T", title="Date"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("ret_pct:Q", title="Forecast change (%)", format=".1f"),
                    alt.Tooltip("ret_lo95:Q", title="Lower 95% (%)", format=".1f"),
                    alt.Tooltip("ret_hi95:Q", title="Upper 95% (%)", format=".1f"),
                ],
            )

            if "ret_lo95" in fc_pct.columns and "ret_hi95" in fc_pct.columns:
                band_pct = base_pct.mark_area(opacity=0.2, color="#2ca02c").encode(
                    y=alt.Y(
                        "ret_lo95:Q",
                        title="Price change (%)",
                        scale=alt.Scale(domain=[dom_min_pct, dom_max_pct]) if dom_min_pct is not None else alt.Undefined,
                    ),
                    y2="ret_hi95:Q",
                )
                line_pct = base_pct.mark_line(color="#2ca02c").encode(
                    y=alt.Y(
                        "ret_pct:Q",
                        scale=alt.Scale(domain=[dom_min_pct, dom_max_pct]) if dom_min_pct is not None else alt.Undefined,
                    )
                )
                chart_pct = band_pct + line_pct
            else:
                chart_pct = base_pct.mark_line(color="#2ca02c").encode(
                    y=alt.Y(
                        "ret_pct:Q",
                        title="Price change (%)",
                        scale=alt.Scale(domain=[dom_min_pct, dom_max_pct]) if dom_min_pct is not None else alt.Undefined,
                    )
                )

            st.altair_chart(chart_pct.properties(height=230).interactive(), use_container_width=True)
        else:
            st.line_chart(fc_pct.set_index("date")[["ret_pct"]], height=230)


# =========================
# MIDDLE: History charts (price + risk)
# =========================
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
        price_window = st.selectbox(
            "Price history window",
            options=["12M", "24M", "All"],
            index=0,
            help="Time window used for the price history chart.",
        )

        hist_price = _apply_window(market_df.copy(), price_window)
        if alt is not None:
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
        else:
            st.line_chart(hist_price.set_index("date")[[price_col]], height=250)

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
            help="Time window used for the climate risk history chart.",
        )

        hist_risk = _apply_window(series_df.copy(), risk_window)
        if alt is not None:
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
        else:
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