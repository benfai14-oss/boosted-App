"""
hedging.advanced
-------------------

Advanced hedging engine connected to the ARIMAX model and climate risk index.

This version:
1. Loads price data from silver_data.csv
2. Loads risk data from climate_index_<commodity>.csv
3. Loads ARIMAX parameters
4. Produces forecasts using ARIMAX coefficients (no refit)
5. Builds a dynamic hedge strategy based on risk levels
6. Evaluates hedge performance and generates a report
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Tuple
from market_models import arimax
from utils.risk_metrics import value_at_risk, conditional_value_at_risk


# === SAFE CSV READER ===================================================
def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV even if it's badly formatted (all in one column)."""
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",", engine="python")
            if df.shape[1] == 1:
                df = pd.read_csv(path, sep=";", engine="python")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")


# === MAIN DYNAMIC HEDGING STRATEGY ====================================
def dynamic_hedging_from_pipeline(
    commodity: str,
    *,
    horizon: int = 8,
    exposure: float = 1000.0,
    role: str = "importer",
    high_risk_threshold: float = 70.0,
    low_risk_threshold: float = 30.0,
) -> Tuple[pd.DataFrame, str]:

    silver_path = Path("data/silver/silver_data.csv")
    risk_path = Path(f"data/silver/climate_index_{commodity}.csv")
    arimax_path = Path(f"data/models/arimax_{commodity}.yaml")

    if not silver_path.exists():
        raise FileNotFoundError(f"Silver data not found: {silver_path}")
    if not risk_path.exists():
        raise FileNotFoundError(f"Climate risk file not found: {risk_path}")
    if not arimax_path.exists():
        raise FileNotFoundError(f"ARIMAX parameters not found: {arimax_path}")

    # --- Load and clean market data ---
    df_market = safe_read_csv(silver_path)
    df_market.columns = [c.strip().lower() for c in df_market.columns]
    price_col = next((c for c in df_market.columns if "price_spot" in c or "price" in c), None)
    if not price_col:
        raise ValueError("No 'price_spot' column found in market data.")
    df_market = df_market.rename(columns={price_col: "price"})
    if "commodity" in df_market.columns:
        df_market = df_market[df_market["commodity"].str.lower() == commodity.lower()].copy()

    df_market = df_market.dropna(subset=["price"])
    if "date" not in df_market.columns:
        raise ValueError("No 'date' column found in silver data.")
    df_market["date"] = pd.to_datetime(df_market["date"])
    df_market = df_market.sort_values("date").reset_index(drop=True)

    # --- Load risk (climate index) ---
    df_risk = safe_read_csv(risk_path)
    df_risk.columns = [c.strip().lower() for c in df_risk.columns]
    risk_col = next((c for c in df_risk.columns if "risk" in c), None)
    if not risk_col:
        raise ValueError("No 'risk' column found in climate index data.")
    df_risk = df_risk.rename(columns={risk_col: "risk"})
    df_risk["date"] = pd.to_datetime(df_risk["date"])

    # --- Merge price + risk on date ---
    df = pd.merge(df_market, df_risk[["date", "risk"]], on="date", how="inner")
    if df.empty:
        raise ValueError("Merged dataset is empty — check date alignment.")

    # --- Load ARIMAX parameters ---
    with open(arimax_path, "r") as f:
        params = yaml.safe_load(f)

    # --- Forecast using ARIMAX model ---
    last_price = df["price"].iloc[-1]
    last_row = df.iloc[-1]
    risk_series = df["risk"].tail(horizon).fillna(50).reset_index(drop=True)

    forecasts, forecast_prices = arimax.forecast_arimax(
        last_price=last_price,
        last_row=last_row,
        params=params,
        risk_future=risk_series.to_list(),
        horizon=horizon,
    )

    if len(forecast_prices) == 0:
        raise ValueError("ARIMAX forecast returned no results — check model coefficients.")

    # --- Compute optimal hedge ratio ---
    returns = np.log(df["price"]).diff().dropna()
    hedge_ratio_opt = optimize_hedge_ratio(returns, returns.shift(1).dropna())

    # --- Adjust dynamically by risk ---
    hedge_ratio = []
    for r in risk_series:
        if r >= high_risk_threshold:
            ratio = min(1.0, hedge_ratio_opt * 1.25)
        elif r <= low_risk_threshold:
            ratio = max(0.0, hedge_ratio_opt * 0.75)
        else:
            scale = (r - low_risk_threshold) / (high_risk_threshold - low_risk_threshold)
            ratio = hedge_ratio_opt * (0.75 + 0.5 * scale)
        hedge_ratio.append(ratio)
    hedge_ratio = np.clip(hedge_ratio, 0, 1)

    # --- Hedge positions ---
    sign = 1.0 if role == "importer" else -1.0
    hedge_position = np.array(hedge_ratio) * exposure * sign

    hedge_df = pd.DataFrame({
        "forecast_return": forecasts,
        "forecast_price": forecast_prices,
        "risk": risk_series,
        "hedge_ratio": hedge_ratio,
        "hedge_position": hedge_position,
    })
    hedge_df.attrs["hedge_ratio_opt"] = hedge_ratio_opt

    # --- Evaluate performance ---
    perf = evaluate_hedge_performance(returns, returns.shift(1).dropna(), hedge_ratio_opt)
    report = generate_hedge_report(hedge_df, perf)

    out_path = Path(f"data/silver/hedging_{commodity}.csv")
    hedge_df.to_csv(out_path, index=False)
    print(f"Hedging results saved to {out_path}")

    return hedge_df, report


# === HELPER FUNCTIONS ==================================================
def optimize_hedge_ratio(target_returns: pd.Series, hedge_returns: pd.Series) -> float:
    joined = pd.concat([target_returns, hedge_returns], axis=1).dropna()
    if joined.empty:
        return 0.0
    y, x = joined.iloc[:, 0].values, joined.iloc[:, 1].values
    cov_xy, var_x = np.cov(x, y, ddof=0)[0, 1], np.var(x, ddof=0)
    return float(cov_xy / var_x) if var_x != 0 else 0.0


def evaluate_hedge_performance(portfolio_returns, hedge_returns, hedge_ratio, *, alpha=0.05):
    hedged = portfolio_returns - hedge_ratio * hedge_returns
    return {
        "mean": float(hedged.mean()),
        "volatility": float(hedged.std(ddof=0)),
        "var": float(value_at_risk(hedged, 1 - alpha)),
        "cvar": float(conditional_value_at_risk(hedged, 1 - alpha)),
    }


def generate_hedge_report(hedge_df: pd.DataFrame, performance: Dict[str, float]) -> str:
    hedge_opt = hedge_df.attrs.get("hedge_ratio_opt", None)
    lines = [
        "=== Hedge Performance Report ===",
        f"Total forecast periods: {len(hedge_df)}",
        f"Optimal hedge ratio: {hedge_opt:.3f}" if hedge_opt is not None else "",
        f"Final dynamic hedge ratio: {hedge_df['hedge_ratio'].iloc[-1]:.3f}",
        f"Final hedge position: {hedge_df['hedge_position'].iloc[-1]:.2f}",
        "",
        "Performance Metrics:",
        f"  Mean return     : {performance['mean']:.4f}",
        f"  Volatility      : {performance['volatility']:.4f}",
        f"  Value-at-Risk   : {performance['var']:.4f}",
        f"  Conditional VaR : {performance['cvar']:.4f}",
    ]
    return "\n".join([l for l in lines if l])
