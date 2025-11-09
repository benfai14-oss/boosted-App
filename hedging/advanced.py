"""
Advanced Hedging Module (ARIMAX-Integrated)
===========================================

This module implements a **dynamic hedging framework** that integrates
climate risk indicators and market price forecasts produced by the
**ARIMAX model**. It is designed for corporate treasurers, traders, and
risk managers who need to quantify and mitigate exposure to commodity
price fluctuations under climate uncertainty.

Core Features:
--------------
1. **Dynamic hedge ratio** — adapts continuously to changes in
   the global climate risk index and the direction of ARIMAX
   price forecasts.
2. **Multi-horizon forecasting** — uses 4-, 8-, and 12-week
   horizons to capture medium-term dynamics and compute
   meaningful risk metrics (mean, volatility, VaR, CVaR).
3. **Automated report generation** — produces a plain-text
   performance summary and a CSV file with the computed hedge
   ratios and positions.
4. **Interpretability** — outputs an interpretation block
   explaining the hedge effectiveness and the level of coverage.

Integration:
------------
This module connects to the ARIMAX forecasting engine
(`market_models.train.train_and_forecast`) to automatically compute
an optimal hedge position for a given exposure and role
(`importer` or `exporter`).

Typical usage (via CLI):
------------------------
    python -m interface.cli hedging-dynamic --commodity wheat --role importer --exposure 5000

Output:
-------
- `data/silver/hedging_<commodity>.csv`  → hedging ratios and positions
- `data/silver/hedging_report_<commodity>.txt`  → performance report
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from utils.risk_metrics import calculate_var, calculate_cvar
from market_models.train import train_and_forecast


# ---------------------------------------------------------------------
# 1️⃣ Dynamic hedging strategy
# ---------------------------------------------------------------------
def dynamic_hedging_strategy(
    risk_series: pd.Series,
    forecast_series: pd.Series,
    *,
    base_hedge_ratio: float = 0.5,
    high_risk_threshold: float = 70.0,
    low_risk_threshold: float = 30.0,
    role: str = "importer",
    exposure: float = 1.0,
) -> pd.DataFrame:
    """Compute a hedge ratio that adapts to climate risk and ARIMAX forecasts."""
    hedge_ratio = []
    for r, f in zip(risk_series, forecast_series):
        if np.isnan(r):
            ratio = base_hedge_ratio
        elif r >= high_risk_threshold:
            ratio = 1.0
        elif r <= low_risk_threshold:
            ratio = 0.0
        else:
            ratio = (r - low_risk_threshold) / (high_risk_threshold - low_risk_threshold)

        # Adjust with forecast direction
        if not np.isnan(f):
            if f > 0 and role == "importer":
                ratio = min(1.0, ratio + 0.1)
            elif f < 0 and role == "exporter":
                ratio = min(1.0, ratio + 0.1)

        hedge_ratio.append(ratio)

    sign = 1.0 if role == "importer" else -1.0
    hedge_position = pd.Series(hedge_ratio, index=risk_series.index) * exposure * sign

    return pd.DataFrame({
        "hedge_ratio": hedge_ratio,
        "hedge_position": hedge_position
    }, index=risk_series.index)


# ---------------------------------------------------------------------
# 2️⃣ Performance evaluation
# ---------------------------------------------------------------------
def evaluate_hedge_performance(
    portfolio_returns: pd.Series,
    hedge_returns: pd.Series,
    hedge_ratio: float,
    *,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Compute mean, volatility, VaR and CVaR of hedged returns."""
    hedged = portfolio_returns - hedge_ratio * hedge_returns
    return {
        "mean": float(hedged.mean()),
        "volatility": float(hedged.std(ddof=0)),
        "var": float(calculate_var(hedged, alpha)),
        "cvar": float(calculate_cvar(hedged, alpha)),
    }


# ---------------------------------------------------------------------
# 3️⃣ Report generation
# ---------------------------------------------------------------------
def generate_hedge_report(
    hedge_positions: pd.DataFrame,
    performance_metrics: Dict[str, float],
) -> str:
    """Format hedge results into a readable text report."""
    lines = [
        "Hedge Performance Report (ARIMAX-integrated)",
        "=" * 45,
        "",
        f"Total adjustments: {len(hedge_positions)}",
        f"Final hedge ratio: {hedge_positions['hedge_ratio'].iloc[-1]:.3f}",
        f"Final hedge position: {hedge_positions['hedge_position'].iloc[-1]:.2f}",
        "",
        "Performance Metrics:"
    ]
    for k, v in performance_metrics.items():
        lines.append(f"  {k.capitalize():<12}: {v:.4f}")

    interpretation = (
        "\nInterpretation:\n"
        f"  - Final hedge ratio = {hedge_positions['hedge_ratio'].iloc[-1]*100:.1f}% of exposure hedged"
    )
    lines.append(interpretation)
    return "\n".join(lines)


# ---------------------------------------------------------------------
# 4️⃣ Full ARIMAX-integrated pipeline
# ---------------------------------------------------------------------
def dynamic_hedging_from_pipeline(
    *,
    commodity: str,
    horizon: int = 8,
    exposure: float = 1000.0,
    role: str = "importer",
) -> Tuple[pd.DataFrame, str]:
    """
    Full ARIMAX + risk index integration with multi-horizon hedging.
    Generates forecasts for several horizons (4, 8, 12 weeks)
    to build a meaningful return distribution and risk metrics.
    """
    price_path = Path("data/silver/market/market_prices.csv")
    risk_path = Path(f"data/silver/climate_index_{commodity}.csv")

    if not price_path.exists() or not risk_path.exists():
        raise FileNotFoundError("Required input data not found under data/silver/")

    price_df = pd.read_csv(price_path)
    risk_df = pd.read_csv(risk_path)
    for df in (price_df, risk_df):
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)

    # Forecasts for multiple horizons (4, 8, 12 weeks)
    fc_df = train_and_forecast(price_df, risk_df, horizons=[28, 56, 84])
    if fc_df.empty:
        raise ValueError("Forecast dataframe is empty.")

    # Build aligned series
    risk_series = pd.Series(
        risk_df["global_risk_0_100"].iloc[-len(fc_df):].values,
        index=fc_df["horizon_days"]
    )
    forecast_series = pd.Series(fc_df["cum_return"].values, index=fc_df["horizon_days"])

    # Hedge logic
    hedge_df = dynamic_hedging_strategy(
        risk_series=risk_series,
        forecast_series=forecast_series,
        role=role,
        exposure=exposure,
    )

    # Performance metrics
    perf = evaluate_hedge_performance(
        portfolio_returns=forecast_series,
        hedge_returns=forecast_series,
        hedge_ratio=float(hedge_df["hedge_ratio"].iloc[-1]),
    )

    # Generate report
    report = generate_hedge_report(hedge_df, perf)

    # Save CSV and report (simple, comma-separated)
    out_dir = Path("data/silver")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"hedging_{commodity}.csv"
    hedge_df.to_csv(csv_path, index_label="Horizon (days)", encoding="utf-8")

    report_path = out_dir / f"hedging_report_{commodity}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Print summary
    print(f"\n=== Hedging Completed for {commodity.upper()} ===")
    print(report)
    print("\n[INFO] Results saved under:")
    print(f"  - {csv_path}")
    print(f"  - {report_path}")

    return hedge_df, report
