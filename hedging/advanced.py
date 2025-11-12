"""
Simple hedging recommendation engine with ARIMAX integration.

This module links the ARIMAX forecast results and the global climate
risk index to produce a business-level hedging recommendation.

Inputs
------
- ARIMAX forecast (latest predicted price and trend)
- Global climate risk score (0–100)
- User profile ('balanced', 'conservative', 'opportunistic')
- Role ('importer' or 'exporter')
- Exposure size

Outputs
-------
A dictionary with:
- hedge_ratio
- instrument
- hedge_notional
- summary (text explanation)
"""

from typing import Dict, Literal
import json
import pandas as pd


# --- Hedge profile configuration ---
profiles: Dict[str, Dict[str, float]] = {
    "conservative": {"base": 0.60, "bump": 0.20, "threshold": 70.0},
    "balanced": {"base": 0.45, "bump": 0.15, "threshold": 75.0},
    "opportunistic": {"base": 0.20, "bump": 0.10, "threshold": 80.0},
}


def recommend_hedge_from_arimax(
    forecast_path: str,
    risk_index_path: str,
    *,
    profile: Literal["conservative", "balanced", "opportunistic"] = "balanced",
    role: Literal["importer", "exporter"] = "importer",
    exposure: float = 10000.0,
    forecast_horizon: int = 8,
) -> Dict[str, float]:
    """
    Generate a hedging recommendation using ARIMAX forecast results
    and the global climate risk index.
    """

    # --- Load data ---
    forecast_df = pd.read_json(forecast_path)
    risk_df = pd.read_json(risk_index_path)

    if "price_forecast" not in forecast_df.columns:
        raise ValueError("Forecast JSON must contain a 'price_forecast' column.")
    if "global_risk_0_100" not in risk_df.columns:
        raise ValueError("Risk index JSON must contain 'global_risk_0_100' column.")

    # --- Extract latest forecast and risk ---
    last_forecast = forecast_df["price_forecast"].iloc[-1]
    prev_forecast = forecast_df["price_forecast"].iloc[-2] if len(forecast_df) > 1 else last_forecast
    last_risk = risk_df["global_risk_0_100"].iloc[-1]

    # --- Determine scenario based on forecast trend ---
    if last_forecast > prev_forecast * 1.02:
        scenario = "bullish"
    elif last_forecast < prev_forecast * 0.98:
        scenario = "extreme"
    else:
        scenario = "baseline"

    # --- Profile configuration ---
    if profile not in profiles:
        raise KeyError(f"Unknown profile '{profile}'. Valid: {list(profiles.keys())}")
    cfg = profiles[profile]
    hedge_ratio = cfg["base"]
    explanation = [f"Base hedge ratio for {profile} profile: {hedge_ratio:.0%}."]

    # --- Adjust for climate risk ---
    if last_risk >= cfg["threshold"]:
        hedge_ratio += cfg["bump"]
        explanation.append(
            f"Climate risk {last_risk:.1f} exceeds threshold {cfg['threshold']} → add bump {cfg['bump']:.0%}."
        )

    # --- Adjust for scenario (from ARIMAX trend) ---
    if scenario == "bullish":
        hedge_ratio += 0.05
        explanation.append("Bullish ARIMAX forecast → increase hedge ratio by 5 pp.")
    elif scenario == "extreme":
        hedge_ratio += 0.10
        explanation.append("Extreme ARIMAX forecast → increase hedge ratio by 10 pp.")

    # --- Final hedge ratio ---
    hedge_ratio = min(max(hedge_ratio, 0.0), 1.0)

    # --- Determine instrument ---
    if role == "importer":
        instrument = "long futures" if scenario != "baseline" else "long call options"
    else:
        instrument = "short futures" if scenario != "baseline" else "long put options"

    hedge_notional = hedge_ratio * exposure
    explanation.append(
        f"Role: {role} → use '{instrument}'. Exposure {exposure} → hedge {hedge_notional:.2f} units."
    )

    # --- Full structured output ---
    result = {
        "commodity": forecast_path.split("/")[-1].replace("_forecast.json", ""),
        "profile": profile,
        "role": role,
        "exposure": exposure,
        "forecast_horizon_weeks": forecast_horizon,
        "scenario": scenario,
        "risk_score": float(last_risk),
        "hedge_ratio": float(hedge_ratio),
        "hedge_notional": float(hedge_notional),
        "instrument": instrument,
        "forecast_price": float(last_forecast),
        "summary": " ".join(explanation),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }

    # --- Save output next to forecast file ---
    out_path = forecast_path.replace("_forecast.json", "_hedge_rec.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Hedging recommendation saved to {out_path}")

    return result


# =========================================================
# COMMAND-LINE ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate hedging recommendation from ARIMAX forecast + risk index.")
    parser.add_argument("--forecast", required=True, help="Path to forecast JSON (e.g. data/gold/soybean_forecast.json)")
    parser.add_argument("--risk", required=True, help="Path to risk index JSON (e.g. data/gold/soybean_global_index.json)")
    parser.add_argument("--profile", choices=["conservative", "balanced", "opportunistic"], default="balanced")
    parser.add_argument("--role", choices=["importer", "exporter"], default="importer")
    parser.add_argument("--exposure", type=float, default=10000.0)
    parser.add_argument("--horizon", type=int, default=8, help="Forecast horizon in weeks")

    args = parser.parse_args()

    result = recommend_hedge_from_arimax(
        forecast_path=args.forecast,
        risk_index_path=args.risk,
        profile=args.profile,
        role=args.role,
        exposure=args.exposure,
        forecast_horizon=args.horizon,
    )

    print("\n=== Hedging Recommendation ===")
    print(json.dumps(result, indent=2))
