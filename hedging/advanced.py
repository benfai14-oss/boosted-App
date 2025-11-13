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

from datetime import datetime, timedelta
from typing import Dict, Literal
import json
import pandas as pd


# --- Hedge profile configuration ---
profiles: Dict[str, Dict[str, float]] = {
    "conservative": {"base": 0.60, "bump": 0.20, "threshold": 70.0},
    "balanced": {"base": 0.45, "bump": 0.15, "threshold": 75.0},
    "opportunistic": {"base": 0.20, "bump": 0.10, "threshold": 80.0},
}


from typing import Literal, Dict

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
    Build a simple hedge recommendation from ARIMAX forecast + climate risk.

    Now also proposes:
      - an indicative maturity (today + forecast_horizon in weeks)
      - an indicative strike (based on last forecast price)
      - a short textual justification explaining the choice
    """
    # --- Load forecast and risk index ---
    import pandas as pd
    import json
    from pathlib import Path

    forecast_df = pd.read_json(forecast_path)
    risk_df = pd.read_json(risk_index_path)

    if "price_forecast" not in forecast_df.columns:
        raise KeyError("price_forecast column missing in forecast data.")
    if "global_risk_0_100" not in risk_df.columns:
        raise KeyError("global_risk_0_100 column missing in risk index data.")

    # Ensure sorted by date if available
    for df in (forecast_df, risk_df):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.sort_values("date", inplace=True)

    last_forecast = float(forecast_df["price_forecast"].iloc[-1])
    if len(forecast_df) >= 2:
        prev_forecast = float(forecast_df["price_forecast"].iloc[-2])
    else:
        prev_forecast = last_forecast

    last_risk = float(risk_df["global_risk_0_100"].iloc[-1])

    # --- Scenario from ARIMAX trend ---
    if last_forecast > prev_forecast * 1.02:
        scenario = "bullish"
    elif last_forecast < prev_forecast * 0.98:
        scenario = "extreme"
    else:
        scenario = "baseline"

    # --- Profile configuration ---
    profiles = {
        "conservative": {"base": 0.60, "bump": 0.20, "threshold": 70.0},
        "balanced":     {"base": 0.45, "bump": 0.15, "threshold": 75.0},
        "opportunistic":{"base": 0.20, "bump": 0.10, "threshold": 80.0},
    }
    cfg = profiles[profile]

    # --- Build hedge ratio step by step (keep explanations) ---
    hedge_ratio = cfg["base"]
    explanation = [
        f"Base hedge ratio for {profile} profile: {hedge_ratio:.0%}."
    ]

    # Climate risk effect
    if last_risk >= cfg["threshold"]:
        hedge_ratio += cfg["bump"]
        explanation.append(
            f"Climate risk {last_risk:.1f} exceeds profile threshold {cfg['threshold']:.1f} → add bump {cfg['bump']:.0%}."
        )
    else:
        explanation.append(
            f"Climate risk {last_risk:.1f} is below profile threshold {cfg['threshold']:.1f} → no additional bump."
        )

    # ARIMAX scenario effect
    if scenario == "bullish":
        hedge_ratio += 0.05
        explanation.append("Bullish ARIMAX forecast (price trending up) → increase hedge ratio by 5 pp.")
    elif scenario == "extreme":
        hedge_ratio += 0.10
        explanation.append("Extreme ARIMAX forecast (sharp move) → increase hedge ratio by 10 pp.")
    else:
        explanation.append("Baseline ARIMAX forecast (no strong trend) → keep profile hedge ratio unchanged.")

    # Bound between 0 and 1
    hedge_ratio = min(max(hedge_ratio, 0.0), 1.0)

    # --- Instrument choice based on role + scenario ---
    if role == "importer":
        # Importer is hurt by rising prices
        instrument = "long futures" if scenario != "baseline" else "long call options"
    else:
        # Exporter is hurt by falling prices
        instrument = "short futures" if scenario != "baseline" else "long put options"

    # --- Hedge notional ---
    hedge_notional = hedge_ratio * exposure
    explanation.append(
        f"Role: {role} → use '{instrument}'. Exposure {exposure:.2f} → hedge {hedge_notional:.2f} units (hedge ratio {hedge_ratio:.0%})."
    )

    # --- Indicative maturity: last available Thursday <= last forecast date ---
    # If we have forecast dates, use the last one; otherwise fall back to "today + horizon"
    if "date" in forecast_df.columns:
        fc_dates = pd.to_datetime(forecast_df["date"], errors="coerce").dropna()
        if not fc_dates.empty:
            last_fc_date = fc_dates.max()
        else:
            last_fc_date = datetime.utcnow()
    else:
        last_fc_date = datetime.utcnow()

    # Python weekday: Monday=0, ..., Thursday=3
    target_weekday = 3  # Thursday
    delta_days = (last_fc_date.weekday() - target_weekday) % 7
    maturity_dt = last_fc_date - timedelta(days=delta_days)

    instrument_maturity = maturity_dt.date().isoformat()

    # --- Indicative strike (anchored on forecast price) ---
    instrument_strike = last_forecast
    if "option" in instrument:
        strike_comment = (
            f"For options, we set the strike close to the ARIMAX fair value "
            f"({instrument_strike:.2f}), so that the hedge is activated around the model-implied price."
        )
    else:
        strike_comment = (
            f"For futures, the economic 'strike' is the forward price, here approximated "
            f"by the ARIMAX forecast ({instrument_strike:.2f})."
        )

    # --- Justification paragraph (for UI) ---
    base_summary = " ".join(explanation)
    justification = (
        base_summary + " " +
        strike_comment +
        f" Climate risk at {last_risk:.1f} on a 0–100 scale justifies this level of protection "
        f"for a {profile} {role}."
    )

    # Infer commodity from forecast filename (e.g. data/gold/wheat_forecast.json)
    commodity = Path(forecast_path).name.replace("_forecast.json", "")

    result: Dict[str, float] = {
        "commodity": commodity,
        "profile": profile,
        "role": role,
        "exposure": float(exposure),
        "forecast_horizon_weeks": int(forecast_horizon),
        "scenario": scenario,
        "risk_score": float(last_risk),
        "hedge_ratio": float(hedge_ratio),
        "hedge_notional": float(hedge_notional),
        "instrument": instrument,
        "forecast_price": float(last_forecast),
        "instrument_maturity": instrument_maturity,
        "instrument_strike": float(instrument_strike),
        "summary": base_summary,
        "justification": justification,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    # Persist JSON next to forecast
    out_path = forecast_path.replace("_forecast.json", "_hedge_rec.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

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
