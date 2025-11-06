"""
hedging_advanced
----------------------

This module provides a simple hedging recommendation engine
for corporate risk management purposes.

It takes as inputs:
- a global climate risk score (0–100),
- a market scenario,
- the user’s risk appetite,
- the user’s role (importer or exporter),
- and the size of their physical exposure.

It returns a structured recommendation containing:
- how much of the exposure to hedge,
- what instrument to use,
- and a plain-English explanation of the reasoning.

Each profile defines a base hedge ratio, plus an additional “bump”
if the risk score exceeds a defined threshold.  
These parameters can be tuned in the ``profiles`` dictionary below.

"""

from __future__ import annotations
from typing import Literal, Dict, Any

# Hedge profile definitions.
# Each profile specifies:
# - a base hedge ratio,
# - an additional bump applied when the risk score exceeds the threshold,
# - the risk threshold itself.
profiles: Dict[str, Dict[str, float]] = {
    "conservative": {"base": 0.60, "bump": 0.20, "threshold": 70.0},
    "balanced": {"base": 0.45, "bump": 0.15, "threshold": 75.0},
    "opportunistic": {"base": 0.20, "bump": 0.10, "threshold": 80.0},
}


def recommend_hedge(
    risk_score: float,
    *,
    scenario: Literal["baseline", "bullish", "extreme"] = "baseline",
    profile: Literal["conservative", "balanced", "opportunistic"] = "balanced",
    role: Literal["importer", "exporter"] = "importer",
    exposure_units: float = 1.0,
) -> Dict[str, Any]:
    
    if profile not in profiles:
        raise KeyError(f"Unknown profile '{profile}'. Valid options are {list(profiles.keys())}.")

    # Extract parameters for the selected profile
    cfg = profiles[profile]
    hedge_ratio = cfg["base"]
    explanation_parts = [f"Base hedge ratio for {profile} profile: {hedge_ratio:.0%}."]

    # Apply bump if risk exceeds threshold
    if risk_score >= cfg["threshold"]:
        hedge_ratio += cfg["bump"]
        explanation_parts.append(
            f"Risk score {risk_score:.1f} exceeds threshold {cfg['threshold']} → add bump {cfg['bump']:.0%}."
        )

    # Scenario adjustments
    if scenario == "bullish":
        hedge_ratio += 0.05
        explanation_parts.append("Bullish scenario → increase hedge ratio by 5 percentage points.")
    elif scenario == "extreme":
        hedge_ratio += 0.10
        explanation_parts.append("Extreme scenario → increase hedge ratio by 10 percentage points.")

    # Clamp hedge ratio to [0, 1]
    hedge_ratio = min(max(hedge_ratio, 0.0), 1.0)

    # Select suggested instrument based on role and scenario
    if role == "importer":
        instrument = "long futures" if scenario != "baseline" else "long call options"
    else:  # exporter
        instrument = "short futures" if scenario != "baseline" else "long put options"

    notional = hedge_ratio * exposure_units
    explanation_parts.append(
        f"Role: {role} → suggest '{instrument}'. Exposure {exposure_units} units → hedge {notional:.2f} units."
    )

    return {
        "hedge_ratio": hedge_ratio,
        "instrument": instrument,
        "hedge_notional": notional,
        "explanation": " ".join(explanation_parts),
    }

# Test the hedging code independently
if __name__ == "__main__":
    print("=== Testing hedging_advanced ===")

    rec = recommend_hedge(
        risk_score=72.0,
        scenario="bullish",
        profile="balanced",
        role="importer",
        exposure_units=5000.0,
    )

    print(rec)
