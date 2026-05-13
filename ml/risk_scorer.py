"""
Risk Scoring System
--------------------
Converts raw ML probability + SHAP signal into a calibrated 0–100 risk score.

Score Bands:
  0  – 25  → Low       (approve, standard monitoring)
  26 – 50  → Moderate  (flag, watch closely)
  51 – 75  → High      (hold for manual review)
  76 – 100 → Critical  (auto-block, contact cardholder)

Scoring formula:
  score = (fraud_probability × 70) + (normalised_positive_shap × 30)

The two-component blend prevents false positives from a briefly uncertain
model — a transaction needs BOTH high probability AND explainable SHAP
evidence to reach the critical band.
"""

import logging
from typing import Dict, Any

log = logging.getLogger(__name__)

BANDS = [
    (76.0, 100.0, "critical",  "Auto-block — multiple strong fraud indicators detected"),
    (51.0,  75.0, "high",      "Hold for manual review — significant risk signals present"),
    (26.0,  50.0, "moderate",  "Flag and monitor — some suspicious patterns observed"),
    (0.0,   25.0, "low",       "Low risk — approve with standard transaction monitoring"),
]


def calculate_risk_score(fraud_probability: float, shap_values: Dict[str, float]) -> Dict[str, Any]:
    prob_component = round(fraud_probability * 70.0, 2)

    # Sum only features pushing toward fraud (positive SHAP)
    positive_shap_total = sum(v for v in shap_values.values() if v > 0)

    # Normalise against expected maximum (~3.0 for this feature set), cap at 1.0
    shap_normalised    = min(positive_shap_total / 3.0, 1.0)
    shap_component     = round(shap_normalised * 30.0, 2)

    raw   = prob_component + shap_component
    score = round(min(max(raw, 0.0), 100.0), 1)

    level, description = _get_band(score)

    return {
        "risk_score":  score,
        "risk_level":  level,
        "description": description,
        "breakdown": {
            "probability_component":  prob_component,
            "shap_penalty_component": shap_component,
        },
    }


def _get_band(score: float):
    for low, high, label, desc in BANDS:
        if low <= score <= high:
            return label, desc
    return "unknown", "Classification unavailable"
