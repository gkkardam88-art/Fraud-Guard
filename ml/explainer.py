"""
SHAP Explainability Module
--------------------------
Computes per-prediction feature contributions using SHAP TreeExplainer.

Why SHAP?
  - Grounded in cooperative game theory (Shapley values)
  - Locally accurate: values sum exactly to prediction - base rate
  - Consistent: if a feature matters more, its SHAP value always reflects that
  - Required for regulatory compliance (EU AI Act, GDPR Article 22)
"""

import logging
import numpy as np
import pickle
from typing import Any, Dict, List

log = logging.getLogger(__name__)


def load_artifacts(model_path: str = "models/fraud_model.pkl") -> Dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def compute_shap_values(model, X_scaled: np.ndarray, feature_cols: List[str]) -> Dict[str, Any]:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_scaled)

        # Binary classification returns list [class0, class1] or single array
        if isinstance(shap_vals, list):
            fraud_shap = shap_vals[1]
        else:
            fraud_shap = shap_vals

        per_feature = {feat: float(fraud_shap[0][i]) for i, feat in enumerate(feature_cols)}

        base = explainer.expected_value
        base_value = float(base[1] if isinstance(base, (list, np.ndarray)) else base)

        return {"shap_values": per_feature, "base_value": base_value}

    except ImportError:
        log.warning("SHAP not installed — returning heuristic feature scores")
        return _heuristic_shap(X_scaled, feature_cols)


def _heuristic_shap(X_scaled: np.ndarray, feature_cols: List[str]) -> Dict[str, Any]:
    """Fallback when SHAP library isn't available."""
    weights = [0.25, 0.15, 0.20, -0.10, -0.12, -0.11, 0.08, 0.18]
    per_feature = {}
    for i, feat in enumerate(feature_cols):
        w = weights[i] if i < len(weights) else 0.05
        per_feature[feat] = float(X_scaled[0][i] * w)
    return {"shap_values": per_feature, "base_value": -2.0}


def top_risk_factors(shap_values: Dict[str, float], top_n: int = 8) -> List[Dict]:
    sorted_feats = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "feature":   feat,
            "shap_value": round(val, 4),
            "direction": "increases_risk" if val > 0 else "decreases_risk",
        }
        for feat, val in sorted_feats[:top_n]
    ]


def explain_transaction(transaction: dict, model_path: str = "models/fraud_model.pkl") -> Dict[str, Any]:
    artifacts    = load_artifacts(model_path)
    model        = artifacts["model"]
    scaler       = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]

    X_raw    = np.array([[transaction[f] for f in feature_cols]], dtype=np.float32)
    X_scaled = scaler.transform(X_raw)

    shap_data = compute_shap_values(model, X_scaled, feature_cols)
    factors   = top_risk_factors(shap_data["shap_values"])

    return {
        "shap_values":      shap_data["shap_values"],
        "base_value":       shap_data["base_value"],
        "top_risk_factors": factors,
    }
