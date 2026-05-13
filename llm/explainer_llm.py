"""
LLM Integration — Human-Readable Fraud Explanations
-----------------------------------------------------
Priority chain:
  1. OpenAI GPT-4   (if OPENAI_API_KEY is set)
  2. Ollama llama3  (if local Ollama server is running)
  3. Rule-based     (always works, no dependencies)
"""

import os
import json
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


def _build_prompt(transaction: dict, prediction: dict, shap_summary: list, risk_score: dict) -> str:
    factors = "\n".join(
        f"  - {f['feature'].replace('_',' ')}: "
        f"{'raises' if f['direction'] == 'increases_risk' else 'lowers'} risk "
        f"(score contribution: {f['shap_value']:+.3f})"
        for f in shap_summary[:5]
    )
    verdict = "FRAUDULENT" if prediction["is_fraud"] else "LEGITIMATE"

    return f"""You are a senior financial fraud analyst. A machine learning model flagged the
following transaction as {verdict}. Write a clear, professional 3–4 sentence explanation
for a fraud analyst (non-technical audience). Be specific about which signals triggered
the alert. Do not use the word "SHAP".

Transaction:
  Amount: ${transaction['amount']:.2f}
  Hour: {transaction['hour']}:00
  Distance from home: {transaction['distance_from_home']:.1f} km
  Used chip: {'Yes' if transaction['used_chip'] else 'No'}
  Used PIN: {'Yes' if transaction['used_pin'] else 'No'}
  Online order: {'Yes' if transaction['online_order'] else 'No'}
  Repeat retailer: {'Yes' if transaction['repeat_retailer'] else 'No'}
  Ratio to median spend: {transaction['ratio_to_median_purchase']:.2f}x

Model Output:
  Verdict: {verdict}
  Fraud Probability: {prediction['fraud_probability']:.1%}
  Risk Score: {risk_score['risk_score']}/100 ({risk_score['risk_level'].upper()})

Key contributing factors:
{factors}

Write your explanation now:"""


def generate_explanation(
    transaction: dict,
    prediction: dict,
    shap_summary: list,
    risk_score: dict,
    provider: str = "fallback",
) -> Optional[str]:
    prompt = _build_prompt(transaction, prediction, shap_summary, risk_score)

    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        result = _call_openai(prompt)
        if result:
            return result

    if provider == "ollama":
        result = _call_ollama(prompt)
        if result:
            return result

    return fallback_explanation(transaction, prediction, shap_summary, risk_score)


def _call_openai(prompt: str) -> Optional[str]:
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=350,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"OpenAI call failed: {e}")
        return None


def _call_ollama(prompt: str) -> Optional[str]:
    try:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=30,
        )
        return resp.json().get("response", "").strip() or None
    except Exception as e:
        log.warning(f"Ollama call failed: {e}")
        return None


def fallback_explanation(
    transaction: dict,
    prediction: dict,
    shap_summary: list,
    risk_score: dict,
) -> str:
    """
    Deterministic rule-based explanation — works with zero dependencies.
    Produces readable analyst-grade output by reading SHAP directions directly.
    """
    verdict  = "flagged as potentially fraudulent" if prediction["is_fraud"] else "classified as legitimate"
    prob_pct = f"{prediction['fraud_probability']:.1%}"
    score    = risk_score["risk_score"]
    level    = risk_score["risk_level"].upper()

    risk_factors  = [f for f in shap_summary if f["direction"] == "increases_risk"][:3]
    safe_factors  = [f for f in shap_summary if f["direction"] == "decreases_risk"][:2]

    risk_parts = [f['feature'].replace('_', ' ') for f in risk_factors]
    safe_parts = [f['feature'].replace('_', ' ') for f in safe_factors]

    lines = [
        f"This transaction was {verdict} with a fraud probability of {prob_pct} "
        f"and a risk score of {score}/100 ({level} RISK)."
    ]

    if risk_parts:
        lines.append(
            f"The primary factors raising suspicion were: {', '.join(risk_parts)}."
        )
    if safe_parts:
        lines.append(
            f"Factors that reduced risk included: {', '.join(safe_parts)}."
        )

    if prediction["is_fraud"]:
        lines.append("Immediate manual review and temporary card hold are recommended pending cardholder verification.")
    else:
        lines.append("No immediate action required; standard monitoring is sufficient.")

    return " ".join(lines)
