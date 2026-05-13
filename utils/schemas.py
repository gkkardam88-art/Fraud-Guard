"""
Pydantic v2 schemas.
Single source of truth for every request and response shape.
Imported by api/main.py, tests, and any internal code that needs typed dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Requests ─────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    Everything the ML model needs, plus optional metadata for audit trails
    and graph analysis. The feature_dict() helper extracts only ML inputs
    in the exact column order the model was trained with.
    """

    # ML features
    amount: float = Field(..., gt=0, example=250.0)
    hour: int = Field(..., ge=0, le=23, example=14)
    distance_from_home: float = Field(..., ge=0, example=12.5)
    repeat_retailer: int = Field(..., ge=0, le=1, example=1)
    used_chip: int = Field(..., ge=0, le=1, example=1)
    used_pin: int = Field(..., ge=0, le=1, example=1)
    online_order: int = Field(..., ge=0, le=1, example=0)
    ratio_to_median_purchase: float = Field(..., gt=0, example=1.2)

    # Metadata — not used by ML, stored for audit / graph
    transaction_id: Optional[str] = Field(None, example="txn-abc-123")
    sender_id: Optional[str] = Field(None, example="acct-4721")
    receiver_id: Optional[str] = Field(None, example="merchant-88")
    timestamp: Optional[str] = Field(None, example="2024-06-15T14:32:00Z")

    @field_validator("amount")
    @classmethod
    def amount_sanity(cls, v: float) -> float:
        if v > 1_000_000:
            raise ValueError("amount exceeds $1,000,000 — possible data error")
        return round(v, 2)

    @field_validator("ratio_to_median_purchase")
    @classmethod
    def ratio_sanity(cls, v: float) -> float:
        # Ratios above 1000x are almost certainly data errors, not fraud signals
        if v > 1000:
            raise ValueError("ratio_to_median_purchase > 1000 is implausible")
        return v

    @model_validator(mode="after")
    def chip_without_pin_is_suspicious_but_valid(self) -> TransactionRequest:
        # Validation hook: useful place to add cross-field business rules later
        return self

    def feature_dict(self) -> Dict[str, float]:
        """Return only ML features in trained column order."""
        return {
            "amount":                   self.amount,
            "hour":                     self.hour,
            "distance_from_home":       self.distance_from_home,
            "repeat_retailer":          self.repeat_retailer,
            "used_chip":                self.used_chip,
            "used_pin":                 self.used_pin,
            "online_order":             self.online_order,
            "ratio_to_median_purchase": self.ratio_to_median_purchase,
        }


class GraphRequest(BaseModel):
    transactions: List[Dict[str, Any]] = Field(
        ...,
        min_length=2,
        description="Raw transaction dicts; each needs sender_id, receiver_id, amount.",
    )


class TrainRequest(BaseModel):
    csv_path: Optional[str] = Field(
        None,
        description="Absolute path to a CSV on the server. Omit to use synthetic data.",
    )
    output_dir: str = Field("models", description="Directory to save model artifacts.")


# ─── Response building blocks ─────────────────────────────────────────────────

class PredictionResult(BaseModel):
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    verdict: str              # "FRAUD" | "LEGITIMATE"


class RiskBreakdown(BaseModel):
    probability_component: float
    shap_penalty_component: float


class RiskScoreResult(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: str            # critical | high | moderate | low
    description: str
    breakdown: RiskBreakdown


class ShapFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str             # increases_risk | decreases_risk


class ShapResult(BaseModel):
    shap_values: Dict[str, float]
    base_value: float
    top_risk_factors: List[ShapFactor]


# ─── Top-level responses ──────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_auc: Optional[float]
    version: str = "1.0.0"


class TrainResponse(BaseModel):
    message: str
    auc_roc: float
    auc_pr: float
    model_path: str


class PredictResponse(BaseModel):
    transaction_id: Optional[str]
    prediction: PredictionResult


class AnalyzeResponse(BaseModel):
    transaction_id: Optional[str]
    prediction: PredictionResult
    risk_score: RiskScoreResult
    shap_explanation: ShapResult
    llm_explanation: str


class GraphResponse(BaseModel):
    graph_stats: Dict[str, Any]
    high_degree_nodes: List[Dict[str, Any]]
    detected_cycles: List[Dict[str, Any]]
    suspicious_communities: List[Dict[str, Any]]
    flagged_accounts: List[str]
    summary: str
