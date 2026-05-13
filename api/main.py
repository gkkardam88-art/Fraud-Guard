"""
FastAPI Backend — Fraud Detection REST API
------------------------------------------
Endpoints:
  GET  /health          Health check + model status
  POST /train           Train or retrain the XGBoost model
  POST /predict         Fast prediction (ML only, no SHAP)
  POST /analyze         Full pipeline: ML + SHAP + Risk Score + LLM
  POST /graph           Graph-based fraud ring analysis
"""

import os
import sys
import pickle
import logging
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Add project root to path so imports work from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = "models/fraud_model.pkl"
_artifacts: dict = {}


# ─── Startup / shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                _artifacts.update(pickle.load(f))
            log.info(f"Model loaded on startup — AUC={_artifacts.get('auc', 'N/A')}")
        except Exception as e:
            log.warning(f"Could not load model on startup: {e}")
    else:
        log.warning("No trained model found. Call POST /train to train one.")
    yield
    _artifacts.clear()


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FraudGuard API",
    description="AI-powered financial fraud detection — ML + SHAP + LLM",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ─── Request schemas ──────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    amount: float                    = Field(..., gt=0,      example=250.0)
    hour: int                        = Field(..., ge=0, le=23, example=14)
    distance_from_home: float        = Field(..., ge=0,      example=12.5)
    repeat_retailer: int             = Field(..., ge=0, le=1, example=1)
    used_chip: int                   = Field(..., ge=0, le=1, example=1)
    used_pin: int                    = Field(..., ge=0, le=1, example=1)
    online_order: int                = Field(..., ge=0, le=1, example=0)
    ratio_to_median_purchase: float  = Field(..., gt=0,      example=1.2)
    transaction_id: Optional[str]    = None
    sender_id: Optional[str]         = None
    receiver_id: Optional[str]       = None
    timestamp: Optional[str]         = None

    def feature_dict(self) -> dict:
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


class TrainRequest(BaseModel):
    csv_path: Optional[str]  = Field(None, description="Path to CSV on server. Omit for synthetic data.")
    output_dir: str          = Field("models", description="Where to save model artifacts.")


class GraphRequest(BaseModel):
    transactions: list = Field(..., min_length=2)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _require_model():
    if not _artifacts:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. POST /train first, or run: python -m ml.train"
        )


def _predict_proba(txn_dict: dict) -> tuple[float, np.ndarray]:
    feature_cols = _artifacts["feature_cols"]
    scaler       = _artifacts["scaler"]
    model        = _artifacts["model"]
    threshold    = _artifacts.get("threshold", 0.5)

    X        = np.array([[txn_dict[f] for f in feature_cols]], dtype=np.float32)
    X_scaled = scaler.transform(X)
    prob     = float(model.predict_proba(X_scaled)[0][1])
    return prob, X_scaled


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status":       "ok",
        "model_loaded": bool(_artifacts),
        "model_auc":    _artifacts.get("auc"),
        "trained_at":   _artifacts.get("trained_at"),
        "version":      "2.0.0",
    }


@app.post("/train", tags=["Model"])
def train_model(req: TrainRequest = TrainRequest()):
    """
    Trains XGBoost on the provided CSV or synthetic data.
    Takes ~30–60 seconds. Stores model to disk and caches in memory.
    """
    try:
        from ml.train import train
        result = train(csv_path=req.csv_path, output_dir=req.output_dir)
        _artifacts.clear()
        _artifacts.update(result)
        return {
            "message":   "Model trained successfully",
            "auc_roc":   result["metrics"]["auc_roc"],
            "auc_pr":    result["metrics"]["auc_pr"],
            "f1":        result["metrics"]["f1"],
            "threshold": result["threshold"],
            "model_path": MODEL_PATH,
        }
    except Exception as e:
        log.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", tags=["Inference"])
def predict(txn: TransactionRequest):
    """Fast prediction — returns verdict and probability only. No SHAP, no LLM."""
    _require_model()
    try:
        txn_dict  = txn.feature_dict()
        threshold = _artifacts.get("threshold", 0.5)
        prob, _   = _predict_proba(txn_dict)
        is_fraud  = prob >= threshold

        return {
            "transaction_id":    txn.transaction_id,
            "is_fraud":          is_fraud,
            "fraud_probability": round(prob, 4),
            "verdict":           "FRAUD" if is_fraud else "LEGITIMATE",
            "threshold_used":    threshold,
        }
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", tags=["Inference"])
def analyze(txn: TransactionRequest, llm_provider: str = "fallback"):
    """
    Full pipeline:
      1. XGBoost prediction
      2. SHAP feature contributions
      3. Risk score (0–100) with band classification
      4. LLM natural-language explanation
    """
    _require_model()
    try:
        txn_dict  = txn.feature_dict()
        threshold = _artifacts.get("threshold", 0.5)

        # Step 1 — ML prediction
        prob, _  = _predict_proba(txn_dict)
        is_fraud = prob >= threshold

        prediction = {
            "is_fraud":          is_fraud,
            "fraud_probability": round(prob, 4),
            "verdict":           "FRAUD" if is_fraud else "LEGITIMATE",
        }

        # Step 2 — SHAP
        from ml.explainer import explain_transaction
        shap_result = explain_transaction(txn_dict, MODEL_PATH)

        # Step 3 — Risk score
        from ml.risk_scorer import calculate_risk_score
        risk = calculate_risk_score(prob, shap_result["shap_values"])

        # Step 4 — LLM explanation
        from llm.explainer_llm import generate_explanation, fallback_explanation
        explanation = generate_explanation(
            transaction  = txn_dict,
            prediction   = prediction,
            shap_summary = shap_result["top_risk_factors"],
            risk_score   = risk,
            provider     = llm_provider,
        )
        if not explanation:
            explanation = fallback_explanation(txn_dict, prediction, shap_result["top_risk_factors"], risk)

        return {
            "transaction_id":   txn.transaction_id,
            "prediction":       prediction,
            "risk_score":       risk,
            "shap_explanation": shap_result,
            "llm_explanation":  explanation,
        }

    except Exception as e:
        log.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph", tags=["Graph"])
def graph_analysis(req: GraphRequest):
    """Analyze a batch of transactions for fraud rings via NetworkX."""
    try:
        from graph.detector import analyze_graph
        return analyze_graph(req.transactions)
    except Exception as e:
        log.exception("Graph analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Dev entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
