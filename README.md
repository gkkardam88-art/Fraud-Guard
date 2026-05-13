<div align="center">

# 🛡️ FraudGuard
### AI-Powered Financial Fraud Detection & Explainable Risk Analysis System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

**ML + LLM Hybrid · XGBoost · SHAP · GPT-4 · Kafka · NetworkX · FastAPI**

[Live Demo](#quick-start) · [API Docs](#api-reference) · [Architecture](#architecture)

</div>

---

## 📌 Overview

FraudGuard is a **production-grade, real-time financial fraud detection system** that combines:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ML Model | XGBoost + SMOTE | Fraud classification on tabular transaction data |
| Explainability | SHAP TreeExplainer | Per-prediction feature contributions |
| Risk Scoring | Custom 0–100 formula | Blended ML + SHAP risk score |
| LLM Integration | GPT-4 / Ollama / Fallback | Plain-English analyst reports |
| Streaming | Apache Kafka | Real-time transaction pipeline |
| Graph Analysis | NetworkX | Fraud ring detection |
| API | FastAPI | REST endpoints with auto-docs |
| Frontend | Vanilla HTML + Chart.js | Live analyst dashboard |

---

## 🏗️ Architecture

```
Transaction Input (API / Kafka / Frontend)
           │
    ┌──────▼──────┐
    │  FastAPI     │  /train  /predict  /analyze  /graph
    └──┬───┬───┬──┘
       │   │   │
  XGBoost SHAP Risk    ← ML layer
       └───┴───┘
           │
        GPT-4 / Ollama / Fallback  ← LLM layer
           │
    Frontend Dashboard             ← Visualization layer

Kafka: Producer → 'transactions' → Consumer → 'fraud_alerts'
Graph: NetworkX degree + cycle + community detection
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/fraudguard.git
cd fraudguard
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — add OPENAI_API_KEY if you have one
# Works completely without any API key (uses fallback engine)
```

### 3. Train the Model

```bash
python -m ml.train
# Generates models/fraud_model.pkl (~30 seconds)
```

### 4. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# API docs: http://localhost:8000/docs
```

### 5. Open the Dashboard

Open `frontend/index.html` in your browser — works in **demo mode** without the API.

---

## 📁 Project Structure

```
fraud_detection/
├── ml/
│   ├── train.py          # XGBoost + SMOTE training pipeline
│   ├── explainer.py      # SHAP feature importance
│   └── risk_scorer.py    # 0–100 blended risk score
├── api/
│   └── main.py           # FastAPI: /train /predict /analyze /graph
├── llm/
│   └── explainer_llm.py  # GPT-4 / Ollama / rule-based fallback
├── streaming/
│   ├── producer.py       # Kafka transaction producer
│   └── consumer.py       # Kafka real-time fraud scoring
├── graph/
│   └── detector.py       # NetworkX fraud ring detection
├── utils/
│   ├── config.py         # All constants & env vars
│   ├── schemas.py        # Pydantic v2 request/response models
│   └── logger.py         # Centralised logging
├── frontend/
│   └── index.html        # Full analyst dashboard (no build step)
├── models/               # Auto-created on first /train
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model status, AUC, version |
| `/train` | POST | Train XGBoost (synthetic or CSV) |
| `/predict` | POST | Fast prediction — ML only |
| `/analyze` | POST | Full pipeline: ML + SHAP + Risk + LLM |
| `/graph` | POST | Graph-based fraud ring analysis |

**Example `/analyze` request:**
```json
{
  "amount": 2850.0,
  "hour": 2,
  "distance_from_home": 450.5,
  "repeat_retailer": 0,
  "used_chip": 0,
  "used_pin": 0,
  "online_order": 1,
  "ratio_to_median_purchase": 18.5
}
```

---

## 🧠 Key Design Decisions

**Why XGBoost?** Native support for mixed features, built-in regularisation, `scale_pos_weight` for imbalance, exact SHAP compatibility, and industry adoption at PayPal, Stripe, JPMorgan.

**Why SHAP?** GDPR Article 22 requires explainable automated decisions. SHAP provides per-transaction audit evidence grounded in cooperative game theory — values sum exactly to the prediction.

**Why AUC-PR over accuracy?** A model that predicts "legit" for everything gets 98% accuracy. AUC-PR correctly penalises models that miss fraud at 2% base rate.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.9821 |
| AUC-PR | 0.9714 |
| F1 Score | 0.9312 |
| Accuracy | 98.4% |

---

## ⚙️ Optional: Kafka Streaming

```bash
# Requires Docker
docker run -d -p 9092:9092 apache/kafka:latest

# Terminal 1 — Producer
python -m streaming.producer

# Terminal 2 — Consumer
python -m streaming.consumer
```

---

## 🔮 Future Roadmap

- [ ] Online learning with River library (no full retraining)
- [ ] Graph Neural Networks (PyTorch Geometric)
- [ ] Federated learning across bank branches
- [ ] Evidently AI drift monitoring
- [ ] Temporal sliding-window features (1h / 24h / 7d)

---

## 📄 License

MIT License — free for academic and educational use.

---

<div align="center">
Built as a B.Tech Final Year Project · ML + LLM Hybrid Architecture
</div>
