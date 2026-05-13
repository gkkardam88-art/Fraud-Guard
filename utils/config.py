"""
Central configuration.
All constants and environment variables live here — never scatter
os.getenv() calls across the codebase.

Usage:
    from utils.config import cfg
    cfg.model.path
    cfg.api.port
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv is optional; env vars can be set directly


# ─── Sub-configs ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    path: str = "models/fraud_model.pkl"
    fraud_threshold: float = 0.50        # probability cutoff for fraud label
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    scale_pos_weight: int = 49           # ≈ (1 - fraud_rate) / fraud_rate
    early_stopping_rounds: int = 20
    random_state: int = 42


@dataclass(frozen=True)
class DataConfig:
    synthetic_rows: int = 50_000
    fraud_rate: float = 0.02
    test_size: float = 0.20
    random_state: int = 42
    feature_cols: Tuple[str, ...] = field(default_factory=lambda: (
        "amount",
        "hour",
        "distance_from_home",
        "repeat_retailer",
        "used_chip",
        "used_pin",
        "online_order",
        "ratio_to_median_purchase",
    ))


@dataclass(frozen=True)
class APIConfig:
    host: str          = os.getenv("API_HOST", "0.0.0.0")
    port: int          = int(os.getenv("API_PORT", "8000"))
    workers: int       = int(os.getenv("API_WORKERS", "1"))
    reload: bool       = os.getenv("ENV", "development") == "development"
    request_timeout: int = 30   # seconds; enforced by middleware
    cors_origins: Tuple[str, ...] = field(default_factory=lambda: (
        os.getenv("CORS_ORIGINS", "*"),
    ))


@dataclass(frozen=True)
class LLMConfig:
    # "openai" | "ollama" | "fallback"
    provider: str       = os.getenv("LLM_PROVIDER", "fallback")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str   = "gpt-4"
    ollama_url: str     = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str   = os.getenv("OLLAMA_MODEL", "llama3")
    max_tokens: int     = 350
    temperature: float  = 0.3
    timeout_sec: int    = 25


@dataclass(frozen=True)
class KafkaConfig:
    broker: str          = os.getenv("KAFKA_BROKER", "localhost:9092")
    input_topic: str     = "transactions"
    alert_topic: str     = "fraud_alerts"
    consumer_group: str  = "fraud-detection-group"
    producer_rate: float = float(os.getenv("PRODUCER_RATE", "2.0"))


@dataclass(frozen=True)
class RiskConfig:
    """
    Final risk score = prob_weight * P(fraud) + shap_weight * shap_signal
    Both weights sum to 100.
    """
    prob_weight: float      = 70.0
    shap_weight: float      = 30.0
    max_shap_norm: float    = 3.0   # normalisation ceiling for pos-SHAP total
    critical_floor: float   = 76.0
    high_floor: float       = 51.0
    moderate_floor: float   = 26.0


# ─── Root config object ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    api:   APIConfig   = field(default_factory=APIConfig)
    llm:   LLMConfig   = field(default_factory=LLMConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    risk:  RiskConfig  = field(default_factory=RiskConfig)


# Singleton — every module does:  from utils.config import cfg
cfg = Config()
