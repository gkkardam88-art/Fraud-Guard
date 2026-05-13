"""
Kafka Consumer — Real-time fraud detection pipeline.
Consumes from 'transactions' topic, scores each transaction,
publishes fraud alerts to 'fraud_alerts' topic.
"""

import json, pickle, logging, numpy as np
from datetime import datetime, timezone

log = logging.getLogger(__name__)
BROKER      = "localhost:9092"
INPUT_TOPIC = "transactions"
ALERT_TOPIC = "fraud_alerts"
MODEL_PATH  = "models/fraud_model.pkl"


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _score(artifacts: dict, txn: dict) -> dict:
    cols   = artifacts["feature_cols"]
    scaler = artifacts["scaler"]
    model  = artifacts["model"]
    thr    = artifacts.get("threshold", 0.5)
    try:
        X    = np.array([[txn[c] for c in cols]], dtype=np.float32)
        prob = float(model.predict_proba(scaler.transform(X))[0][1])
        return {"is_fraud": prob >= thr, "fraud_probability": round(prob, 4),
                "transaction_id": txn.get("transaction_id"),
                "processed_at": datetime.now(timezone.utc).isoformat()}
    except KeyError as e:
        log.error(f"Missing feature {e}"); return None


def start_consumer():
    artifacts = _load_model()
    log.info("Model loaded. Starting consumer...")
    try:
        from kafka import KafkaConsumer, KafkaProducer
        consumer = KafkaConsumer(INPUT_TOPIC, bootstrap_servers=[BROKER],
                                 value_deserializer=lambda m: json.loads(m.decode()),
                                 group_id="fraud-group", auto_offset_reset="latest")
        alerter  = KafkaProducer(bootstrap_servers=[BROKER],
                                 value_serializer=lambda v: json.dumps(v).encode())
        for msg in consumer:
            result = _score(artifacts, msg.value)
            if not result: continue
            tag = "🚨 FRAUD" if result["is_fraud"] else "✅ LEGIT"
            log.info(f"{tag}  {result['transaction_id'][:8]}  prob={result['fraud_probability']:.3f}")
            if result["is_fraud"]:
                alerter.send(ALERT_TOPIC, value={**msg.value, **result})
    except Exception as e:
        log.warning(f"Kafka not available ({e}) — running simulation")
        _simulate(artifacts)


def _simulate(artifacts, n=15):
    from streaming.producer import _make_transaction
    for _ in range(n):
        txn = _make_transaction()
        r   = _score(artifacts, txn)
        if r:
            log.info(f"{'FRAUD' if r['is_fraud'] else 'LEGIT'}  ${txn['amount']:.2f}  {r['fraud_probability']:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    start_consumer()
