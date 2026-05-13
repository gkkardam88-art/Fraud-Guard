"""
Kafka Producer — Simulates real-time transaction stream.
In production this is replaced by your payment gateway pushing events.
"""

import json, time, random, uuid, logging
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)
TOPIC  = "transactions"
BROKER = "localhost:9092"


def _make_transaction(force_fraud: bool = False) -> dict:
    is_fraud = force_fraud or random.random() < 0.03
    if is_fraud:
        return {
            "transaction_id":           str(uuid.uuid4()),
            "timestamp":                datetime.now(timezone.utc).isoformat(),
            "amount":                   round(random.uniform(600, 4000), 2),
            "hour":                     random.choice([0,1,2,3,23]),
            "distance_from_home":       round(random.uniform(200, 2000), 1),
            "repeat_retailer":          0,
            "used_chip":                0,
            "used_pin":                 0,
            "online_order":             1,
            "ratio_to_median_purchase": round(random.uniform(8, 30), 2),
            "sender_id":   f"acct_{random.randint(1000,9999)}",
            "receiver_id": f"merchant_{random.randint(100,200)}",
        }
    return {
        "transaction_id":           str(uuid.uuid4()),
        "timestamp":                datetime.now(timezone.utc).isoformat(),
        "amount":                   round(random.uniform(5, 300), 2),
        "hour":                     random.randint(9, 21),
        "distance_from_home":       round(random.uniform(0, 50), 1),
        "repeat_retailer":          1,
        "used_chip":                1,
        "used_pin":                 1,
        "online_order":             random.choice([0, 1]),
        "ratio_to_median_purchase": round(random.uniform(0.5, 2.0), 2),
        "sender_id":   f"acct_{random.randint(1000,9999)}",
        "receiver_id": f"merchant_{random.randint(1,100)}",
    }


def start_producer(rate_per_second: float = 2.0, total: Optional[int] = None):
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=[BROKER],
            value_serializer=lambda v: json.dumps(v).encode(),
            acks="all", retries=3,
        )
        kafka_ok = True
        log.info(f"Kafka producer connected → {BROKER}")
    except Exception as e:
        log.warning(f"Kafka unavailable ({e}) — logging only")
        producer, kafka_ok = None, False

    count, interval = 0, 1.0 / rate_per_second
    while total is None or count < total:
        txn = _make_transaction()
        if kafka_ok:
            producer.send(TOPIC, value=txn)
        else:
            log.info(f"[SIM] {txn['transaction_id'][:8]}  ${txn['amount']:.2f}  fraud={txn['used_chip']==0}")
        count += 1
        time.sleep(interval)

    if kafka_ok:
        producer.flush(); producer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    start_producer(rate_per_second=1.0, total=30)
