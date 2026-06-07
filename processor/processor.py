"""
Kafka consumer: reads transactions, builds features, runs inference, persists results.

The processing loop is deliberately single-threaded per instance. Kafka's
partition-by-user_id guarantee (set in the producer) means all events for a
given user arrive at one partition in order. If you need more throughput, scale
by adding processor replicas — Kafka's consumer group protocol will rebalance
partitions automatically without any code change here.
"""

import json
import logging
import os
import pickle
import signal
import sys
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from confluent_kafka import Consumer, KafkaError

from feature_store import FeatureStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

KAFKA_SERVERS = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
REDIS_URL     = os.environ["REDIS_URL"]
DATABASE_URL  = os.environ["DATABASE_URL"]
MODEL_PATH    = Path(os.environ.get("MODEL_PATH", "/app/artifacts/model.pkl"))
TOPIC         = "transactions"


def load_model(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_consumer(servers: str) -> Consumer:
    return Consumer({
        "bootstrap.servers": servers,
        "group.id": "fraud-processor",
        "auto.offset.reset": "earliest",
        # manual commit after successful DB write — prevents data loss on crash
        "enable.auto.commit": False,
    })


def build_feature_vector(record: dict, velocity, feature_cols: list[str]) -> np.ndarray:
    row = {**record["features"], "Amount": record["amount"]}
    # hour_of_day from the dataset's Time field (seconds since first transaction)
    row["hour_of_day"] = (record["time_offset"] / 3600) % 24
    row["tx_count_1h"]   = velocity.tx_count_1h
    row["tx_count_24h"]  = velocity.tx_count_24h
    row["avg_amount_7d"] = velocity.avg_amount_7d

    return np.array([[row[col] for col in feature_cols]], dtype=np.float32)


def persist(conn, record: dict, fraud_prob: float, velocity, model_version: str, threshold: float):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions
                (transaction_id, user_id, amount, fraud_prob, is_fraud,
                 model_version, tx_count_1h, tx_count_24h, avg_amount_7d)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record["transaction_id"],
                record["user_id"],
                record["amount"],
                round(fraud_prob, 6),
                fraud_prob >= threshold,
                model_version,
                velocity.tx_count_1h,
                velocity.tx_count_24h,
                velocity.avg_amount_7d,
            ),
        )
    conn.commit()


def run():
    artifact = load_model(MODEL_PATH)
    pipeline      = artifact["pipeline"]
    feature_cols  = artifact["feature_cols"]
    threshold     = artifact["threshold"]
    model_version = artifact["version"]

    log.info(f"Model version={model_version}  threshold={threshold}  features={len(feature_cols)}")

    store    = FeatureStore(REDIS_URL)
    consumer = build_consumer(KAFKA_SERVERS)
    consumer.subscribe([TOPIC])

    conn = psycopg2.connect(DATABASE_URL)

    # graceful shutdown on SIGTERM (Docker stop sends this)
    running = True
    def _shutdown(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, _shutdown)

    processed = flagged = 0

    try:
        while running:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.error(f"Kafka error: {msg.error()}")
                continue

            record = json.loads(msg.value().decode())
            velocity = store.get_and_update(record["user_id"], record["amount"])

            X = build_feature_vector(record, velocity, feature_cols)
            fraud_prob = float(pipeline.predict_proba(X)[0, 1])

            persist(conn, record, fraud_prob, velocity, model_version, threshold)
            consumer.commit(asynchronous=False)

            processed += 1
            if fraud_prob >= threshold:
                flagged += 1
                log.warning(
                    f"FRAUD  txn={record['transaction_id'][:8]}  "
                    f"user={record['user_id']}  amount=${record['amount']:.2f}  "
                    f"prob={fraud_prob:.3f}  1h_tx={velocity.tx_count_1h}"
                )
            elif processed % 500 == 0:
                log.info(f"processed={processed:,}  flagged={flagged}  ({flagged/processed*100:.2f}%)")

    finally:
        consumer.close()
        conn.close()


if __name__ == "__main__":
    run()
