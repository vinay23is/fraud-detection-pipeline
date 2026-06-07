"""
Streams historical credit card transactions into Kafka at a configurable rate.

The producer replays the dataset as if the transactions are arriving live,
preserving the original time ordering. Each message is keyed on user_id so
that all transactions for a given user land on the same partition — this
guarantees that the processor sees a user's events in order, which matters
when computing velocity features.
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic


KAFKA_SERVERS = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
DATA_PATH = Path(os.environ.get("DATA_PATH", "/data/creditcard.csv"))
TPS = int(os.environ.get("TPS", 50))  # transactions per second
TOPIC = "transactions"


def ensure_topic(servers: str):
    admin = AdminClient({"bootstrap.servers": servers})
    existing = admin.list_topics(timeout=10).topics
    if TOPIC not in existing:
        admin.create_topics([NewTopic(TOPIC, num_partitions=4, replication_factor=1)])
        print(f"Created topic '{TOPIC}'")


def build_producer(servers: str) -> Producer:
    return Producer({
        "bootstrap.servers": servers,
        # batch up to 500ms to improve throughput without sacrificing latency much
        "linger.ms": 50,
        "batch.size": 65536,
        "compression.type": "lz4",
        "acks": "1",
    })


def assign_user_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Same assignment logic as train.py — deterministic so velocity features
    computed during training map to the same users at inference time."""
    rng = np.random.default_rng(seed=42)
    user_pool = [f"u{i:04d}" for i in range(800)]
    weights = rng.exponential(scale=1.0, size=len(user_pool))
    weights /= weights.sum()
    df = df.copy()
    df["user_id"] = rng.choice(user_pool, size=len(df), p=weights)
    return df


def stream(producer: Producer, df: pd.DataFrame):
    interval = 1.0 / TPS
    sent = 0

    print(f"Streaming {len(df):,} transactions at {TPS} TPS ...")

    for _, row in df.iterrows():
        record = {
            "transaction_id": str(uuid.uuid4()),
            "user_id": row["user_id"],
            "amount": round(float(row["Amount"]), 2),
            "time_offset": float(row["Time"]),
            "features": {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)},
            # include ground truth so the dashboard can show actual vs predicted
            # in a real system this would not exist at inference time
            "label": int(row["Class"]),
        }
        producer.produce(
            topic=TOPIC,
            key=row["user_id"],                     # partition by user
            value=json.dumps(record).encode(),
            on_delivery=_delivery_report,
        )
        producer.poll(0)
        sent += 1

        if sent % 1000 == 0:
            producer.flush()
            print(f"  {sent:,} sent  ({sent / len(df) * 100:.1f}%)")

        time.sleep(interval)

    producer.flush()
    print(f"Done. {sent:,} transactions streamed.")


def _delivery_report(err, msg):
    if err:
        print(f"Delivery error: {err}", file=sys.stderr)


def main():
    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df = assign_user_ids(df)
    df = df.sort_values("Time").reset_index(drop=True)

    ensure_topic(KAFKA_SERVERS)
    producer = build_producer(KAFKA_SERVERS)
    stream(producer, df)


if __name__ == "__main__":
    main()
