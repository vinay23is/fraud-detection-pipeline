"""
Redis-backed feature store for user velocity features.

Velocity features answer questions like "how many transactions has this user
made in the last hour?" — you can't answer that from a single transaction
record. The feature store keeps a sorted set per user where the score is the
Unix timestamp, so range queries over time windows are O(log N).

Keys:
    txn:{user_id}         ZSET  — timestamps of recent transactions (TTL 7d)
    amt:{user_id}         ZSET  — (timestamp, amount) pairs for avg computation

We use 7 days as the outer TTL. Anything older than that is trimmed on write,
keeping memory bounded without a separate cleanup job.
"""

import time
from dataclasses import dataclass

import redis


WINDOW_1H  = 3600
WINDOW_24H = 86400
WINDOW_7D  = 7 * 86400


@dataclass
class VelocityFeatures:
    tx_count_1h:   int
    tx_count_24h:  int
    avg_amount_7d: float


class FeatureStore:
    def __init__(self, redis_url: str):
        self._r = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )

    def get_and_update(self, user_id: str, amount: float) -> VelocityFeatures:
        """
        Read velocity features for the user, then record this transaction.
        Reading before writing ensures the current transaction is not counted
        in its own velocity features — consistent with how training computes them.
        """
        now = time.time()
        txn_key = f"txn:{user_id}"
        amt_key  = f"amt:{user_id}"

        pipe = self._r.pipeline()

        # count transactions in the last 1h and 24h
        pipe.zcount(txn_key, now - WINDOW_1H,  "+inf")
        pipe.zcount(txn_key, now - WINDOW_24H, "+inf")
        # fetch amounts in the last 7d — stored as "timestamp:amount" members
        pipe.zrangebyscore(amt_key, now - WINDOW_7D, "+inf")

        counts_1h, counts_24h, amt_members = pipe.execute()

        if amt_members:
            amounts = [float(m.split(":")[1]) for m in amt_members]
            avg_amount_7d = sum(amounts) / len(amounts)
        else:
            avg_amount_7d = amount  # no history — use current as baseline

        # record this transaction (write after read)
        member_id = f"{now:.6f}"
        pipe = self._r.pipeline()
        pipe.zadd(txn_key, {member_id: now})
        pipe.zadd(amt_key, {f"{member_id}:{amount}": now})
        # trim entries older than 7 days and reset TTL
        pipe.zremrangebyscore(txn_key, "-inf", now - WINDOW_7D)
        pipe.zremrangebyscore(amt_key,  "-inf", now - WINDOW_7D)
        pipe.expire(txn_key, WINDOW_7D)
        pipe.expire(amt_key,  WINDOW_7D)
        pipe.execute()

        return VelocityFeatures(
            tx_count_1h=int(counts_1h),
            tx_count_24h=int(counts_24h),
            avg_amount_7d=round(avg_amount_7d, 2),
        )
