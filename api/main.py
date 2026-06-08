"""
FastAPI service — two responsibilities:

1. Synchronous /predict endpoint for callers who need an immediate decision
   (e.g. a payment gateway that can't wait for the async Kafka pipeline).
   Uses the same model artifact and feature store as the processor so results
   are consistent regardless of which path a transaction takes.

2. Read-only /metrics and /predictions endpoints that expose what the async
   pipeline has written to PostgreSQL — used by the dashboard and for
   operational visibility.
"""

import os
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import redis
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from feature_store import FeatureStore
from schemas import AggregateMetrics, PredictionResponse, RecentPrediction, TransactionRequest

# --- shared state loaded once at startup ---

DATABASE_URL = os.environ["DATABASE_URL"]
REDIS_URL    = os.environ["REDIS_URL"]

# Resolve model path relative to this file so it works regardless of CWD.
# Falls back to MODEL_PATH env var if set explicitly (e.g. Docker).
_default_model = Path(__file__).parent.parent / "model" / "artifacts" / "model.pkl"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", _default_model))

_artifact: dict = {}
_store: FeatureStore | None = None
_db_conn = None


def _connect_db():
    """Separate function so we can retry without restarting the app."""
    return psycopg2.connect(DATABASE_URL, connect_timeout=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _artifact, _store, _db_conn

    with open(MODEL_PATH, "rb") as f:
        _artifact = pickle.load(f)

    _store   = FeatureStore(REDIS_URL)
    _db_conn = _connect_db()
    psycopg2.extras.register_uuid()

    yield

    if _db_conn:
        _db_conn.close()


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-Ms"] = f"{ms:.1f}"
    return response


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_feature_vector(req: TransactionRequest, velocity) -> np.ndarray:
    feature_cols = _artifact["feature_cols"]
    row = {**req.features, "Amount": req.amount}
    row["hour_of_day"]   = (req.time_offset / 3600) % 24
    row["tx_count_1h"]   = velocity.tx_count_1h
    row["tx_count_24h"]  = velocity.tx_count_24h
    row["avg_amount_7d"] = velocity.avg_amount_7d
    return np.array([[row[col] for col in feature_cols]], dtype=np.float32)


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_version": _artifact.get("version")}


@app.get("/model")
def model_info():
    """Artifact metadata — version hash, evaluation metric, decision threshold."""
    if not _artifact:
        raise HTTPException(503, "Model not loaded")
    return {
        "version":       _artifact["version"],
        "pr_auc":        _artifact["pr_auc"],
        "threshold":     _artifact["threshold"],
        "feature_count": len(_artifact["feature_cols"]),
        "features":      _artifact["feature_cols"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TransactionRequest):
    """
    Score a single transaction synchronously.

    The velocity features are read from Redis and updated atomically —
    the same operation the async processor performs, keeping both paths consistent.
    """
    if not _artifact:
        raise HTTPException(503, "Model not loaded")

    velocity   = _store.get_and_update(req.user_id, req.amount)
    X          = _build_feature_vector(req, velocity)
    fraud_prob = float(_artifact["pipeline"].predict_proba(X)[0, 1])
    is_fraud   = fraud_prob >= _artifact["threshold"]
    txn_id     = str(uuid.uuid4())

    # persist via the API path too so the dashboard reflects synchronous calls
    with _db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions
                (transaction_id, user_id, amount, fraud_prob, is_fraud,
                 model_version, tx_count_1h, tx_count_24h, avg_amount_7d)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                txn_id, req.user_id, req.amount, round(fraud_prob, 6), is_fraud,
                _artifact["version"],
                velocity.tx_count_1h, velocity.tx_count_24h, velocity.avg_amount_7d,
            ),
        )
    _db_conn.commit()

    return PredictionResponse(
        transaction_id=txn_id,
        user_id=req.user_id,
        fraud_probability=round(fraud_prob, 4),
        is_fraud=is_fraud,
        model_version=_artifact["version"],
        velocity={
            "tx_count_1h":   velocity.tx_count_1h,
            "tx_count_24h":  velocity.tx_count_24h,
            "avg_amount_7d": velocity.avg_amount_7d,
        },
    )


@app.get("/metrics", response_model=AggregateMetrics)
def metrics(hours: int = Query(default=1, ge=1, le=168)):
    """Aggregate stats over the last N hours."""
    with _db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*)                        AS total,
                SUM(is_fraud::int)              AS flagged,
                AVG(fraud_prob)                 AS avg_prob
            FROM predictions
            WHERE processed_at >= NOW() - INTERVAL '%s hours'
            """,
            (hours,),
        )
        row = cur.fetchone()

    total, flagged, avg_prob = row
    total   = total   or 0
    flagged = flagged or 0

    return AggregateMetrics(
        total_transactions=total,
        total_flagged=flagged,
        fraud_rate_pct=round(flagged / total * 100, 3) if total else 0.0,
        avg_fraud_prob=round(float(avg_prob or 0), 4),
        window_hours=hours,
    )


@app.post("/demo/seed", summary="Seed dashboard with synthetic predictions (demo only)")
def demo_seed(n: int = Query(default=500, le=2000)):
    """
    Generates n synthetic transactions and scores them through the real pipeline.

    Roughly 8% are generated with fraud-like PCA feature patterns so the
    dashboard shows non-zero fraud rates. The patterns are based on the
    feature distributions observed in the training data: fraud transactions
    cluster around extreme negative values in V14 and V4.
    """
    if not _artifact:
        raise HTTPException(503, "Model not loaded")

    rng = np.random.default_rng()
    inserted = 0

    for i in range(n):
        is_fraud_pattern = rng.random() < 0.08

        if is_fraud_pattern:
            # Approximate the fraud cluster in PCA space.
            # V14 and V4 carry the most fraud signal in this dataset —
            # fraud transactions tend to have V14 << 0 and V4 < 0.
            features = {f"V{j}": float(rng.normal()) for j in range(1, 29)}
            features["V14"] = float(rng.normal(-7.0, 1.5))
            features["V4"]  = float(rng.normal(-4.0, 1.0))
            features["V11"] = float(rng.normal(-3.0, 1.0))
            amount = round(float(rng.uniform(1.0, 500.0)), 2)
        else:
            features = {f"V{j}": float(rng.normal()) for j in range(1, 29)}
            amount = round(float(min(np.exp(rng.normal(3.8, 1.4)), 4999.99)), 2)

        user_id = f"u{rng.integers(0, 800):04d}"
        req = TransactionRequest(
            user_id=user_id,
            amount=amount,
            features=features,
            time_offset=float(rng.uniform(0, 172800)),
        )
        predict(req)
        inserted += 1

    return {"seeded": inserted}


@app.get("/predictions", response_model=list[RecentPrediction])
def recent_predictions(limit: int = Query(default=50, le=200), fraud_only: bool = False):
    """Most recent predictions, optionally filtered to flagged ones only."""
    where = "WHERE is_fraud = TRUE" if fraud_only else ""
    with _db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT transaction_id, user_id, amount, fraud_prob,
                   is_fraud, tx_count_1h, processed_at
            FROM predictions
            {where}
            ORDER BY processed_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()

    return [RecentPrediction(**r) for r in rows]
