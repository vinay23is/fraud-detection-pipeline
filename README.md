# Real-time Fraud Detection Pipeline

An end-to-end system for scoring card transactions for fraud as they arrive, including the part most tutorials skip: computing a user's real-time transaction history from a live feature store instead of a static training table.

**Live Demo:** [Streamlit Dashboard](https://fraud-detection-pipeline-8l3szzw6xzjmhhkcx8fq2x.streamlit.app) *(Render/Streamlit Cloud free tier — first load can take ~30s to wake, and Streamlit Cloud may show its own sign-in interstitial to non-browser clients before the app itself loads)*

## What problem does this solve?

Most fraud detection writeups train a model on a static, already-labeled dataset and stop there. That skips the part that actually makes fraud detection hard in production: a transaction arrives as a single row with no history attached, but questions like "is this user transacting unusually often right now?" require external state — a live count of that user's recent activity, pulled from somewhere fast enough to answer in milliseconds. This project builds the full loop instead of stopping at the model: a Kafka producer streams transactions, a consumer enriches each one with velocity features read from a Redis feature store, runs XGBoost inference, and writes the decision to PostgreSQL, with a FastAPI endpoint for synchronous scoring and a Streamlit dashboard for live visibility into what the system is doing.

## Tech Stack

- **Backend/API:** Python, FastAPI, Pydantic
- **Streaming:** Apache Kafka (partitioned by `user_id`)
- **Feature Store/Cache:** Redis (sorted sets with TTL-bounded windows)
- **ML:** XGBoost, scikit-learn, pandas/numpy
- **Database:** PostgreSQL
- **Dashboard:** Streamlit
- **Infra/Deployment:** Docker Compose (full local stack), Neon (hosted Postgres), Upstash (hosted Redis), Render (API), Streamlit Cloud (dashboard) — see the live backend API docs at [fraud-detection-pipeline-pdv9.onrender.com/docs](https://fraud-detection-pipeline-pdv9.onrender.com/docs)
- **Testing/CI:** pytest + fakeredis, GitHub Actions

## Architecture

```
creditcard.csv
     │
     ▼
┌─────────────┐     transactions      ┌─────────────────────────────────────────┐
│  Producer   │ ─────────────────────▶│             Processor                   │
│  (Python)   │   (keyed by user_id,  │                                         │
└─────────────┘    4 partitions)      │  1. read velocity features from Redis   │
                                      │  2. build feature vector                │
                                      │  3. XGBoost inference                   │
                                      │  4. write prediction to PostgreSQL      │
                                      │  5. update Redis feature store          │
                                      └──────────────────┬──────────────────────┘
                                                         │
                                                    PostgreSQL
                                                         │
                                          ┌──────────────┴──────────────┐
                                          │                             │
                                     FastAPI /metrics            Streamlit
                                     FastAPI /predict            Dashboard
                                     (sync scoring path)         (live view)
```

Kafka partitions transactions by `user_id`, so every event for a given user lands on the same partition in order — required for the processor's velocity window to be correct without any distributed coordination between consumers. The deployed (free-tier) version runs only the sync scoring API and dashboard, since a full Kafka + ZooKeeper stack needs more RAM than free hosting tiers give you; the async pipeline architecture above is fully implemented and runs via Docker Compose locally.

## Key Features

- Full streaming pipeline: Kafka producer → feature-enriched consumer → PostgreSQL, in addition to a synchronous FastAPI scoring path for cases that can't wait on the async pipeline
- Online feature store in Redis computing `tx_count_1h`, `tx_count_24h`, and `avg_amount_7d` per user before each write
- Training/serving parity enforced by saving the exact `feature_cols` list inside the model artifact, so the API and the Kafka processor can never compute a different feature set than the model was trained on
- FastAPI contract with Pydantic validation across `/predict`, `/metrics`, `/predictions`, `/model`, and `/health`
- Live Streamlit dashboard showing fraud rate, score distribution, and flagged transactions
- `pytest` + `fakeredis` test coverage for feature-store logic with no external infrastructure required
- CI (GitHub Actions) runs the test suite on every push and pull request

## Model

Trained on the [ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 fraud cases — 0.17% positive rate). Features are the dataset's PCA components (V1–V28), `Amount`/`hour_of_day`, plus the three engineered velocity features. Key modeling decisions:

| Decision | Reasoning |
|---|---|
| Time-based train/test split | A random split leaks future behavior into training, which makes the model look better in testing than it will in production |
| `scale_pos_weight` over synthetic resampling | Keeps the real feature space intact at a 0.17% fraud rate instead of introducing synthetic minority samples |
| XGBoost over Random Forest | Sequential boosting handles this fraud decision boundary better than bagging at this class ratio |
| PR-AUC as the primary metric | With this much class imbalance, ROC-AUC is misleadingly optimistic; precision-recall reflects the actual operational tradeoff |

Typical held-out test performance: **PR-AUC ~0.85**, recall ~0.82 at a 0.5 threshold. Full training/evaluation contract and limitations are in [docs/model-card.md](docs/model-card.md).

## Interesting Engineering Decisions

- **Feature parity is enforced by the artifact, not by convention.** The trained model saves its own `feature_cols` list, and both the API and the Kafka processor read that list rather than hardcoding a feature order in two places — the usual way training/serving skew creeps in.
- **Kafka partitioning key is what makes the velocity window correct.** Partitioning by `user_id` guarantees in-order processing per user on a single partition, which is what lets the Redis sorted-set windows stay correct without needing distributed locking or coordination across consumers.
- **The deployed version is intentionally a subset of the architecture.** Rather than force a full Kafka/ZooKeeper stack onto a free-tier host (or fake the architecture in the README), the deployment only runs what fits — sync API + dashboard — and the streaming path is documented and runnable locally via `docker-compose.yml`, which is the more honest way to demo a resource-heavy design under a free-tier budget.

## Running Locally

```bash
# 1. Get the dataset — download creditcard.csv from Kaggle and place it in data/
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 2. Train the model
make train
# Produces model/artifacts/model.pkl (~30 seconds)

# 3. Start the pipeline
cp .env.example .env
make up

# 4. Open
# Dashboard: http://localhost:8501
# API docs:  http://localhost:8000/docs

# 5. Tear down
make down   # stops containers and removes volumes
```

Example sync scoring call:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u0042", "amount": 249.99, "features": {"V1": -1.36, "V2": -0.07}, "time_offset": 3600}'
```

## Project Structure

```
├── model/                 offline training → model/artifacts/model.pkl
├── producer/              reads creditcard.csv, streams to Kafka at configurable TPS
├── processor/             Kafka consumer: Redis feature store + inference + PostgreSQL write
├── api/                   FastAPI — sync /predict + read-only metrics endpoints
├── dashboard/              Streamlit — live fraud rate, score distribution, flagged table
├── init.sql               PostgreSQL schema
└── docker-compose.yml     full stack: Kafka + ZooKeeper + Redis + PostgreSQL + all services
```
