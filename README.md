# Real-time Fraud Detection Pipeline

End-to-end system for scoring card transactions as they arrive. Transactions stream through Kafka, a consumer enriches each one with user velocity features from Redis, runs XGBoost inference, and writes decisions to PostgreSQL. A FastAPI endpoint handles cases where a synchronous score is needed before the async pipeline can return. A Streamlit dashboard shows what the system is doing in real time.

**Live:** [API](https://fraud-detection-pipeline-pdv9.onrender.com/docs) · [Dashboard](https://fraud-detection-pipeline-8l3szzw6xzjmhhkcx8fq2x.streamlit.app) *(Render free tier — first request may take ~30s to wake)*

**Stack:** Python · Kafka · Redis · PostgreSQL · XGBoost · FastAPI · Streamlit · Docker Compose

---

## The problem this addresses

Most fraud detection tutorials train a model on a static dataset and call it done. That misses the part that makes it actually hard: a transaction arrives as a single row with no history attached. To answer "is this user transacting unusually often right now?" you need external state — specifically, a count of their recent transactions pulled from somewhere fast. That somewhere is a feature store.

This project builds the full loop:
- Offline training computes velocity features from the dataset's time ordering
- At inference time those same features come from Redis sorted sets with TTL-bounded windows
- The feature column list is saved with the model artifact so training and serving can never diverge

---

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

Kafka partitions transactions by `user_id`. This means all events for a given user land on the same partition in order — a necessary property for the processor's velocity window to be correct without distributed coordination.

---

## Model

**Dataset:** [ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud cases (0.17%).

**Features:**
- V1–V28: PCA components from the card network (original features are confidential)
- Amount, hour_of_day: derived from the dataset
- `tx_count_1h`, `tx_count_24h`, `avg_amount_7d`: velocity features computed per-user

**Training decisions:**

| Decision | Reasoning |
|---|---|
| Time-based train/test split | Random splits leak future behavior into training — a model trained that way performs better on test than in production |
| `scale_pos_weight` | 0.17% fraud rate. Balanced weighting without synthetic resampling keeps the feature space intact |
| XGBoost over RandomForest | Sequential boosting handles the fraud boundary better than bagging at this class ratio |
| PR-AUC as primary metric | With severe imbalance, ROC-AUC is misleadingly optimistic. Precision-recall measures the actual operational tradeoff |

Typical results on the held-out test set: **PR-AUC ~0.85**, recall ~0.82 at 0.5 threshold.

For the full training/evaluation contract, feature parity notes, inference contract, and limitations, see [docs/model-card.md](docs/model-card.md).

---

## Engineering Proof

| Area | Implementation |
|---|---|
| Training/serving parity | `feature_cols` are saved in the model artifact and used by both the API and Kafka processor |
| Online feature store | Redis sorted sets compute `tx_count_1h`, `tx_count_24h`, and `avg_amount_7d` before each write |
| Streaming correctness | Kafka partitions by `user_id` so each user's events are processed in order |
| API contract | FastAPI/Pydantic validates request shape and exposes `/predict`, `/metrics`, `/predictions`, `/model`, and `/health` |
| Test coverage | `pytest` + `fakeredis` validates feature-store behavior without external infrastructure |
| CI | GitHub Actions installs dependencies and runs the test suite on pushes and pull requests |

---

## Deployment (free tier)

The full Kafka pipeline is too resource-heavy for free hosting (ZooKeeper + Kafka alone need ~1.5GB RAM). The deployed version runs the sync scoring API and dashboard — the async pipeline architecture is documented below and runs locally.

**Services used:**
- [Neon](https://neon.tech) — serverless Postgres (free tier)
- [Upstash](https://upstash.com) — managed Redis (free tier)
- [Render](https://render.com) — API deployment (free web service)
- [Streamlit Cloud](https://streamlit.io/cloud) — dashboard (free)

**Steps:**

**1. Neon — create a database**
- Sign up at neon.tech, create a project, copy the connection string
- Run `init.sql` against it once: `psql <connection-string> -f init.sql`

**2. Upstash — create a Redis instance**
- Sign up at upstash.com → Redis → Create database
- Copy the `rediss://` connection URL

**3. Render — deploy the API**
- Connect your GitHub repo, select "New Web Service"
- Render auto-detects `render.yaml` — just fill in `DATABASE_URL` and `REDIS_URL` environment variables
- After deploy, hit `POST /demo/seed` once to populate the database: `curl -X POST https://<your-render-url>/demo/seed`

**4. Streamlit Cloud — deploy the dashboard**
- Go to share.streamlit.io → New app
- Repo: `vinay23is/fraud-detection-pipeline`, branch: `main`, main file: `dashboard/app.py`
- Add secret: `DATABASE_URL = "<neon-connection-string>"`

---

## Running locally

**1. Get the dataset**
```bash
# Download creditcard.csv from Kaggle and place it in data/
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

**2. Train the model**
```bash
make train
# Produces model/artifacts/model.pkl (~30 seconds)
```

**3. Start the pipeline**
```bash
cp .env.example .env
make up
```

**4. Open**
- Dashboard: http://localhost:8501
- API docs:   http://localhost:8000/docs

**5. Tear down**
```bash
make down   # stops containers and removes volumes
```

---

## API

**Sync scoring** — for cases where you need a decision before the Kafka pipeline returns:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u0042",
    "amount": 249.99,
    "features": {"V1": -1.36, "V2": -0.07, ...},
    "time_offset": 3600
  }'
```

```json
{
  "transaction_id": "a3f1...",
  "fraud_probability": 0.0043,
  "is_fraud": false,
  "model_version": "a1b2c3d4",
  "velocity": {"tx_count_1h": 2, "tx_count_24h": 5, "avg_amount_7d": 87.50}
}
```

**Metrics:**
```bash
curl http://localhost:8000/metrics?hours=1
```

---

## Project structure

```
├── model/
│   ├── train.py          offline training, produces model/artifacts/model.pkl
│   └── artifacts/        .gitignored — generated by make train
├── producer/
│   └── producer.py       reads creditcard.csv, streams to Kafka at configurable TPS
├── processor/
│   ├── feature_store.py  Redis sorted-set feature store (velocity windows)
│   └── processor.py      Kafka consumer, inference, PostgreSQL write
├── api/
│   └── main.py           FastAPI — sync /predict + read-only metrics endpoints
├── dashboard/
│   └── app.py            Streamlit — live fraud rate, score distribution, flagged table
├── init.sql              PostgreSQL schema
└── docker-compose.yml    full stack: Kafka + ZooKeeper + Redis + PostgreSQL + all services
```
