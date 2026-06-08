# Model Card — Fraud Detection Pipeline

## Purpose

This model scores card transactions for fraud risk in a streaming pipeline. It is designed to demonstrate the engineering path from offline training to online inference, not to make production financial decisions without human review, monitoring, and compliance controls.

## Dataset

The training data uses the ULB Credit Card Fraud dataset:

- 284,807 transactions
- 492 fraud cases
- 0.17% positive class rate
- PCA-transformed `V1`-`V28` features, plus `Amount` and time-derived features

The dataset does not include cardholder IDs, so the project assigns deterministic synthetic `user_id` values to model user-level velocity features.

## Features

| Feature group | Source | Used at training | Used at inference |
|---|---|---:|---:|
| `V1`-`V28` | Dataset PCA fields | Yes | Yes |
| `Amount` | Transaction payload | Yes | Yes |
| `hour_of_day` | `Time` / `time_offset` | Yes | Yes |
| `tx_count_1h` | Rolling user history | Yes | Yes, Redis |
| `tx_count_24h` | Rolling user history | Yes | Yes, Redis |
| `avg_amount_7d` | Rolling user history | Yes | Yes, Redis |

The feature list is saved inside `model/artifacts/model.pkl` so the processor and API build the inference vector in the same order as training.

## Training Design

| Decision | Reason |
|---|---|
| Time-based train/test split | Avoids future leakage from random splits |
| XGBoost classifier | Handles non-linear decision boundaries and class imbalance well |
| `scale_pos_weight` | Compensates for the 0.17% fraud rate without synthetic samples |
| PR-AUC primary metric | More meaningful than ROC-AUC for severe class imbalance |
| Redis-backed online features | Keeps training and serving feature definitions aligned |

## Evaluation

The training script prints:

- PR-AUC on the held-out future test set
- Precision, recall, and F1 at the default threshold
- Best-F1 threshold for operating-point analysis
- Model version hash saved with the artifact

Typical held-out performance:

| Metric | Value |
|---|---:|
| PR-AUC | ~0.85 |
| Recall at `0.5` threshold | ~0.82 |
| Default threshold | `0.5` |

The exact values can move slightly if dependencies or generated user assignments change.

## Inference Contract

`POST /predict` expects:

- `user_id`
- positive `amount`
- PCA features `V1`-`V28`
- `time_offset` in seconds

The API reads velocity features from Redis before writing the current transaction, so a transaction never counts itself in its own rolling-window features.

The response includes:

- `fraud_probability`
- `is_fraud`
- `model_version`
- velocity features used for the decision

## Operational Notes

- The async path partitions Kafka messages by `user_id`, preserving per-user transaction order.
- The sync API path and Kafka processor use the same model artifact and feature-store logic.
- Prediction records are written to PostgreSQL for dashboarding and auditability.
- `/model` exposes model version, PR-AUC, threshold, and feature list.

## Limitations

- PCA features cannot be interpreted directly because the original card-network fields are anonymized.
- Synthetic `user_id` assignment is a project constraint; a production system would use real account/card identifiers.
- The deployed free-tier version runs the API and dashboard; the full Kafka pipeline is intended for local Docker Compose.
- A production system would add calibrated probabilities, drift monitoring, approval/decline policy controls, and manual review queues.
