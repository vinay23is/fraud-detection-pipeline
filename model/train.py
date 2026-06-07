"""
Offline training script for the fraud detection model.

A few deliberate choices worth explaining:

1. Time-based split, not random.
   Fraud patterns shift over time. A random split leaks future information into
   training — the model sees behavioral patterns from "tomorrow" when learning
   about "today". Using the last 20% of transactions as the test set gives an
   honest estimate of how the model performs on unseen future data.

2. class_weight='balanced' instead of SMOTE.
   The dataset is 0.17% fraud. Resampling with SMOTE inflates training time and
   can introduce artifacts on PCA-transformed features (V1-V28 here). Balanced
   class weights achieve similar recall with less complexity.

3. XGBoost over RandomForest.
   XGBoost's sequential boosting catches the subtle patterns that distinguish
   the fraud cluster at the boundary better than bagging approaches. It also
   supports scale_pos_weight natively.

4. PR-AUC as the primary metric, not ROC-AUC.
   With severe class imbalance, ROC-AUC is optimistic (the large negative class
   makes FPR look good even with many missed frauds). Precision-recall directly
   measures the tradeoff that matters operationally.

5. Velocity features require a feature store at inference time.
   tx_count_1h and tx_count_24h cannot be derived from a single transaction —
   they depend on a user's recent history. During training we compute them from
   the dataset; at runtime the processor reads them from Redis. The feature names
   saved in the artifact enforce consistency between the two paths.
"""

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DATA_PATH = Path(os.environ.get("DATA_PATH", "data/creditcard.csv"))
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "artifacts"))
FRAUD_THRESHOLD = 0.5   # adjustable at serving time without retraining


def load_and_enrich(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path)

    # Simulate user IDs. The dataset has no user identifier, so we assign
    # transactions to 800 synthetic users. The assignment is index-based to keep
    # it deterministic across runs without leaking class information.
    rng = np.random.default_rng(seed=42)
    user_pool = [f"u{i:04d}" for i in range(800)]
    # Skew the distribution — a small number of users generate most transactions,
    # which is realistic and creates meaningful velocity signal.
    weights = rng.exponential(scale=1.0, size=len(user_pool))
    weights /= weights.sum()
    df["user_id"] = rng.choice(user_pool, size=len(df), p=weights)

    df["hour_of_day"] = (df["Time"] / 3600) % 24

    return df.sort_values("Time").reset_index(drop=True)


def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each transaction, count how many prior transactions the same user
    made in the last 1 hour and 24 hours, and compute their rolling average
    amount over the last 7 days.

    This mirrors what the Redis feature store does at inference time, so the
    feature distribution at train time matches the feature distribution in prod.
    """
    print("Computing velocity features (this takes ~30s) ...")

    time_s = df["Time"].values
    user_ids = df["user_id"].values
    amounts = df["Amount"].values

    tx_count_1h = np.zeros(len(df), dtype=np.int32)
    tx_count_24h = np.zeros(len(df), dtype=np.int32)
    avg_amount_7d = np.zeros(len(df), dtype=np.float32)

    # Index transactions by user for fast lookups
    from collections import defaultdict
    user_indices: dict[str, list[int]] = defaultdict(list)

    for i, (t, uid, amt) in enumerate(zip(time_s, user_ids, amounts)):
        prior = user_indices[uid]
        # walk backwards through this user's prior transactions
        c1h = c24h = 0
        amt_sum_7d = 0.0
        cnt_7d = 0
        for j in reversed(prior):
            dt = t - time_s[j]
            if dt <= 3600:
                c1h += 1
            if dt <= 86400:
                c24h += 1
            if dt <= 7 * 86400:
                amt_sum_7d += amounts[j]
                cnt_7d += 1
            if dt > 7 * 86400:
                break
        tx_count_1h[i] = c1h
        tx_count_24h[i] = c24h
        avg_amount_7d[i] = amt_sum_7d / cnt_7d if cnt_7d > 0 else amt

        user_indices[uid].append(i)

    df = df.copy()
    df["tx_count_1h"] = tx_count_1h
    df["tx_count_24h"] = tx_count_24h
    df["avg_amount_7d"] = avg_amount_7d
    return df


def build_features(df: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    feature_cols = (
        [f"V{i}" for i in range(1, 29)]
        + ["Amount", "hour_of_day", "tx_count_1h", "tx_count_24h", "avg_amount_7d"]
    )
    X = df[feature_cols].values.astype(np.float32)
    y = df["Class"].values
    return feature_cols, X, y


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def train(df_train: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    _, X_train, y_train = build_features(df_train)

    # scale_pos_weight: ratio of negatives to positives — equivalent to
    # class_weight='balanced' for XGBoost
    neg, pos = np.bincount(y_train)
    spw = neg / pos
    print(f"  Training set — negatives: {neg:,}  positives: {pos:,}  scale_pos_weight: {spw:.1f}")

    pipeline = Pipeline([
        # Amount and velocity features need scaling; V1-V28 are already unit-scaled
        # from PCA, but scaling them again does no harm and keeps the pipeline simple.
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=spw,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(pipeline: Pipeline, df_test: pd.DataFrame, feature_cols: list[str]):
    _, X_test, y_test = build_features(df_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= FRAUD_THRESHOLD).astype(int)

    pr_auc = average_precision_score(y_test, probs)
    print(f"\nTest set PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_test, preds, target_names=["legit", "fraud"], digits=4))

    # Find the threshold that maximises F1 — useful context even if we ship 0.5
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = f1_scores.argmax()
    print(
        f"Best F1 threshold: {thresholds[best_idx]:.3f}  "
        f"→  precision={precision[best_idx]:.3f}  recall={recall[best_idx]:.3f}"
    )
    return pr_auc


def save_artifact(pipeline: Pipeline, feature_cols: list[str], pr_auc: float, artifact_dir: Path):
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Version the model by hashing its parameters so the processor can log
    # which model version produced each prediction.
    params_str = json.dumps(
        pipeline.named_steps["clf"].get_params(), sort_keys=True, default=str
    )
    version = hashlib.md5(params_str.encode()).hexdigest()[:8]

    artifact = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "threshold": FRAUD_THRESHOLD,
        "pr_auc": round(pr_auc, 4),
        "version": version,
    }

    out = artifact_dir / "model.pkl"
    with open(out, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nSaved → {out}  (version={version}  PR-AUC={pr_auc:.4f})")


def main():
    if not DATA_PATH.exists():
        print(
            f"Dataset not found at {DATA_PATH}.\n"
            "Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the data/ directory."
        )
        sys.exit(1)

    df = load_and_enrich(DATA_PATH)
    df = compute_velocity_features(df)

    df_train, df_test = time_split(df)
    print(f"Train: {len(df_train):,}  |  Test: {len(df_test):,}")

    feature_cols, _, _ = build_features(df_train)
    pipeline = train(df_train, feature_cols)
    pr_auc = evaluate(pipeline, df_test, feature_cols)
    save_artifact(pipeline, feature_cols, pr_auc, ARTIFACT_DIR)


if __name__ == "__main__":
    main()
