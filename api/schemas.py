from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    user_id: str
    amount: float = Field(gt=0)
    features: dict[str, float] = Field(
        description="V1–V28 from the card network (PCA-encoded)"
    )
    time_offset: float = Field(
        default=0.0,
        description="Seconds since the cardholder's first transaction on record"
    )


class PredictionResponse(BaseModel):
    transaction_id: str
    user_id: str
    fraud_probability: float
    is_fraud: bool
    model_version: str
    velocity: dict


class AggregateMetrics(BaseModel):
    total_transactions: int
    total_flagged: int
    fraud_rate_pct: float
    avg_fraud_prob: float
    window_hours: int


class RecentPrediction(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    fraud_prob: float
    is_fraud: bool
    tx_count_1h: Optional[int]
    processed_at: datetime
