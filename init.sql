CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    transaction_id  TEXT        NOT NULL,
    user_id         TEXT        NOT NULL,
    amount          NUMERIC(12, 2) NOT NULL,
    fraud_prob      FLOAT       NOT NULL,
    is_fraud        BOOLEAN     NOT NULL,
    model_version   TEXT        NOT NULL,
    -- velocity features captured at decision time, useful for offline analysis
    tx_count_1h     INTEGER,
    tx_count_24h    INTEGER,
    avg_amount_7d   FLOAT,
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- queries almost always filter by time or scan for fraud — cover both
CREATE INDEX IF NOT EXISTS idx_predictions_time    ON predictions (processed_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_fraud   ON predictions (is_fraud, processed_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_user    ON predictions (user_id, processed_at DESC);
