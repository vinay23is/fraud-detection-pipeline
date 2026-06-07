"""
Streamlit dashboard — live view of the fraud detection pipeline.

Refreshes automatically every 3 seconds. Connects directly to PostgreSQL
rather than going through the API to keep the dashboard decoupled from
API availability.
"""

import os
import time

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st

DATABASE_URL = os.environ["DATABASE_URL"]

st.set_page_config(
    page_title="Fraud Detection — Live",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def get_conn():
    return psycopg2.connect(DATABASE_URL)


def query(sql: str, params=()) -> pd.DataFrame:
    conn = get_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall())


# ── layout ────────────────────────────────────────────────────────────────────

st.title("🔍 Fraud Detection Pipeline")
st.caption("Auto-refreshes every 3 seconds")

placeholder = st.empty()

while True:
    with placeholder.container():

        # ── top-line metrics ──────────────────────────────────────────────────
        summary = query("""
            SELECT
                COUNT(*)                           AS total,
                SUM(is_fraud::int)                 AS flagged,
                AVG(fraud_prob)                    AS avg_prob,
                MAX(processed_at)                  AS last_seen
            FROM predictions
            WHERE processed_at >= NOW() - INTERVAL '1 hour'
        """)

        row = summary.iloc[0]
        total   = int(row["total"]   or 0)
        flagged = int(row["flagged"] or 0)
        avg_p   = float(row["avg_prob"] or 0)
        last    = row["last_seen"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions (1h)",  f"{total:,}")
        c2.metric("Flagged (1h)",       f"{flagged:,}")
        c3.metric("Fraud Rate",         f"{flagged/total*100:.2f}%" if total else "—")
        c4.metric("Avg Fraud Prob",     f"{avg_p:.3f}")

        st.divider()

        col_left, col_right = st.columns(2)

        # ── fraud rate over time ──────────────────────────────────────────────
        with col_left:
            st.subheader("Fraud rate — last 60 minutes")
            ts_df = query("""
                SELECT
                    date_trunc('minute', processed_at) AS minute,
                    COUNT(*)                           AS total,
                    SUM(is_fraud::int)                 AS fraud
                FROM predictions
                WHERE processed_at >= NOW() - INTERVAL '60 minutes'
                GROUP BY 1
                ORDER BY 1
            """)
            if not ts_df.empty:
                ts_df["rate"] = ts_df["fraud"] / ts_df["total"] * 100
                st.line_chart(ts_df.set_index("minute")["rate"], height=220)
            else:
                st.info("Waiting for data...")

        # ── fraud probability distribution ────────────────────────────────────
        with col_right:
            st.subheader("Score distribution (last 5 min)")
            dist_df = query("""
                SELECT fraud_prob
                FROM predictions
                WHERE processed_at >= NOW() - INTERVAL '5 minutes'
            """)
            if not dist_df.empty:
                counts = (
                    dist_df["fraud_prob"]
                    .pipe(pd.cut, bins=20)
                    .value_counts()
                    .sort_index()
                )
                counts.index = [str(i) for i in counts.index]
                st.bar_chart(counts.rename("count"), height=220)
            else:
                st.info("Waiting for data...")

        st.divider()

        # ── recent flagged transactions ───────────────────────────────────────
        st.subheader("Recently flagged transactions")
        flagged_df = query("""
            SELECT
                to_char(processed_at, 'HH24:MI:SS') AS time,
                transaction_id,
                user_id,
                amount,
                round(fraud_prob::numeric, 4)        AS score,
                tx_count_1h                          AS "1h tx count"
            FROM predictions
            WHERE is_fraud = TRUE
            ORDER BY processed_at DESC
            LIMIT 20
        """)
        if flagged_df.empty:
            st.success("No fraud flagged in the current window.")
        else:
            st.dataframe(
                flagged_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "amount": st.column_config.NumberColumn(format="$%.2f"),
                    "score":  st.column_config.ProgressColumn(min_value=0, max_value=1),
                },
            )

    time.sleep(3)
    placeholder.empty()
