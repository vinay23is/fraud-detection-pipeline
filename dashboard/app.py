"""
Streamlit dashboard — live view of the fraud detection pipeline.
Refreshes automatically every 5 seconds.
"""

import os
import time

import pandas as pd
import psycopg2
import psycopg2.extras
import streamlit as st

DATABASE_URL = st.secrets.get("DATABASE_URL") or os.environ["DATABASE_URL"]

st.set_page_config(
    page_title="Fraud Detection — Live",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
/* dark background */
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stHeader"] { background: transparent; }

/* metric cards */
[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700; }

/* section headers */
h3 { color: #e2e8f0 !important; font-size: 0.85rem !important;
     text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem !important; }

div[data-testid="stHorizontalBlock"] { gap: 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_conn():
    return psycopg2.connect(DATABASE_URL)


def query(sql: str, params=()) -> pd.DataFrame:
    conn = get_conn()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall())


# ── header ────────────────────────────────────────────────────────────────────

st.markdown("## 🛡️ &nbsp;Fraud Detection Pipeline")
st.markdown(
    "<p style='color:#64748b; margin-top:-12px; font-size:0.85rem;'>"
    "XGBoost · Kafka · Redis feature store · PostgreSQL &nbsp;·&nbsp; "
    "refreshes every 5 seconds</p>",
    unsafe_allow_html=True,
)

placeholder = st.empty()

while True:
    with placeholder.container():

        # ── top-line metrics ──────────────────────────────────────────────────
        summary = query("""
            SELECT
                COUNT(*)               AS total,
                SUM(is_fraud::int)     AS flagged,
                AVG(fraud_prob)        AS avg_prob,
                MAX(processed_at)      AS last_seen
            FROM predictions
            WHERE processed_at >= NOW() - INTERVAL '1 hour'
        """)

        row     = summary.iloc[0]
        total   = int(row["total"]   or 0)
        flagged = int(row["flagged"] or 0)
        avg_p   = float(row["avg_prob"] or 0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions (1h)", f"{total:,}")
        c2.metric("Flagged (1h)",      f"{flagged:,}",
                  delta=None if flagged == 0 else f"{flagged} alerts",
                  delta_color="inverse")
        c3.metric("Fraud Rate",        f"{flagged/total*100:.2f}%" if total else "—")
        c4.metric("Avg Risk Score",    f"{avg_p:.4f}")

        st.markdown("<div style='margin: 1.5rem 0 0.5rem; border-top: 1px solid #2a2d3a'></div>",
                    unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        # ── fraud rate over time ──────────────────────────────────────────────
        with col_left:
            st.subheader("Fraud rate — last 60 min")
            ts_df = query("""
                SELECT
                    date_trunc('minute', processed_at) AS minute,
                    COUNT(*)                           AS total,
                    SUM(is_fraud::int)                 AS fraud
                FROM predictions
                WHERE processed_at >= NOW() - INTERVAL '60 minutes'
                GROUP BY 1 ORDER BY 1
            """)
            if not ts_df.empty:
                ts_df["rate"] = ts_df["fraud"] / ts_df["total"] * 100
                st.line_chart(
                    ts_df.set_index("minute")["rate"],
                    height=240,
                    color="#6366f1",
                )
            else:
                st.info("Waiting for data...")

        # ── score distribution ────────────────────────────────────────────────
        with col_right:
            st.subheader("Score distribution (last 10 min)")
            dist_df = query("""
                SELECT fraud_prob FROM predictions
                WHERE processed_at >= NOW() - INTERVAL '10 minutes'
            """)
            if not dist_df.empty:
                counts = (
                    dist_df["fraud_prob"]
                    .pipe(pd.cut, bins=20)
                    .value_counts()
                    .sort_index()
                )
                counts.index = [str(i) for i in counts.index]
                st.bar_chart(counts.rename("count"), height=240, color="#818cf8")
            else:
                st.info("Waiting for data...")

        st.markdown("<div style='margin: 1rem 0 0.5rem; border-top: 1px solid #2a2d3a'></div>",
                    unsafe_allow_html=True)

        # ── throughput sparkline ──────────────────────────────────────────────
        st.subheader("Throughput — transactions per minute")
        tpm_df = query("""
            SELECT
                date_trunc('minute', processed_at) AS minute,
                COUNT(*)                           AS count
            FROM predictions
            WHERE processed_at >= NOW() - INTERVAL '30 minutes'
            GROUP BY 1 ORDER BY 1
        """)
        if not tpm_df.empty:
            st.area_chart(tpm_df.set_index("minute")["count"], height=140, color="#34d399")
        else:
            st.info("Waiting for data...")

        st.markdown("<div style='margin: 1rem 0 0.5rem; border-top: 1px solid #2a2d3a'></div>",
                    unsafe_allow_html=True)

        # ── flagged transactions ──────────────────────────────────────────────
        st.subheader("Recently flagged transactions")
        flagged_df = query("""
            SELECT
                to_char(processed_at, 'HH24:MI:SS') AS time,
                user_id,
                amount,
                round(fraud_prob::numeric, 4)        AS score,
                tx_count_1h                          AS "tx / 1h",
                transaction_id
            FROM predictions
            WHERE is_fraud = TRUE
            ORDER BY processed_at DESC
            LIMIT 20
        """)
        if flagged_df.empty:
            st.success("✓  No fraud flagged in the last window.")
        else:
            st.dataframe(
                flagged_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "amount":         st.column_config.NumberColumn(format="$%.2f"),
                    "score":          st.column_config.ProgressColumn(
                                          min_value=0, max_value=1, format="%.4f"),
                    "transaction_id": st.column_config.TextColumn(width="small"),
                },
            )

    time.sleep(5)
    placeholder.empty()
