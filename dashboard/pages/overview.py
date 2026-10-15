"""Observed dataset summaries; no invented prevented-loss estimates."""
import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from src.config import CSV_PATH, ROOT


@st.cache_data(show_spinner="Summarizing synthetic transactions…")
def load_portfolio():
    path = CSV_PATH.resolve().as_posix().replace("'", "''")
    with duckdb.connect() as con:
        con.execute(f"CREATE VIEW paysim AS SELECT * FROM read_csv_auto('{path}', header=true)")
        summary = con.execute("SELECT count(*) transactions, sum(isFraud) fraud_count, sum(amount) total_amount, avg(isFraud) fraud_rate FROM paysim").fetchdf()
        trend = con.execute("SELECT step, count(*) transactions, sum(isFraud) fraud_count FROM paysim GROUP BY step ORDER BY step").fetchdf()
        by_type = con.execute("SELECT type, count(*) transactions, sum(isFraud) labelled_frauds FROM paysim GROUP BY type").fetchdf()
    return summary.iloc[0], trend, by_type


def render():
    st.markdown('<div class="finshield-kicker">FRAUD INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("Every transaction tells a story")
    st.caption("Observed synthetic dataset labels. Fraud counts below are ground truth, not model detections or prevented losses.")
    summary, trend, by_type = load_portfolio()
    if "sample" in CSV_PATH.parts:
        st.info("Curated 168-row demo sample enriched by label and type. Its fraud rate is not population prevalence.")
    a,b,c,d = st.columns(4)
    a.metric("Dataset transactions", f"{int(summary.transactions):,}")
    b.metric("Labelled fraud", f"{int(summary.fraud_count):,}")
    c.metric("Labelled fraud share", f"{summary.fraud_rate:.2%}")
    d.metric("Total value · synthetic units", f"{summary.total_amount:,.0f}")
    left,right=st.columns(2)
    with left:
        st.plotly_chart(px.line(trend,x="step",y="transactions",title="Transaction volume by step",color_discrete_sequence=["#63e6bd"]),use_container_width=True)
    with right:
        st.plotly_chart(px.bar(by_type,x="type",y="labelled_frauds",title="Labelled fraud by transaction type",color_discrete_sequence=["#ff8d92"]),use_container_width=True)
    st.subheader("Start an investigation")
    st.write("Use the alert queue to inspect model and policy evidence, or score your own transaction. The model uses post-event balances and supports retrospective detection.")
    st.subheader("Historical baseline comparison")
    st.caption("Existing artifact metrics, not reproduced in this release. Historical model ranking used holdout loss; future training now selects by validation loss. Probabilities are uncalibrated.")
    comparison=pd.read_csv(ROOT / "models/model_comparison.csv")
    st.dataframe(comparison[["model","threshold","precision","recall","pr_auc","expected_loss"]],hide_index=True,use_container_width=True)
