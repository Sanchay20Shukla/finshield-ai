"""Admin API, model performance, and drift monitoring dashboard."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.cases import get_feedback_repository
from src.monitoring.telemetry import MonitoringRepository, categorical_psi, numeric_drift


ROOT = Path(__file__).resolve().parents[2]


@st.cache_resource
def monitoring_repository() -> MonitoringRepository:
    return MonitoringRepository.default()


@st.cache_data(show_spinner=False)
def training_baseline():
    from src.config import CSV_PATH
    frame = pd.read_csv(CSV_PATH)
    return float(frame.amount.mean()), float(frame.amount.std()), frame.type.value_counts(normalize=True).to_dict()



def render() -> None:
    st.markdown('<div class="finshield-kicker">ADMIN · MODEL OBSERVABILITY</div>', unsafe_allow_html=True)
    st.title("API, model, and drift monitoring")
    api = monitoring_repository().api_frame()
    predictions = monitoring_repository().prediction_frame()
    feedback = get_feedback_repository().list()
    model = pd.read_csv(ROOT / "models" / "model_comparison.csv").set_index("model").loc["xgboost"]

    if not api.empty:
        api["occurred_at"] = pd.to_datetime(api["occurred_at"], utc=True)
    if not predictions.empty:
        predictions["occurred_at"] = pd.to_datetime(predictions["occurred_at"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    recent_predictions = predictions.loc[predictions["occurred_at"] >= now - pd.Timedelta(hours=1)] if not predictions.empty else predictions
    error_count = int((api["status_code"] >= 400).sum()) if not api.empty else 0
    latency_p95 = float(api["latency_ms"].quantile(.95)) if not api.empty else 0.0
    fraud_predictions = recent_predictions["decision"].isin(["MANUAL_REVIEW", "BLOCK"]).sum() if not recent_predictions.empty else 0
    average_risk = float(predictions["risk_score"].mean()) if not predictions.empty else 0.0
    predicted_fraud_rate = float(predictions["decision"].isin(["MANUAL_REVIEW", "BLOCK"]).mean()) if not predictions.empty else 0.0

    outcomes = [record.final_label for record in feedback]
    operational_precision = sum(outcomes) / len(outcomes) if outcomes else None
    operational_fpr = (len(outcomes) - sum(outcomes)) / len(outcomes) if outcomes else None
    first = st.columns(5)
    first[0].metric("API p95 latency", f"{latency_p95:,.1f} ms")
    first[1].metric("API errors", f"{error_count:,}", delta=f"{error_count / len(api):.2%}" if len(api) else "No traffic")
    first[2].metric("Transactions / hour", f"{len(recent_predictions):,}")
    first[3].metric("Fraud predictions / hour", f"{int(fraud_predictions):,}")
    first[4].metric("Average risk score", f"{average_risk:.2f}")
    second = st.columns(4)
    second[0].metric("Predicted fraud rate", f"{predicted_fraud_rate:.2%}")
    second[1].metric("Reviewed false-discovery share", f"{operational_fpr:.2%}" if operational_fpr is not None else "Pending labels")
    second[2].metric("Model precision", f"{operational_precision:.2%}" if operational_precision is not None else f"{model.precision:.2%} holdout")
    second[3].metric("Model recall", f"{model.recall:.2%} holdout", help="Operational recall requires labelled missed transactions, which the current review queue cannot observe.")

    left, right = st.columns(2)
    with left:
        if not api.empty:
            hourly = api.set_index("occurred_at").resample("1h").agg(requests=("event_id", "count"), latency_ms=("latency_ms", "mean"), errors=("status_code", lambda values: (values >= 400).sum())).reset_index()
            fig = px.line(hourly, x="occurred_at", y=["requests", "errors"], title="API traffic and errors")
            fig.update_layout(template="plotly_dark", height=330)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No API telemetry yet.")
    with right:
        if not predictions.empty:
            fig = px.histogram(predictions, x="risk_score", color="decision", nbins=20, title="Prediction risk distribution")
            fig.update_layout(template="plotly_dark", height=330)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Score transactions through the API to populate prediction monitoring.")

    st.markdown("### Data and model drift")
    train_mean, train_std, train_types = training_baseline()
    amount_drift = numeric_drift("average_transaction_amount", train_mean, train_std, predictions["amount"] if not predictions.empty else [])
    type_drift = categorical_psi(train_types, predictions["transaction_type"] if not predictions.empty else [])
    for drift in (amount_drift, type_drift):
        if drift.status == "DATA DRIFT WARNING":
            st.error(f"DATA DRIFT WARNING · {drift.metric}: {drift.explanation}")
        elif drift.status == "INSUFFICIENT_DATA":
            st.warning(f"{drift.metric}: {drift.explanation}")
        else:
            st.success(f"STABLE · {drift.metric}: {drift.explanation}")
    if len(outcomes) >= 20:
        precision_drop = float(model.precision) - float(operational_precision)
        if precision_drop >= 0.10:
            st.error(f"MODEL DRIFT WARNING · Operational precision is {precision_drop:.1%} below the holdout baseline.")
        else:
            st.success(f"MODEL PERFORMANCE STABLE · Operational precision is within {abs(precision_drop):.1%} of holdout precision.")
    else:
        st.warning(f"MODEL DRIFT PENDING · {20 - len(outcomes)} more reviewed outcomes required for the first operational precision check.")
    if not predictions.empty:
        st.dataframe(pd.DataFrame([
            {"Metric": "Average transaction amount", "Training": train_mean, "Production": amount_drift.production_value, "Drift score": amount_drift.drift_score, "Status": amount_drift.status},
            {"Metric": "Transaction type PSI", "Training": 0.0, "Production": type_drift.production_value, "Drift score": type_drift.drift_score, "Status": type_drift.status},
        ]), hide_index=True, use_container_width=True)
    st.caption("Holdout metrics are shown until sufficient investigator outcomes exist. Operational recall requires labels for missed fraud, not only reviewed alerts.")
