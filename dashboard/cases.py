"""Demonstration investigation queue built from scored PaySim fraud records."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from src.features.fraud_features import FEATURE_COLUMNS, FraudFeatureEngineer
from src.models.predict import FraudModel
from src.models.feedback import FeedbackRepository
from src.rules.risk_scoring import UnifiedRiskScoringEngine
from src.investigation.case_management import CaseRepository


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def get_case_repository() -> CaseRepository:
    return CaseRepository.default()


@st.cache_resource
def get_feedback_repository() -> FeedbackRepository:
    return FeedbackRepository.default()


def synchronize_cases(queue: pd.DataFrame) -> dict[str, object]:
    repository = get_case_repository()
    return {
        row["Case"]: repository.create(row["Case"], row["Transaction"], float(row["Risk"]))
        for _, row in queue.iterrows()
    }


def feature_snapshot_for_case(case: pd.Series) -> dict:
    raw = pd.DataFrame([{
        "step": case["Step"], "type": case["Type"], "amount": case["Amount"],
        "nameOrig": case["Origin"], "oldbalanceOrg": case["Origin balance before"],
        "newbalanceOrig": case["Origin balance after"], "nameDest": case["Destination"],
        "oldbalanceDest": case["Destination balance before"],
        "newbalanceDest": case["Destination balance after"],
    }])
    featured = FraudFeatureEngineer().transform(raw).iloc[0].to_dict()
    featured["destination_frequency"] = int(case["Destination frequency"])
    return {column: featured[column] for column in FEATURE_COLUMNS}


@st.cache_data(show_spinner="Scoring investigation queue…")
def load_case_queue(limit: int = 100) -> pd.DataFrame:
    from src.config import CSV_PATH, MODEL_PATH
    model_path = MODEL_PATH
    raw = pd.read_csv(CSV_PATH, nrows=10000).sort_values("step", kind="stable")
    # Mixed labels, no ground-truth label filtering or future destination counts.
    raw = raw.head(limit).reset_index(drop=True)
    raw = raw.sort_values("step", kind="stable").reset_index(drop=True)
    featured = FraudFeatureEngineer().transform(raw)
    model = FraudModel(model_path)
    probabilities = model.pipeline.predict_proba(featured[FEATURE_COLUMNS])[:, 1]
    engine = UnifiedRiskScoringEngine()
    records = []
    for index, (row_index, row) in enumerate(featured.iterrows(), start=1):
        decision = engine.score(row.to_dict(), float(probabilities[row_index]))
        records.append({
            "Case": f"C{1000 + index}",
            "Transaction": f"TX-{int(row['step']):03d}-{str(row['origin_account_id'])[-6:]}",
            "Origin": row["origin_account_id"],
            "Destination": row["destination_account_id"],
            "Type": row["transaction_type"],
            "Amount": float(row["amount"]),
            "Risk": round(decision.final_risk_score, 1),
            "Status": "Critical" if decision.final_risk_score > 80 else "Review" if decision.final_risk_score > 60 else "Monitor" if decision.final_risk_score > 30 else "Low risk",
            "ML score": decision.ml_score,
            "Rule score": decision.rule_score,
            "Behavioral score": decision.behavioral_score,
            "Rules": ", ".join(hit["rule"] for hit in decision.rule_hits) or "None",
            "Origin balance before": float(row["origin_old_balance"]),
            "Origin balance after": float(row["origin_new_balance"]),
            "Destination balance before": float(row["destination_old_balance"]),
            "Destination balance after": float(row["destination_new_balance"]),
            "Destination frequency": int(row["destination_frequency"]),
            "Step": int(row["step"]),
        })
    return pd.DataFrame(records).sort_values(["Risk", "Amount"], ascending=False).reset_index(drop=True)


def render_case(case: pd.Series) -> None:
    st.markdown(f"### {case['Case']} · {case['Transaction']}")
    severity = "🔴" if case["Status"] == "Critical" else "🟠"
    st.markdown(f"{severity} **{case['Status']}** · {case['Type']} · ₹{case['Amount']:,.2f}")
    a, b, c = st.columns(3)
    a.metric("Final risk", f"{case['Risk']:.1f}")
    b.metric("ML score", f"{case['ML score']:.1f}")
    c.metric("Rule score", f"{case['Rule score']:.1f}")
    st.markdown("**Triggered rules**")
    st.write(case["Rules"])
    st.markdown("**Account flow**")
    st.caption(f"{case['Origin']}  →  {case['Destination']}")
    st.write(f"Origin: ₹{case['Origin balance before']:,.2f} → ₹{case['Origin balance after']:,.2f}")
    st.write(f"Destination: ₹{case['Destination balance before']:,.2f} → ₹{case['Destination balance after']:,.2f}")
