"""Transaction 360-degree fraud analyst workstation."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from dashboard.cases import get_case_repository, load_case_queue, synchronize_cases
from src.features.fraud_features import FEATURE_COLUMNS, FraudFeatureEngineer
from src.investigation.case_builder import build_case_evidence
from src.investigation.report_generator import InvestigationReportDraft, InvestigationReportGenerator, save_report
from src.investigation.risk_explanation import ShapRiskExplainer
from src.models.predict import FraudModel
from src.rag.copilot import FraudInvestigationCopilot
from src.rag.knowledge_base import FraudKnowledgeBase
from src.rules.risk_scoring import UnifiedRiskScoringEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
from src.config import CSV_PATH, MODEL_PATH



@st.cache_resource
def scoring_components():
    return FraudModel(MODEL_PATH), ShapRiskExplainer(MODEL_PATH), UnifiedRiskScoringEngine()


@st.cache_data(show_spinner="Building account and relationship context…")
def load_context(origin: str, destination: str, step: int) -> dict[str, pd.DataFrame]:
    escaped = CSV_PATH.resolve().as_posix().replace("'", "''")
    safe_origin = origin.replace("'", "''")
    safe_destination = destination.replace("'", "''")
    with duckdb.connect() as connection:
        connection.execute(f"CREATE VIEW paysim AS SELECT * FROM read_csv_auto('{escaped}', header=true)")
        history = connection.execute(f"""
            SELECT step, type, amount, nameDest AS destination,
                   oldbalanceOrg AS balance_before, newbalanceOrig AS balance_after,
                   isFraud AS labelled_fraud
            FROM paysim
            WHERE nameOrig = '{safe_origin}' AND step < {int(step)}
            ORDER BY step DESC LIMIT 25
        """).fetchdf()
        origin_profile = connection.execute(f"""
            SELECT count(*) AS lifetime_transactions, coalesce(sum(amount), 0) AS lifetime_amount,
                   coalesce(avg(amount), 0) AS average_amount, coalesce(max(amount), 0) AS maximum_amount,
                   count(DISTINCT nameDest) AS unique_destinations,
                   min(step) AS first_step, max(step) AS latest_step
            FROM paysim WHERE nameOrig = '{safe_origin}' AND step < {int(step)}
        """).fetchdf()
        related = connection.execute(f"""
            SELECT nameOrig AS related_origin, count(*) AS transactions,
                   sum(amount) AS total_amount, coalesce(sum(isFraud), 0) AS labelled_frauds,
                   max(step) AS latest_step
            FROM paysim
            WHERE nameDest = '{safe_destination}' AND nameOrig <> '{safe_origin}' AND step < {int(step)}
            GROUP BY nameOrig ORDER BY labelled_frauds DESC, total_amount DESC LIMIT 15
        """).fetchdf()
        destination_profile = connection.execute(f"""
            SELECT count(*) AS incoming_transactions, count(DISTINCT nameOrig) AS unique_origins,
                   sum(amount) AS incoming_amount, coalesce(sum(isFraud), 0) AS labelled_frauds,
                   avg(isFraud) AS labelled_fraud_rate
            FROM paysim WHERE nameDest = '{safe_destination}' AND step < {int(step)}
        """).fetchdf()
    return {
        "history": history, "origin_profile": origin_profile,
        "related": related, "destination_profile": destination_profile,
    }


def analyze_case(case: pd.Series, context: dict[str, pd.DataFrame]):
    profile = context["origin_profile"].iloc[0]
    raw = pd.DataFrame([{
        "step": case["Step"], "type": case["Type"], "amount": case["Amount"],
        "nameOrig": case["Origin"], "oldbalanceOrg": case["Origin balance before"],
        "newbalanceOrig": case["Origin balance after"], "nameDest": case["Destination"],
        "oldbalanceDest": case["Destination balance before"],
        "newbalanceDest": case["Destination balance after"],
    }])
    features = FraudFeatureEngineer().transform(raw).iloc[0].to_dict()
    prior_count = int(profile.lifetime_transactions)
    features.update({
        "account_transaction_count": prior_count,
        "average_account_amount": float(profile.average_amount) if prior_count else 0.0,
        "destination_frequency": case["Destination frequency"],
        "deviation_from_account_average": case["Amount"] - float(profile.average_amount) if prior_count else 0.0,
    })
    model, explainer, risk_engine = scoring_components()
    probability = model.predict({column: features[column] for column in FEATURE_COLUMNS})["fraud_probability"]
    decision = risk_engine.score(features, probability)
    explanation = explainer.explain(features, top_n=5)
    return decision, explanation


def recommended_action(score: float) -> tuple[str, str]:
    if score > 80:
        return "FLAG & ESCALATE", "Prioritize retrospective investigation; no payment is blocked by this demo."
    if score > 60:
        return "MANUAL REVIEW", "Ask an investigator to review the completed transaction evidence."
    if score > 30:
        return "MONITOR", "Allow processing with enhanced account and destination monitoring."
    return "APPROVE", "No immediate intervention; retain the score for monitoring and audit."


def render() -> None:
    st.markdown('<div class="finshield-kicker">TRANSACTION 360°</div>', unsafe_allow_html=True)
    st.title("Fraud analyst workstation")
    queue = load_case_queue()
    synchronize_cases(queue)
    case_ids = queue["Case"].tolist()
    initial = case_ids.index(st.session_state["selected_case"]) if st.session_state.get("selected_case") in case_ids else 0
    selected = st.selectbox("Investigation case", case_ids, index=initial)
    st.session_state["selected_case"] = selected
    case = queue.loc[queue["Case"] == selected].iloc[0]
    managed_case = get_case_repository().get(selected)
    context = load_context(case["Origin"], case["Destination"], int(case["Step"]))
    decision, explanation = analyze_case(case, context)
    action, action_detail = recommended_action(decision.final_risk_score)

    top = st.columns([1, 1, 1, 1.35])
    top[0].metric("Risk score", f"{decision.final_risk_score:.1f}")
    top[1].metric("ML probability", f"{explanation.fraud_risk:.1%}")
    top[2].metric("Triggered rules", len(decision.rule_hits))
    top[3].metric("Recommended action", action)
    st.info(action_detail)
    st.caption(f"Case state: {managed_case.status.value} · Assigned: {managed_case.assigned_investigator or 'Unassigned'} · Updated: {managed_case.updated_at}")

    details, account = st.columns(2, gap="large")
    with details:
        st.markdown("### Transaction details")
        st.dataframe(pd.DataFrame({
            "Field": ["Transaction", "Type", "Amount", "PaySim step", "Origin", "Destination"],
            "Value": [case["Transaction"], case["Type"], f"₹{case['Amount']:,.2f}", case["Step"], case["Origin"], case["Destination"]],
        }).astype(str), hide_index=True, use_container_width=True)
    with account:
        st.markdown("### Customer / account details")
        profile = context["origin_profile"].iloc[0]
        st.dataframe(pd.DataFrame({
            "Metric": ["Account type", "Prior-step transactions", "Average amount", "Maximum amount", "Unique destinations", "Observed steps"],
            "Value": ["CUSTOMER", int(profile.lifetime_transactions), f"₹{profile.average_amount:,.2f}", f"₹{profile.maximum_amount:,.2f}", int(profile.unique_destinations), f"{profile.first_step}–{profile.latest_step}"],
        }).astype(str), hide_index=True, use_container_width=True)

    rules, shap_panel = st.columns(2, gap="large")
    with rules:
        st.markdown("### Triggered rules")
        if decision.rule_hits:
            for hit in decision.rule_hits:
                st.markdown(f"**{hit['severity']} · {hit['rule']}**  \n{hit['reason']} · +{hit['score']} points")
        else:
            st.success("No deterministic rules triggered.")
    with shap_panel:
        st.markdown("### SHAP explanation")
        for factor in explanation.primary_contributing_factors:
            st.markdown(f"**+ {factor.description}**  \nContribution: `{factor.contribution:+.4f}` · Value: `{factor.value}`")
        with st.expander("Protective factors"):
            for factor in explanation.protective_factors:
                st.write(f"− {factor.description} ({factor.contribution:+.4f})")

    history, relationships = st.columns(2, gap="large")
    with history:
        st.markdown("### Previous transactions & historical behavior")
        if context["history"].empty:
            st.warning("No prior origin transactions are present in PaySim. Treat as a new-account or limited-history case.")
        else:
            st.dataframe(context["history"], hide_index=True, use_container_width=True)
        st.caption(f"Historical average: ₹{profile.average_amount:,.2f} · Current deviation: ₹{case['Amount'] - profile.average_amount:,.2f}")
    with relationships:
        st.markdown("### Related destination accounts")
        destination = context["destination_profile"].iloc[0]
        st.caption(f"Destination received {int(destination.incoming_transactions):,} transactions from {int(destination.unique_origins):,} origins; {int(destination.labelled_frauds):,} are labelled fraud in PaySim.")
        if context["related"].empty:
            st.info("No other origins are linked to this destination.")
        else:
            st.dataframe(context["related"], hide_index=True, use_container_width=True)

    st.markdown("### AI Fraud Investigation Copilot")
    st.caption("Answers are generated only from the retrieved transaction, rule, model, account, and relationship evidence shown above.")
    st.caption("Try: What policy applies? · Why escalate? · What evidence should I collect? · Summarize this case · Generate investigation notes")
    question = st.text_input(
        "Ask about this case",
        value=f"Why was {case['Transaction']} flagged?",
        key=f"copilot_question_{selected}",
    )
    if st.button("Ask copilot", key=f"ask_copilot_{selected}", type="primary"):
        evidence = build_case_evidence(case, decision, explanation, context, f"{action}. {action_detail}")
        with st.spinner("Retrieving case evidence and preparing summary…"):
            answer = FraudInvestigationCopilot(knowledge_base=FraudKnowledgeBase.default()).answer(question, evidence)
        st.session_state[f"copilot_answer_{selected}"] = answer.to_dict()
    stored_answer = st.session_state.get(f"copilot_answer_{selected}")
    if stored_answer:
        st.markdown(stored_answer["answer"])
        st.caption(f"Grounding: {stored_answer['provider']} · Evidence: {', '.join(stored_answer['evidence_ids'])}")

    st.markdown("### AI Investigation Report")
    st.caption("Generate a grounded draft, edit it freely, then save an analyst-approved version with audit metadata.")
    analyst = st.text_input("Analyst", value="Pending", key=f"report_analyst_{selected}")
    if st.button("Generate Investigation Report", key=f"generate_report_{selected}", type="primary", use_container_width=True):
        with st.spinner("Retrieving policy and assembling grounded report…"):
            draft = InvestigationReportGenerator().generate(case, decision, explanation, context, analyst)
        st.session_state[f"report_draft_{selected}"] = draft.to_dict()
        st.session_state[f"report_content_{selected}"] = draft.content
    draft_data = st.session_state.get(f"report_draft_{selected}")
    if draft_data:
        edited = st.text_area(
            "Editable report draft", height=620, key=f"report_content_{selected}",
            help="Review every statement and edit the draft before saving.",
        )
        left_report, right_report = st.columns([1, 2])
        left_report.caption(f"Generator: {draft_data['generator']}")
        if right_report.button("Save analyst-approved report", key=f"save_report_{selected}", use_container_width=True):
            try:
                draft = InvestigationReportDraft(**draft_data)
                path = save_report(
                    draft, edited, PROJECT_ROOT / "data" / "processed" / "investigation_reports", analyst,
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.success(f"Saved {path.name} with content hash and analyst audit metadata.")

    st.markdown("### Investigation notes")
    note_key = f"360_notes_{selected}"
    disposition_key = f"360_disposition_{selected}"
    notes = st.text_area("Analyst notes", key=note_key, placeholder="Record evidence, outreach, and decision rationale…")
    left, right = st.columns([1, 3])
    disposition = left.selectbox("Disposition", ["IN_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "INCONCLUSIVE"], key=disposition_key)
    if right.button("Save investigation update", type="primary", use_container_width=True):
        st.session_state[f"saved_{selected}"] = {"notes": notes, "disposition": disposition}
        st.success(f"{selected} saved for this dashboard session.")

    st.caption("SHAP explains model behavior, not causality. PaySim account relationships and fraud labels are synthetic demonstration data.")
