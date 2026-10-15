"""Investigation workspace page."""

import streamlit as st

from dashboard.cases import (
    feature_snapshot_for_case, get_case_repository, get_feedback_repository,
    load_case_queue, render_case, synchronize_cases,
)
from src.investigation.case_management import ALLOWED_TRANSITIONS, CaseStatus, InvalidCaseTransition


def render() -> None:
    st.markdown('<div class="finshield-kicker">CASE MANAGEMENT</div>', unsafe_allow_html=True)
    st.title("Investigation workspace")
    queue = load_case_queue()
    synchronize_cases(queue)
    repository = get_case_repository()
    feedback_repository = get_feedback_repository()
    selected = st.selectbox(
        "Case", queue["Case"].tolist(),
        index=(queue["Case"].tolist().index(st.session_state["selected_case"]) if st.session_state.get("selected_case") in set(queue["Case"]) else 0),
    )
    st.session_state["selected_case"] = selected
    case = queue.loc[queue["Case"] == selected].iloc[0]
    render_case(case)
    record = repository.get(selected)
    a, b, c = st.columns(3)
    a.metric("Workflow state", record.status.value)
    b.metric("Assigned investigator", record.assigned_investigator or "Unassigned")
    c.metric("Decision", record.decision or "Pending")
    st.divider()
    st.markdown("### Case workflow")
    if record.status is CaseStatus.NEW:
        investigator = st.text_input("Assign investigator", key=f"assignee_{selected}", placeholder="Analyst name")
        if st.button("Assign case", type="primary", use_container_width=True):
            try:
                repository.assign(selected, investigator)
            except (ValueError, InvalidCaseTransition) as error:
                st.error(str(error))
            else:
                st.rerun()
    elif record.status is CaseStatus.ASSIGNED:
        if st.button("Start investigation", type="primary", use_container_width=True):
            repository.transition(selected, CaseStatus.INVESTIGATING)
            st.rerun()
    elif record.status is CaseStatus.INVESTIGATING:
        fraud, false_positive = st.columns(2)
        if fraud.button("Confirm fraud", type="primary", use_container_width=True):
            feedback_repository.record(
                case_id=selected, transaction_id=case["Transaction"], model_name="xgboost",
                model_version="paysim-baseline-v1", fraud_probability=float(case["ML score"]) / 100,
                risk_score=float(case["Risk"]), investigator=record.assigned_investigator,
                final_outcome="CONFIRMED_FRAUD", feature_snapshot=feature_snapshot_for_case(case), notes=record.notes,
            )
            repository.transition(selected, CaseStatus.CONFIRMED_FRAUD)
            st.rerun()
        if false_positive.button("Mark false positive", use_container_width=True):
            feedback_repository.record(
                case_id=selected, transaction_id=case["Transaction"], model_name="xgboost",
                model_version="paysim-baseline-v1", fraud_probability=float(case["ML score"]) / 100,
                risk_score=float(case["Risk"]), investigator=record.assigned_investigator,
                final_outcome="FALSE_POSITIVE", feature_snapshot=feature_snapshot_for_case(case), notes=record.notes,
            )
            repository.transition(selected, CaseStatus.FALSE_POSITIVE)
            st.rerun()
    elif record.status in {CaseStatus.CONFIRMED_FRAUD, CaseStatus.FALSE_POSITIVE}:
        st.success(f"Disposition recorded: {record.decision}")
        if st.button("Close case", type="primary", use_container_width=True):
            repository.transition(selected, CaseStatus.CLOSED)
            st.rerun()
    else:
        st.success("Case is closed. The decision and notes are retained for audit.")

    st.markdown("### Investigation notes")
    notes = st.text_area(
        "Notes", value=record.notes, height=220, key=f"persistent_notes_{selected}_{record.updated_at}",
        placeholder="Document evidence, customer contact, rationale, and next steps…",
    )
    if st.button("Save notes", key=f"save_notes_{selected}", use_container_width=True):
        repository.update_notes(selected, notes)
        st.success("Notes saved persistently.")

    with st.expander("Valid state transitions"):
        next_states = [state.value for state in ALLOWED_TRANSITIONS[record.status]]
        st.write(" → ".join(next_states) if next_states else "No transitions; case is terminal.")
        st.caption(f"Created {record.created_at} · Updated {record.updated_at}")

    st.caption(f"Reviewed outcomes available for retraining: {len(feedback_repository.list())}")
