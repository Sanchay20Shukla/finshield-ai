"""Interactive transaction scoring page."""

import streamlit as st

from api.schemas import TransactionScoreRequest
from api.service import get_scoring_service


def render() -> None:
    st.markdown('<div class="finshield-kicker">LIVE DECISIONING</div>', unsafe_allow_html=True)
    st.title("Score a transaction")
    with st.form("score_transaction"):
        left, right = st.columns(2)
        tx_type = left.selectbox("Transaction type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = left.number_input("Amount", min_value=0.0, value=250000.0)
        old_origin = left.number_input("Origin balance before", min_value=0.0, value=260000.0)
        new_origin = left.number_input("Origin balance after", min_value=0.0, value=10000.0)
        old_dest = right.number_input("Destination balance before", min_value=0.0, value=20000.0)
        new_dest = right.number_input("Destination balance after", min_value=0.0, value=270000.0)
        velocity = right.number_input("Transaction velocity", min_value=1, value=1)
        submitted = st.form_submit_button("Score transaction", use_container_width=True)
    if submitted:
        with st.spinner("Calculating model, rule, and behavioral risk…"):
            request = TransactionScoreRequest(type=tx_type, amount=amount, oldbalanceOrg=old_origin, newbalanceOrig=new_origin, oldbalanceDest=old_dest, newbalanceDest=new_dest, transaction_velocity=velocity)
            result = get_scoring_service().score(request)
        a, b, c = st.columns(3)
        a.metric("Final risk score", f"{result.risk_score:.1f}")
        b.metric("Fraud probability", f"{result.fraud_probability:.1%}")
        c.metric("Decision", result.decision)
        st.markdown("#### Primary risk factors")
        for factor in result.risk_factors:
            st.write(f"• {factor}")
        with st.expander("Rule and score evidence"):
            st.json(result.model_dump())
