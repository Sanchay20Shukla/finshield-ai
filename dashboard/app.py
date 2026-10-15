"""FinShield multipage investigator dashboard."""

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.pages import alerts, investigation, monitoring, overview, transaction, transaction_360


st.set_page_config(
    page_title="FinShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #07111f; color: #e8eef7; }
    [data-testid="stSidebar"] { background: #0b1728; border-right: 1px solid #203149; }
    [data-testid="stMetric"] { background: #0d1b2d; border: 1px solid #223651; padding: 14px; border-radius: 10px; }
    h1, h2, h3 { color: #f3f7fc; }
    .finshield-kicker { color: #4ad6b8; font-size: .78rem; letter-spacing: .14em; font-weight: 700; }
    .finshield-note { color: #8ea3bd; font-size: .82rem; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🛡️ FinShield AI")
st.sidebar.caption("Fraud operations command center")
page = st.sidebar.radio(
    "Workspace",
    ["Executive dashboard", "Alert queue", "Transaction 360°", "Score transaction", "Investigations", "Model monitoring"],
    label_visibility="collapsed",
)
st.sidebar.divider()
from api.service import get_scoring_service
st.sidebar.caption(f"Model: {get_scoring_service().model.pipeline.named_steps['model'].__class__.__name__} · threshold {get_scoring_service().model.threshold:.2f}")
st.sidebar.caption("Data: PaySim · synthetic")
st.sidebar.info("Post-transaction portfolio demo. Decisions are simulated; no funds are blocked.")

{
    "Executive dashboard": overview.render,
    "Alert queue": alerts.render,
    "Transaction 360°": transaction_360.render,
    "Score transaction": transaction.render,
    "Investigations": investigation.render,
    "Model monitoring": monitoring.render,
}[page]()
