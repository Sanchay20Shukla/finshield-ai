"""End-to-end component integration tests without external infrastructure."""

import pytest

from src.data.preprocessing import transform_transactions
from src.data.validation import validate_batch
from src.features.fraud_features import FEATURE_COLUMNS, FraudFeatureEngineer
from src.investigation.case_management import CaseRepository, CaseStatus
from src.models.feedback import FeedbackRepository
from src.rules.risk_scoring import UnifiedRiskScoringEngine


pytestmark = pytest.mark.integration


def test_transaction_to_risk_decision_pipeline(valid_paysim_frame):
    validated = validate_batch(valid_paysim_frame).valid
    canonical = transform_transactions(validated)
    featured = FraudFeatureEngineer().transform(canonical).iloc[0].to_dict()
    decision = UnifiedRiskScoringEngine().score(featured, ml_probability=0.94)
    assert 0 <= decision.final_risk_score <= 100
    assert decision.decision in {"APPROVE", "MONITOR", "MANUAL_REVIEW", "BLOCK"}
    assert {hit["rule"] for hit in decision.rule_hits} >= {"LARGE_TRANSFER", "FULL_BALANCE_TRANSFER"}


def test_investigator_disposition_becomes_training_feedback(tmp_path, valid_paysim_frame):
    featured = FraudFeatureEngineer().transform(valid_paysim_frame).iloc[0].to_dict()
    cases = CaseRepository(tmp_path / "cases.sqlite3")
    feedback = FeedbackRepository(tmp_path / "feedback.sqlite3")
    cases.create("C500", "TX500", 91)
    cases.assign("C500", "Integration Analyst")
    cases.transition("C500", CaseStatus.INVESTIGATING)
    feedback.record(
        case_id="C500", transaction_id="TX500", model_name="random_forest",
        model_version="test-v1", fraud_probability=.94, risk_score=91,
        investigator="Integration Analyst", final_outcome="CONFIRMED_FRAUD",
        feature_snapshot={column: featured[column] for column in FEATURE_COLUMNS},
    )
    cases.transition("C500", CaseStatus.CONFIRMED_FRAUD)
    training = feedback.training_frame()
    assert cases.get("C500").decision == "CONFIRMED_FRAUD"
    assert training.loc[0, "is_fraud"] == 1
    assert set(FEATURE_COLUMNS).issubset(training.columns)
