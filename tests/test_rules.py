"""Phase 1 data validation tests."""

import pandas as pd
import pytest

from src.data.validation import SchemaValidationError, validate_batch
from src.rules.fraud_rules import FraudDecisionEngine, FraudRulesEngine, RuleConfig
from src.rules.risk_scoring import RiskScoreConfig, UnifiedRiskScoringEngine
from src.investigation.case_management import CaseRepository, CaseStatus, InvalidCaseTransition


pytestmark = pytest.mark.unit


def valid_row():
    return {
        "step": 1, "type": "TRANSFER", "amount": 100.0, "nameOrig": "C1",
        "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0, "nameDest": "C2",
        "oldbalanceDest": 0.0, "newbalanceDest": 100.0, "isFraud": 0,
        "isFlaggedFraud": 0,
    }


def test_validation_rejects_duplicates_and_business_violations():
    good = valid_row()
    negative = {**good, "nameOrig": "C3", "amount": -1}
    result = validate_batch(pd.DataFrame([good, good, negative]))
    assert len(result.valid) == 1
    assert len(result.rejected) == 2
    assert result.issue_counts["duplicate_in_batch"] == 1
    assert result.issue_counts["negative_amount"] == 1


def test_validation_rejects_schema_drift():
    frame = pd.DataFrame([valid_row()]).drop(columns=["step"])
    try:
        validate_batch(frame)
    except SchemaValidationError as error:
        assert "missing=['step']" in str(error)
    else:
        raise AssertionError("schema drift was not rejected")


def suspicious_transaction():
    return {
        "transaction_type": "TRANSFER", "amount": 300_000.0,
        "origin_old_balance": 300_000.0, "origin_new_balance": 0.0,
        "transaction_velocity": 5, "destination_frequency": 4,
        "account_transaction_count": 10, "average_account_amount": 50_000.0,
        "full_balance_transfer": 1,
    }


def test_rules_return_stable_ids_scores_and_evidence():
    hits = FraudRulesEngine().evaluate(suspicious_transaction())
    assert [hit.rule_id for hit in hits] == [
        "RULE_001", "RULE_002", "RULE_003", "RULE_004", "RULE_005", "RULE_006"
    ]
    full_balance = hits[2].to_dict()
    assert full_balance["rule"] == "FULL_BALANCE_TRANSFER"
    assert full_balance["severity"] == "HIGH"
    assert full_balance["score"] == 25
    assert full_balance["evidence"]["amount"] == 300_000.0


def test_hybrid_decision_combines_model_and_rules():
    decision = FraudDecisionEngine().decide(suspicious_transaction(), model_probability=0.8)
    assert decision.risk_score == 86.0
    assert decision.rules_score == 100
    assert decision.severity.value == "CRITICAL"
    assert decision.action == "BLOCK_AND_REVIEW"


def test_rules_only_fallback_and_configurable_thresholds():
    engine = FraudDecisionEngine(FraudRulesEngine(RuleConfig(large_transfer_threshold=500_000)))
    decision = engine.decide({
        "transaction_type": "TRANSFER", "amount": 300_000,
        "origin_old_balance": 1_000_000, "origin_new_balance": 700_000,
    })
    assert decision.risk_score == 0
    assert decision.action == "APPROVE"
    assert decision.model_score is None


def test_unified_risk_score_exposes_all_three_components():
    result = UnifiedRiskScoringEngine().score(suspicious_transaction(), ml_probability=0.8)
    # ML=80, rules=100, behavior=82 => 52 + 20 + 12.3
    assert result.final_risk_score == 84.3
    assert result.decision == "BLOCK"
    assert result.ml_score == 80
    assert result.rule_score == 100
    assert result.behavioral_score == 82
    assert len(result.rule_hits) == 6
    assert len(result.behavioral_signals) == 4


def test_unified_decision_boundaries_are_exact():
    engine = UnifiedRiskScoringEngine(config=RiskScoreConfig(ml_weight=1, rule_weight=0, behavioral_weight=0))
    neutral = {}
    assert engine.score(neutral, 0.30).decision == "APPROVE"
    assert engine.score(neutral, 0.31).decision == "MONITOR"
    assert engine.score(neutral, 0.61).decision == "MANUAL_REVIEW"
    assert engine.score(neutral, 0.81).decision == "BLOCK"


def test_risk_weights_must_sum_to_one():
    try:
        RiskScoreConfig(ml_weight=0.5, rule_weight=0.2, behavioral_weight=0.2)
    except ValueError as error:
        assert "sum to 1" in str(error)
    else:
        raise AssertionError("invalid weights were accepted")


def test_case_state_machine_and_persistence(tmp_path):
    repository = CaseRepository(tmp_path / "cases.sqlite3")
    case = repository.create("C102", "TX901", 96)
    assert case.status is CaseStatus.NEW
    case = repository.assign("C102", "A. Investigator")
    assert case.status is CaseStatus.ASSIGNED
    assert case.assigned_investigator == "A. Investigator"
    assert repository.transition("C102", CaseStatus.INVESTIGATING).status is CaseStatus.INVESTIGATING
    resolved = repository.transition("C102", CaseStatus.CONFIRMED_FRAUD)
    assert resolved.decision == "CONFIRMED_FRAUD"
    assert repository.transition("C102", CaseStatus.CLOSED).status is CaseStatus.CLOSED


def test_case_state_machine_rejects_skipped_steps(tmp_path):
    repository = CaseRepository(tmp_path / "cases.sqlite3")
    repository.create("C103", "TX902", 87)
    try:
        repository.transition("C103", CaseStatus.INVESTIGATING)
    except InvalidCaseTransition as error:
        assert "NEW -> INVESTIGATING" in str(error)
    else:
        raise AssertionError("case skipped assignment")
