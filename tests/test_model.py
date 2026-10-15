"""Fraud model evaluation and business-cost tests."""

import numpy as np
import joblib
import pandas as pd
import pytest

from src.models.evaluate import evaluate_probabilities, optimize_threshold
from src.investigation.risk_explanation import _factor, _fraud_class_values
from src.rag.copilot import FraudInvestigationCopilot
from src.rag.retriever import EvidenceDocument, EvidenceRetriever
from src.rag.knowledge_base import chunk_document
from src.investigation.report_generator import InvestigationReportDraft, save_report
from src.models.feedback import FeedbackRepository
from src.features.fraud_features import FEATURE_COLUMNS
from src.monitoring.telemetry import MonitoringRepository, categorical_psi, numeric_drift
from src.models.predict import FraudModel
from src.models.train import build_models


pytestmark = pytest.mark.model


def test_metrics_include_weighted_confusion_and_expected_loss():
    result = evaluate_probabilities(
        [0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], 0.5,
        fraud_cost=10_000, review_cost=50,
    )
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.expected_loss == 10_050
    assert result.false_positive_rate == 0.5
    assert result.false_negative_rate == 0.5


def test_cost_optimization_can_prefer_recall_over_default_threshold():
    y = np.array([0, 0, 0, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    result = optimize_threshold(
        y, probabilities, fraud_cost=10_000, review_cost=50,
        thresholds=[0.25, 0.5],
    )
    assert result.threshold == 0.25
    assert result.recall == 1.0
    assert result.expected_loss == 50


def test_shap_output_selects_positive_fraud_class():
    values = np.arange(12).reshape(1, 6, 2)
    assert _fraud_class_values(values).tolist() == [1, 3, 5, 7, 9, 11]


def test_technical_features_translate_to_investigator_language():
    factor = _factor(
        "numeric__zero_balance_after_transaction", 0.18,
        {"zero_balance_after_transaction": 1},
    )
    assert factor.description == "complete origin balance depletion"
    assert factor.direction == "INCREASES_RISK"
    assert factor.value == 1


def test_copilot_fallback_uses_only_retrieved_evidence(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    documents = [
        EvidenceDocument("E-TX", "transaction", "Transaction TX10291 is a TRANSFER for INR 84,000."),
        EvidenceDocument("E-RISK", "risk", "ML fraud probability is 94%."),
        EvidenceDocument("E-RULE-1", "rules", "Triggered rule FULL_BALANCE_TRANSFER."),
        EvidenceDocument("E-RECOMMEND", "recommendation", "Recommended action: MANUAL REVIEW."),
    ]
    result = FraudInvestigationCopilot().answer("Why was TX10291 flagged?", documents)
    assert result.provider == "grounded-local"
    assert "94%" in result.answer
    assert "FULL_BALANCE_TRANSFER" in result.answer
    assert "MANUAL REVIEW" in result.answer
    assert set(result.evidence_ids).issubset({document.evidence_id for document in documents})


def test_retriever_rejects_invalid_limit():
    try:
        EvidenceRetriever().retrieve("why", [], top_k=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid retrieval limit was accepted")


def test_markdown_policy_is_chunked_with_source_identity(tmp_path):
    policy = tmp_path / "policy.md"
    policy.write_text("# Escalation\n\nScores above 80 require immediate escalation. " * 30, encoding="utf-8")
    chunks = chunk_document(policy, chunk_size=180, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.source == "policy.md" for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_report_save_requires_named_analyst_and_writes_audit_metadata(tmp_path):
    draft = InvestigationReportDraft("FIN-2026-00192", "TX10291", "CASE: FIN-2026-00192", "2026-01-01T00:00:00+00:00", "test", ["E-TX"])
    try:
        save_report(draft, draft.content, tmp_path, "Pending")
    except ValueError as error:
        assert "analyst name" in str(error)
    else:
        raise AssertionError("unnamed analyst report was saved")
    path = save_report(draft, draft.content + "\nReviewed", tmp_path, "A. Investigator")
    assert path.is_file()
    metadata = path.with_suffix(".json").read_text(encoding="utf-8")
    assert "content_sha256" in metadata
    assert "A. Investigator" in metadata


def test_feedback_round_trip_exports_training_labels(tmp_path):
    repository = FeedbackRepository(tmp_path / "feedback.sqlite3")
    features = {column: 0.0 for column in FEATURE_COLUMNS}
    features["transaction_type"] = "TRANSFER"
    record = repository.record(
        case_id="C102", transaction_id="TX901", model_name="random_forest", model_version="baseline",
        fraud_probability=0.94, risk_score=91, investigator="A. Investigator",
        final_outcome="FALSE_POSITIVE", feature_snapshot=features, notes="Customer verified transfer.",
    )
    assert record.final_label == 0
    training = repository.training_frame()
    assert training.loc[0, "is_fraud"] == 0
    assert training.loc[0, "feedback_case_id"] == "C102"


def test_monitoring_records_api_and_prediction_events(tmp_path):
    repository = MonitoringRepository(tmp_path / "monitoring.sqlite3")
    repository.record_api("POST", "/api/v1/transactions/score", 200, 12.5)
    repository.record_prediction("TX1", "TRANSFER", 250000, .9, 87, "BLOCK")
    assert len(repository.api_frame()) == 1
    assert repository.prediction_frame().loc[0, "decision"] == "BLOCK"


def test_drift_warning_thresholds():
    amount = numeric_drift("amount", 100, 20, [150, 160])
    assert amount.status == "DATA DRIFT WARNING"
    types = categorical_psi({"TRANSFER": .5, "PAYMENT": .5}, ["TRANSFER"] * 20)
    assert types.status == "DATA DRIFT WARNING"


def test_persisted_model_inference_returns_bounded_probability(tmp_path):
    rows = []
    for index in range(8):
        row = {column: float(index % 2) for column in FEATURE_COLUMNS if column != "transaction_type"}
        row["amount"] = 1000.0 + index * 100_000
        row["transaction_type"] = "TRANSFER" if index % 2 else "PAYMENT"
        rows.append(row)
    features = pd.DataFrame(rows)
    labels = pd.Series([index % 2 for index in range(8)])
    pipeline = build_models(labels)["logistic_regression"]
    pipeline.fit(features, labels)
    artifact = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipeline, "threshold": .5, "features": FEATURE_COLUMNS}, artifact)
    result = FraudModel(artifact).predict(features.iloc[1].to_dict())
    assert 0 <= result["fraud_probability"] <= 1
    assert isinstance(result["predicted_fraud"], bool)
