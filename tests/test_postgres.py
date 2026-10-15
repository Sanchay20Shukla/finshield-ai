"""Real PostgreSQL audit round trip; exercised by the CI service container."""
import os
import uuid

import pytest


@pytest.mark.integration
def test_postgres_prediction_roundtrip(monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not configured")
    import psycopg
    from api.audit import record_prediction
    from api.schemas import TransactionScoreRequest, TransactionScoreResponse
    monkeypatch.setenv("DATABASE_URL", url)
    key = "TEST-" + uuid.uuid4().hex
    request = TransactionScoreRequest(type="PAYMENT", amount=5, oldbalanceOrg=20, newbalanceOrig=15, oldbalanceDest=0, newbalanceDest=0)
    response = TransactionScoreResponse(transaction_id=key, fraud_probability=.1, risk_score=10, decision="APPROVE", triggered_rules=[], risk_factors=[], rule_details=[], score_breakdown={}, model_threshold=.19, explanation_method="SHAP")
    record_prediction(request, response, "test-artifact")
    with psycopg.connect(url) as connection:
        row = connection.execute("SELECT request, response, model_version FROM inference_audit WHERE transaction_id=%s", (key,)).fetchone()
        assert row[0]["amount"] == 5
        assert row[1]["decision"] == "APPROVE"
        assert row[2] == "test-artifact"
        connection.execute("DELETE FROM inference_audit WHERE transaction_id=%s", (key,))
