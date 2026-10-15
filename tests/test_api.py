"""Fraud scoring API contract tests."""

from fastapi.testclient import TestClient

from api.main import app
from api.schemas import TransactionScoreResponse
from api.service import get_scoring_service
import pytest


pytestmark = pytest.mark.api


class FakeScoringService:
    def score(self, request):
        return TransactionScoreResponse(
            transaction_id="TX-82912", risk_score=91, fraud_probability=0.94,
            decision="BLOCK", triggered_rules=["LARGE_TRANSFER", "ORIGIN_BALANCE_DEPLETED"],
            risk_factors=["unusually high transaction amount", "complete origin balance depletion"],
            rule_details=[], score_breakdown={"ml_score": 94, "rule_score": 75, "behavioral_score": 80},
            model_threshold=0.34, explanation_method="SHAP",
        )


def test_score_endpoint_contract():
    app.dependency_overrides[get_scoring_service] = lambda: FakeScoringService()
    client = TestClient(app)
    response = client.post("/api/v1/transactions/score", json={
        "type": "TRANSFER", "amount": 250000,
        "oldbalanceOrg": 260000, "newbalanceOrig": 10000,
        "oldbalanceDest": 20000, "newbalanceDest": 270000,
    })
    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "TX-82912"
    assert body["decision"] == "BLOCK"
    assert body["fraud_probability"] == 0.94
    assert body["triggered_rules"][0] == "LARGE_TRANSFER"
    assert body["explanation_method"] == "SHAP"


def test_score_endpoint_rejects_invalid_payload():
    client = TestClient(app)
    response = client.post("/api/v1/transactions/score", json={
        "type": "TRANSFER", "amount": -1,
        "oldbalanceOrg": 10, "newbalanceOrig": 11,
        "oldbalanceDest": 0, "newbalanceDest": 0, "unexpected": True,
    })
    assert response.status_code == 422


def test_health_endpoint():
    assert TestClient(app).get("/health").json() == {"status": "healthy"}
