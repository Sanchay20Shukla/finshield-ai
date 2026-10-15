"""Real-artifact checks for an interviewer cloning this repository."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import TransactionScoreRequest
from api.service import FraudScoringService, get_scoring_service
from src.features.fraud_features import FEATURE_COLUMNS, FraudFeatureEngineer
from src.investigation.risk_explanation import ShapRiskExplainer

ROOT = Path(__file__).resolve().parents[1]


def test_real_model_api_prediction_and_explanation(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    row = pd.read_csv(ROOT / "data/sample/paysim_demo.csv").query("isFraud == 1").iloc[0]
    payload = {key: row[key] for key in ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]}
    response = TestClient(app).post("/api/v1/transactions/score", json=payload)
    assert response.status_code == 200
    assert 0 <= response.json()["fraud_probability"] <= 1
    assert response.json()["explanation_method"] == "SHAP"
    assert response.json()["risk_factors"]


def test_export_matches_real_artifact_and_shap_additivity():
    import xgboost as xgb
    explainer = ShapRiskExplainer(ROOT / "models/xgboost.joblib")
    cases = json.loads((ROOT / "web/data/cases.json").read_text())
    for case in cases[::12]:
        featured = FraudFeatureEngineer().transform(pd.DataFrame([case["transaction"]]))
        transformed = explainer.preprocessor.transform(featured[FEATURE_COLUMNS])
        matrix = xgb.DMatrix(transformed)
        booster = explainer.estimator.get_booster()
        contributions = booster.predict(matrix, pred_contribs=True)
        margin = booster.predict(matrix, output_margin=True)
        np.testing.assert_allclose(contributions.sum(axis=1), margin, atol=2e-5)
        assert abs(float(booster.predict(matrix)[0]) - case["probability"]) < 1e-6


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_api_rejects_nonfinite_numbers(value):
    with pytest.raises(ValueError):
        TransactionScoreRequest(type="TRANSFER", amount=value, oldbalanceOrg=1, newbalanceOrig=0, oldbalanceDest=0, newbalanceDest=1)
