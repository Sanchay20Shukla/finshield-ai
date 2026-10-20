"""Platform checks use an isolated warehouse and case database, never analyst state."""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import platform
from src.data import warehouse


@pytest.fixture
def client(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "data/sample/paysim_demo.csv"
    data = pd.read_csv(source).sort_values("step", kind="stable").reset_index(drop=True)
    csv = tmp_path / "paysim.csv"
    data.to_csv(csv, index=False)
    target = tmp_path / "analytics.duckdb"
    warehouse.build(csv, target)
    monkeypatch.setattr(warehouse, "WAREHOUSE", target)
    monkeypatch.setattr(platform, "ROOT", tmp_path)
    platform.analytics_snapshot.cache_clear()
    yield TestClient(app), data
    platform.analytics_snapshot.cache_clear()


def test_overview_reconciles_to_source(client):
    api, frame = client
    d = api.get("/api/v1/platform/overview").json()
    s = d["summary"]
    assert s["transactions"] == len(frame)
    assert s["fraud_count"] == int(frame.isFraud.sum())
    assert sum(s[k] for k in ["tp", "fp", "fn", "tn"]) == len(frame)
    assert sum(row["transactions"] for row in d["daily"]) == len(frame)
    assert sum(row["fraud_count"] for row in d["segments"]) == s["fraud_count"]


def test_search_pagination_and_input_validation(client):
    api, frame = client
    first = api.get("/api/v1/platform/transactions?limit=5&page=1").json()
    second = api.get("/api/v1/platform/transactions?limit=5&page=2").json()
    assert not {r["id"] for r in first["rows"]} & {r["id"] for r in second["rows"]}
    assert api.get("/api/v1/platform/transactions?search=%27%20OR%201%3D1--").json()["total"] == 0
    assert api.get("/api/v1/platform/transactions?type=INVALID").status_code == 422
    assert api.get("/api/v1/platform/transactions?page=0").status_code == 422
    assert api.get("/api/v1/platform/transactions/99999999").status_code == 404


def test_prior_history_excludes_current_and_future(client):
    _, frame = client
    for index in [0, 50, len(frame)-1]:
        row, history = platform.transaction(index+1)
        previous = frame.iloc[:index]
        origin = previous[previous.nameOrig == row["nameOrig"]]
        assert history["account_transaction_count"] == len(origin)
        assert history["transaction_velocity"] == 1 + int((origin.step == row["step"]).sum())
        assert history["destination_frequency"] == int((previous.nameDest == row["nameDest"]).sum())


def test_persistent_review_preserves_audit_history(client):
    api, _ = client
    for status in ["INVESTIGATING", "CONFIRMED_FRAUD"]:
        response = api.put("/api/v1/platform/cases/1", json={"status": status, "analyst": "Test analyst", "notes": "Verified test evidence"})
        assert response.status_code == 200
    cases = api.get("/api/v1/platform/cases").json()
    assert len(cases) == 1 and cases[0]["status"] == "CONFIRMED_FRAUD"
    assert len(api.get("/api/v1/platform/cases/1/events").json()) == 2
    assert api.put("/api/v1/platform/cases/1",json={"status":"UNKNOWN","analyst":"x"}).status_code == 422


def test_quality_failure_preserves_published_warehouse(client, tmp_path):
    _, frame = client
    frame.loc[0, "amount"] = -1
    invalid = tmp_path / "invalid.csv"
    frame.to_csv(invalid,index=False)
    with pytest.raises(ValueError, match="Data quality"):
        warehouse.build(invalid, warehouse.WAREHOUSE)
    with warehouse.connect() as db:
        assert db.execute("SELECT count(*) FROM fact_transactions").fetchone()[0] == len(frame)


def test_chronology_failure(client, tmp_path):
    _, frame = client
    backwards = tmp_path / "backwards.csv"
    frame.sort_values("step",ascending=False).to_csv(backwards,index=False)
    with pytest.raises(ValueError, match="chronological"):
        warehouse.build(backwards,tmp_path / "bad.duckdb")


def test_frontend_is_served_on_api_origin(client):
    api, _ = client
    response = api.get("/")
    assert response.status_code == 200 and "FinShield" in response.text
    assert api.get("/platform/app.mjs").status_code == 200


def test_summary_uses_model_evidence_without_label_leakage(client, monkeypatch):
    api, _ = client
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    response = api.post("/api/v1/platform/transactions/1/summary", json={"question":"Explain why this transaction was flagged"})
    assert response.status_code == 200
    packet = response.json()
    assert packet["provider"] == "grounded-local"
    assert {"TX", "RISK", "SHAP", "HISTORY", "RULES", "ACTION"} == {e["evidence_id"] for e in packet["evidence"]}
    assert "isFraud" not in json.dumps(packet["evidence"])
    assert "is_fraud" not in json.dumps(packet["evidence"])
    assert "[SHAP]" in packet["answer"]
