"""Dataset-backed analytics, evidence and persistent local investigations."""
from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.data.warehouse import ROOT, connect, records

router = APIRouter(prefix="/api/v1/platform", tags=["Platform"])


def warehouse():
    try:
        return connect()
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error


@lru_cache(maxsize=1)
def analytics_snapshot():
    with warehouse() as db:
        summary = records(db, """SELECT count(*) AS transactions, sum(amount) AS volume,
            sum(isFraud) AS fraud_count, sum(CASE WHEN isFraud=1 THEN amount ELSE 0 END) AS fraud_value,
            sum(rule_alert::INT) AS alerts, sum((rule_alert AND isFraud=1)::INT) AS tp,
            sum((rule_alert AND isFraud=0)::INT) AS fp,
            sum((NOT rule_alert AND isFraud=1)::INT) AS fn,
            sum((NOT rule_alert AND isFraud=0)::INT) AS tn,
            sum(CASE WHEN rule_alert AND isFraud=1 THEN amount ELSE 0 END) AS flagged_fraud_value,
            min(step) AS first_step, max(step) AS last_step FROM fact_transactions""")[0]
        summary["precision"] = summary["tp"] / max(summary["alerts"], 1)
        summary["recall"] = summary["tp"] / max(summary["fraud_count"], 1)
        summary["false_positive_rate"] = summary["fp"] / max(summary["fp"] + summary["tn"], 1)
        return {"summary": summary, "daily": records(db, "SELECT * FROM daily"),
                "segments": records(db, "SELECT * FROM segments"),
                "metadata": json.loads(db.execute("SELECT payload FROM metadata").fetchone()[0])}


@router.get("/overview")
def overview():
    return analytics_snapshot()


@router.get("/transactions")
def transactions(search: str = Query("", max_length=64),
                 type: Literal["ALL", "TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"] = "ALL",
                 risk: Literal["ALL", "HIGH", "MEDIUM", "LOW"] = "ALL",
                 page: int = Query(1, ge=1), limit: int = Query(25, ge=1, le=100),
                 order: Literal["recent", "amount", "earliest"] = "recent"):
    conditions, params = ["true"], []
    if search:
        conditions.append("(contains(nameOrig, ?) OR contains(nameDest, ?) OR CAST(id AS VARCHAR)=?)")
        params += [search, search, search.removeprefix("TX-")]
    for column, value in [("type", type), ("risk_band", risk)]:
        if value != "ALL":
            conditions.append(f"{column}=?")
            params.append(value)
    where = " AND ".join(conditions)
    ordering = {"recent": "id DESC", "earliest": "id ASC", "amount": "amount DESC, id DESC"}[order]
    with warehouse() as db:
        total = db.execute(f"SELECT count(*) FROM fact_transactions WHERE {where}", params).fetchone()[0]
        rows = records(db, f"SELECT * FROM fact_transactions WHERE {where} ORDER BY {ordering} LIMIT ? OFFSET ?",
                       params + [limit, (page - 1) * limit])
    return {"rows": rows, "total": total, "page": page, "limit": limit}


def transaction(transaction_id: int):
    with warehouse() as db:
        found = records(db, "SELECT * FROM fact_transactions WHERE id=?", [transaction_id])
        if not found:
            raise HTTPException(404, "Transaction not found")
        row = found[0]
        # Stable source order breaks ties inside the simulated hour. Current event excluded.
        history = records(db, """SELECT count(*) AS account_transaction_count,
            coalesce(avg(amount),0) AS average_account_amount,
            count(*) FILTER (WHERE step=?) + 1 AS transaction_velocity
            FROM fact_transactions WHERE nameOrig=? AND id<?""", [row["step"], row["nameOrig"], row["id"]])[0]
        history["destination_frequency"] = db.execute(
            "SELECT count(*) FROM fact_transactions WHERE nameDest=? AND id<?", [row["nameDest"], row["id"]]).fetchone()[0]
    return row, history


@router.get("/transactions/{transaction_id}")
def detail(transaction_id: int):
    from api.schemas import TransactionScoreRequest
    from api.service import get_scoring_service
    row, history = transaction(transaction_id)
    fields = ["step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    response = get_scoring_service().score(TransactionScoreRequest(**{key: row[key] for key in fields}, **history))
    return {"transaction": row, "history": history, "analysis": response.model_dump()}


def cases_db():
    path = ROOT / "data/processed/platform_cases.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS cases (transaction_id INTEGER PRIMARY KEY,
        status TEXT NOT NULL, analyst TEXT NOT NULL, notes TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))""")
    db.execute("""CREATE TABLE IF NOT EXISTS case_events (event_id INTEGER PRIMARY KEY,
        transaction_id INTEGER, status TEXT, analyst TEXT, notes TEXT,
        occurred_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))""")
    return db


class Review(BaseModel):
    status: Literal["OPEN", "INVESTIGATING", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"]
    analyst: str = Field(min_length=1, max_length=80)
    notes: str = Field(default="", max_length=8000)


@router.get("/cases")
def cases():
    with cases_db() as db:
        return [dict(row) for row in db.execute("SELECT * FROM cases ORDER BY updated_at DESC")]


@router.get("/cases/{transaction_id}/events")
def case_events(transaction_id: int):
    with cases_db() as db:
        return [dict(row) for row in db.execute("SELECT * FROM case_events WHERE transaction_id=? ORDER BY event_id DESC", [transaction_id])]


@router.put("/cases/{transaction_id}")
def save_case(transaction_id: int, review: Review):
    transaction(transaction_id)
    with cases_db() as db:
        values = [transaction_id, review.status, review.analyst, review.notes]
        db.execute("""INSERT INTO cases(transaction_id,status,analyst,notes) VALUES (?,?,?,?)
            ON CONFLICT(transaction_id) DO UPDATE SET status=excluded.status, analyst=excluded.analyst,
            notes=excluded.notes, updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')""", values)
        db.execute("INSERT INTO case_events(transaction_id,status,analyst,notes) VALUES (?,?,?,?)", values)
    return {"saved": True, "transaction_id": transaction_id}


class Question(BaseModel):
    question: str = Field(default="Summarize the transaction evidence and investigation priorities", min_length=1, max_length=1000)


@router.post("/transactions/{transaction_id}/summary")
def summarize(transaction_id: int, question: Question):
    from src.rag.copilot import FraudInvestigationCopilot
    from src.rag.retriever import EvidenceDocument
    packet = detail(transaction_id)
    row, analysis = packet["transaction"], packet["analysis"]
    documents = [EvidenceDocument("TX", "transaction", f"TX-{transaction_id}: {row['type']}, amount {row['amount']} synthetic units; simulation hour {row['step']}."),
                 EvidenceDocument("RISK", "risk", f"Model output {analysis['fraud_probability']}; uncalibrated. Policy risk {analysis['risk_score']}/100. Decision {analysis['decision']}."),
                 EvidenceDocument("HISTORY", "history", f"Before this event, the origin had {packet['history']['account_transaction_count']} transactions averaging {packet['history']['average_account_amount']:.2f} units. Same-hour velocity including this event: {packet['history']['transaction_velocity']}. Destination prior events: {packet['history']['destination_frequency']}."),
                 EvidenceDocument("RULES", "rules", "Triggered rules: " + ("; ".join(f"{r['rule']}: {r['reason']}" for r in analysis["rule_details"]) or "none")),
                 EvidenceDocument("SHAP", "model_explanation", "Model influences in log-odds: " + "; ".join(f"{f['feature']} {f['contribution']:+.3f} (value {f['value']})" for f in analysis["shap_factors"])),
                 EvidenceDocument("ACTION", "recommendation", "Review balances, account history, rule evidence and SHAP. Confirm independently before taking action. This is retrospective synthetic data.")]
    answer = FraudInvestigationCopilot().answer(question.question, documents).to_dict()
    return {**answer, "evidence": [document.__dict__ for document in documents]}


@router.get("/monitoring")
def monitoring():
    with warehouse() as db:
        # Fixed amount buckets avoid fitting boundaries to the comparison window.
        rows = records(db, """SELECT CASE WHEN step <= (SELECT quantile_disc(step,.70) FROM fact_transactions)
            THEN 'reference' ELSE 'recent' END AS window,
            CASE WHEN amount<1000 THEN 0 WHEN amount<10000 THEN 1 WHEN amount<100000 THEN 2
                 WHEN amount<1000000 THEN 3 ELSE 4 END AS bucket, count(*) AS n
            FROM fact_transactions GROUP BY 1,2""")
    counts = {window: np.full(5, .5) for window in ["reference", "recent"]}
    for row in rows:
        counts[row["window"]][row["bucket"]] += row["n"]
    ref, recent = [counts[key] / counts[key].sum() for key in ["reference", "recent"]]
    psi = float(np.sum((recent - ref) * np.log(recent / ref)))
    comparison = json.loads((ROOT / "models/model_comparison.json").read_text())
    reproduced_path = ROOT / "models/reproduced/benchmark.json"
    reproduced = json.loads(reproduced_path.read_text()) if reproduced_path.is_file() else None
    return {"reproduced_benchmark": reproduced, "amount_psi": psi, "buckets": ["<1K", "1K–10K", "10K–100K", "100K–1M", "≥1M"],
            "reference": ref.tolist(), "recent": recent.tolist(), "historical_models": comparison,
            "scope": "Amount distribution: first 70% by simulation hour versus remaining 30%. Historical model metrics are inherited, not a new holdout evaluation."}
