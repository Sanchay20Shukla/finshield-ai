"""Investigator outcome feedback storage and training-data export."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.features.fraud_features import FEATURE_COLUMNS


VALID_OUTCOMES = {"CONFIRMED_FRAUD": 1, "FALSE_POSITIVE": 0}


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: int
    case_id: str
    transaction_id: str
    model_name: str
    model_version: str
    fraud_probability: float
    risk_score: float
    investigator: str
    final_outcome: str
    final_label: int
    feature_snapshot: dict[str, Any]
    notes: str
    created_at: str
    updated_at: str

    def to_dict(self):
        return asdict(self)


class FeedbackRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @classmethod
    def default(cls) -> "FeedbackRepository":
        root = Path(__file__).resolve().parents[2]
        return cls(root / "data" / "processed" / "feedback.sqlite3")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS investigator_feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL UNIQUE,
                transaction_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                fraud_probability REAL NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
                risk_score REAL NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
                investigator TEXT NOT NULL,
                final_outcome TEXT NOT NULL CHECK (final_outcome IN ('CONFIRMED_FRAUD','FALSE_POSITIVE')),
                final_label INTEGER NOT NULL CHECK (final_label IN (0,1)),
                feature_snapshot TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")

    def record(
        self, *, case_id: str, transaction_id: str, model_name: str,
        model_version: str, fraud_probability: float, risk_score: float,
        investigator: str, final_outcome: str, feature_snapshot: Mapping[str, Any],
        notes: str = "",
    ) -> FeedbackRecord:
        if final_outcome not in VALID_OUTCOMES:
            raise ValueError(f"invalid final outcome: {final_outcome}")
        if not investigator.strip():
            raise ValueError("investigator is required")
        missing = sorted(set(FEATURE_COLUMNS) - set(feature_snapshot))
        if missing:
            raise ValueError(f"feedback feature snapshot is incomplete: {missing}")
        if not 0 <= fraud_probability <= 1 or not 0 <= risk_score <= 100:
            raise ValueError("prediction and risk scores are outside valid ranges")
        snapshot = {column: _json_value(feature_snapshot[column]) for column in FEATURE_COLUMNS}
        now = _now()
        with self._connect() as connection:
            connection.execute("""INSERT INTO investigator_feedback (
                case_id, transaction_id, model_name, model_version, fraud_probability,
                risk_score, investigator, final_outcome, final_label, feature_snapshot,
                notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
                investigator=excluded.investigator, final_outcome=excluded.final_outcome,
                final_label=excluded.final_label, feature_snapshot=excluded.feature_snapshot,
                notes=excluded.notes, updated_at=excluded.updated_at""", (
                case_id, transaction_id, model_name, model_version, fraud_probability,
                risk_score, investigator.strip(), final_outcome, VALID_OUTCOMES[final_outcome],
                json.dumps(snapshot), notes, now, now,
            ))
        return self.get(case_id)

    def get(self, case_id: str) -> FeedbackRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM investigator_feedback WHERE case_id=?", (case_id,)).fetchone()
        if row is None:
            raise KeyError(f"feedback not found: {case_id}")
        return _feedback_record(row)

    def list(self) -> list[FeedbackRecord]:
        with self._connect() as connection:
            return [_feedback_record(row) for row in connection.execute("SELECT * FROM investigator_feedback ORDER BY created_at").fetchall()]

    def training_frame(self) -> pd.DataFrame:
        records = self.list()
        rows = [{**record.feature_snapshot, "is_fraud": record.final_label, "feedback_case_id": record.case_id} for record in records]
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["is_fraud", "feedback_case_id"])


def _feedback_record(row: sqlite3.Row) -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=int(row["feedback_id"]), case_id=row["case_id"], transaction_id=row["transaction_id"],
        model_name=row["model_name"], model_version=row["model_version"],
        fraud_probability=float(row["fraud_probability"]), risk_score=float(row["risk_score"]),
        investigator=row["investigator"], final_outcome=row["final_outcome"], final_label=int(row["final_label"]),
        feature_snapshot=json.loads(row["feature_snapshot"]), notes=row["notes"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
