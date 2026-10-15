"""Persistent API and prediction telemetry plus drift calculations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftResult:
    metric: str
    training_value: float
    production_value: float
    drift_score: float
    status: str
    explanation: str


class MonitoringRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @classmethod
    def default(cls) -> "MonitoringRepository":
        root = Path(__file__).resolve().parents[2]
        return cls(root / "data" / "processed" / "monitoring.sqlite3")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS api_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL,
                method TEXT NOT NULL, endpoint TEXT NOT NULL, status_code INTEGER NOT NULL,
                latency_ms REAL NOT NULL, error_type TEXT
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS prediction_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL,
                transaction_id TEXT NOT NULL, transaction_type TEXT NOT NULL,
                amount REAL NOT NULL, fraud_probability REAL NOT NULL,
                risk_score REAL NOT NULL, decision TEXT NOT NULL
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_api_time ON api_events(occurred_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_prediction_time ON prediction_events(occurred_at)")

    def record_api(self, method: str, endpoint: str, status_code: int, latency_ms: float, error_type: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO api_events (occurred_at, method, endpoint, status_code, latency_ms, error_type) VALUES (?, ?, ?, ?, ?, ?)",
                (_now(), method, endpoint, status_code, latency_ms, error_type),
            )

    def record_prediction(
        self, transaction_id: str, transaction_type: str, amount: float,
        fraud_probability: float, risk_score: float, decision: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO prediction_events (
                    occurred_at, transaction_id, transaction_type, amount,
                    fraud_probability, risk_score, decision
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (_now(), transaction_id, transaction_type, amount, fraud_probability, risk_score, decision),
            )

    def api_frame(self) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query("SELECT * FROM api_events ORDER BY occurred_at", connection)

    def prediction_frame(self) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query("SELECT * FROM prediction_events ORDER BY occurred_at", connection)


def numeric_drift(metric: str, training_mean: float, training_std: float, production_values) -> DriftResult:
    values = np.asarray(production_values, dtype=float)
    if not len(values):
        return DriftResult(metric, training_mean, float("nan"), 0.0, "INSUFFICIENT_DATA", "No production observations are available.")
    production_mean = float(values.mean())
    score = abs(production_mean - training_mean) / max(training_std, 1e-9)
    status = "DATA DRIFT WARNING" if score >= 0.5 else "STABLE"
    return DriftResult(
        metric, training_mean, production_mean, score, status,
        f"Production mean is {score:.2f} training standard deviations from baseline.",
    )


def categorical_psi(training_proportions: dict[str, float], production_values) -> DriftResult:
    values = pd.Series(production_values, dtype="string")
    if values.empty:
        return DriftResult("transaction_type_psi", 0.0, float("nan"), 0.0, "INSUFFICIENT_DATA", "No production observations are available.")
    production = values.value_counts(normalize=True).to_dict()
    categories = set(training_proportions) | set(production)
    epsilon = 1e-6
    psi = sum(
        (max(production.get(category, 0), epsilon) - max(training_proportions.get(category, 0), epsilon))
        * np.log(max(production.get(category, 0), epsilon) / max(training_proportions.get(category, 0), epsilon))
        for category in categories
    )
    status = "DATA DRIFT WARNING" if psi >= 0.2 else "STABLE"
    return DriftResult("transaction_type_psi", 0.0, float(psi), float(psi), status, "PSI >= 0.20 indicates material transaction-type mix drift.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
