"""Persistent case management with an enforced fraud-investigation state machine."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class CaseStatus(str, Enum):
    NEW = "NEW"
    ASSIGNED = "ASSIGNED"
    INVESTIGATING = "INVESTIGATING"
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    CLOSED = "CLOSED"


ALLOWED_TRANSITIONS = {
    CaseStatus.NEW: {CaseStatus.ASSIGNED},
    CaseStatus.ASSIGNED: {CaseStatus.INVESTIGATING},
    CaseStatus.INVESTIGATING: {CaseStatus.CONFIRMED_FRAUD, CaseStatus.FALSE_POSITIVE},
    CaseStatus.CONFIRMED_FRAUD: {CaseStatus.CLOSED},
    CaseStatus.FALSE_POSITIVE: {CaseStatus.CLOSED},
    CaseStatus.CLOSED: set(),
}


class InvalidCaseTransition(ValueError):
    pass


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    transaction_id: str
    risk_score: float
    assigned_investigator: str | None
    status: CaseStatus
    notes: str
    decision: str | None
    created_at: str
    updated_at: str

    def to_dict(self):
        value = asdict(self)
        value["status"] = self.status.value
        return value


class CaseRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @classmethod
    def default(cls) -> "CaseRepository":
        root = Path(__file__).resolve().parents[2]
        return cls(root / "data" / "processed" / "cases.sqlite3")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL UNIQUE,
                risk_score REAL NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
                assigned_investigator TEXT,
                status TEXT NOT NULL CHECK (status IN ('NEW','ASSIGNED','INVESTIGATING','CONFIRMED_FRAUD','FALSE_POSITIVE','CLOSED')),
                notes TEXT NOT NULL DEFAULT '',
                decision TEXT CHECK (decision IN ('CONFIRMED_FRAUD','FALSE_POSITIVE')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")

    def create(self, case_id: str, transaction_id: str, risk_score: float) -> CaseRecord:
        if not case_id.strip() or not transaction_id.strip():
            raise ValueError("case_id and transaction_id are required")
        if not 0 <= risk_score <= 100:
            raise ValueError("risk_score must be between 0 and 100")
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO cases (case_id, transaction_id, risk_score, status, created_at, updated_at)
                   VALUES (?, ?, ?, 'NEW', ?, ?) ON CONFLICT(case_id) DO NOTHING""",
                (case_id, transaction_id, risk_score, now, now),
            )
        return self.get(case_id)

    def get(self, case_id: str) -> CaseRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise KeyError(f"case not found: {case_id}")
        return _record(row)

    def list(self, status: CaseStatus | None = None) -> list[CaseRecord]:
        query, params = "SELECT * FROM cases", ()
        if status:
            query, params = query + " WHERE status = ?", (status.value,)
        query += " ORDER BY risk_score DESC, created_at ASC"
        with self._connect() as connection:
            return [_record(row) for row in connection.execute(query, params).fetchall()]

    def assign(self, case_id: str, investigator: str) -> CaseRecord:
        if not investigator.strip():
            raise ValueError("investigator is required")
        case = self.get(case_id)
        if case.status is not CaseStatus.NEW:
            raise InvalidCaseTransition(f"only NEW cases can be assigned; {case_id} is {case.status.value}")
        with self._connect() as connection:
            connection.execute(
                "UPDATE cases SET assigned_investigator=?, status='ASSIGNED', updated_at=? WHERE case_id=?",
                (investigator.strip(), _now(), case_id),
            )
        return self.get(case_id)

    def transition(self, case_id: str, target: CaseStatus) -> CaseRecord:
        case = self.get(case_id)
        if target not in ALLOWED_TRANSITIONS[case.status]:
            raise InvalidCaseTransition(f"invalid transition: {case.status.value} -> {target.value}")
        if target is CaseStatus.INVESTIGATING and not case.assigned_investigator:
            raise InvalidCaseTransition("case must have an assigned investigator")
        decision = case.decision
        if target in {CaseStatus.CONFIRMED_FRAUD, CaseStatus.FALSE_POSITIVE}:
            decision = target.value
        if target is CaseStatus.CLOSED and not decision:
            raise InvalidCaseTransition("case requires a fraud or false-positive decision before closing")
        with self._connect() as connection:
            connection.execute(
                "UPDATE cases SET status=?, decision=?, updated_at=? WHERE case_id=?",
                (target.value, decision, _now(), case_id),
            )
        return self.get(case_id)

    def update_notes(self, case_id: str, notes: str) -> CaseRecord:
        self.get(case_id)
        with self._connect() as connection:
            connection.execute("UPDATE cases SET notes=?, updated_at=? WHERE case_id=?", (notes, _now(), case_id))
        return self.get(case_id)


def _record(row: sqlite3.Row) -> CaseRecord:
    return CaseRecord(
        case_id=row["case_id"], transaction_id=row["transaction_id"], risk_score=float(row["risk_score"]),
        assigned_investigator=row["assigned_investigator"], status=CaseStatus(row["status"]),
        notes=row["notes"], decision=row["decision"], created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
