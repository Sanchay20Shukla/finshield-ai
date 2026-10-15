"""Schema, quality, type, and business validation for PaySim batches."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


REQUIRED_COLUMNS = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
)
ALLOWED_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
NUMERIC_COLUMNS = (
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
)


class SchemaValidationError(ValueError):
    """Raised when a batch cannot safely enter the pipeline."""


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    rejected: pd.DataFrame
    issue_counts: dict[str, int] = field(default_factory=dict)


def validate_schema(frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))
    if missing or unexpected:
        raise SchemaValidationError(
            f"PaySim schema mismatch; missing={missing}, unexpected={unexpected}"
        )


def _invalid_numeric_mask(frame: pd.DataFrame) -> pd.Series:
    invalid = pd.Series(False, index=frame.index)
    for column in NUMERIC_COLUMNS:
        invalid |= pd.to_numeric(frame[column], errors="coerce").isna()
    return invalid


def validate_batch(frame: pd.DataFrame) -> ValidationResult:
    """Partition one raw batch into valid and rejected records.

    Every rejected record receives a pipe-delimited ``validation_errors`` value.
    Exact duplicate rows are removed from the valid set and counted separately.
    """
    validate_schema(frame)
    work = frame.copy()
    errors = pd.Series("", index=work.index, dtype="string")

    def mark(mask: pd.Series, code: str) -> None:
        nonlocal errors
        errors.loc[mask] = errors.loc[mask].map(lambda value: f"{value}|{code}".strip("|"))

    missing = work[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    mark(missing, "missing_value")
    invalid_numeric = _invalid_numeric_mask(work)
    mark(invalid_numeric, "invalid_data_type")

    numeric = work.copy()
    for column in NUMERIC_COLUMNS:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    mark(numeric["step"].notna() & (numeric["step"] < 1), "invalid_step")
    for column in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
        mark(numeric[column].notna() & (numeric[column] < 0), f"negative_{column}")
    mark(~work["type"].isin(ALLOWED_TYPES), "invalid_transaction_type")
    mark(~numeric["isFraud"].isin([0, 1]), "invalid_fraud_label")
    mark(~numeric["isFlaggedFraud"].isin([0, 1]), "invalid_flag_label")
    mark(~work["nameOrig"].astype("string").str.match(r"^C\d+$", na=False), "invalid_origin_account")
    mark(~work["nameDest"].astype("string").str.match(r"^[CM]\d+$", na=False), "invalid_destination_account")
    mark(work["nameOrig"].eq(work["nameDest"]), "self_transfer")

    duplicate = work.duplicated(keep="first")
    mark(duplicate, "duplicate_in_batch")
    rejected_mask = errors.ne("")
    rejected = work.loc[rejected_mask].copy()
    rejected["validation_errors"] = errors.loc[rejected_mask]
    valid = numeric.loc[~rejected_mask].copy()

    issue_counts = {
        code: int(errors.str.split("|").map(lambda parts: code in parts).sum())
        for code in sorted({code for value in errors if value for code in value.split("|")})
    }
    return ValidationResult(valid=valid, rejected=rejected, issue_counts=issue_counts)
