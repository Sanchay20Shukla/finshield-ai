"""PaySim ingestion and validation boundary tests."""

import pandas as pd
import pytest

from src.data.preprocessing import transform_transactions
from src.data.validation import SchemaValidationError, validate_batch


pytestmark = pytest.mark.data


def test_valid_transaction_passes_all_validation_gates(valid_paysim_frame):
    result = validate_batch(valid_paysim_frame)
    assert len(result.valid) == 1
    assert result.rejected.empty
    assert result.issue_counts == {}


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("amount", -1, "negative_amount"),
        ("type", "WIRE", "invalid_transaction_type"),
        ("isFraud", 2, "invalid_fraud_label"),
        ("nameOrig", "INVALID", "invalid_origin_account"),
    ],
)
def test_business_and_type_validation_rejects_invalid_values(valid_paysim_transaction, field, value, error_code):
    invalid = {**valid_paysim_transaction, field: value}
    result = validate_batch(pd.DataFrame([invalid]))
    assert result.valid.empty
    assert error_code in result.rejected.iloc[0]["validation_errors"]


def test_missing_required_column_fails_schema_validation(valid_paysim_frame):
    with pytest.raises(SchemaValidationError, match="missing=.*amount"):
        validate_batch(valid_paysim_frame.drop(columns=["amount"]))


def test_transformation_produces_idempotency_hash(valid_paysim_frame):
    valid = validate_batch(valid_paysim_frame).valid
    first = transform_transactions(valid)
    second = transform_transactions(valid)
    assert first.loc[0, "source_row_hash"] == second.loc[0, "source_row_hash"]
    assert len(first.loc[0, "source_row_hash"]) == 64
