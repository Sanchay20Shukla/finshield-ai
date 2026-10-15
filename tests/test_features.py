"""Phase 1 transformation tests."""

import pandas as pd
import pytest

from src.data.preprocessing import customer_records, transform_transactions
from src.features.fraud_features import FEATURE_COLUMNS, FraudFeatureEngineer


pytestmark = [pytest.mark.unit, pytest.mark.data]


def test_transform_creates_stable_transaction_metadata():
    frame = pd.DataFrame([{
        "step": 1, "type": "PAYMENT", "amount": 10.0, "nameOrig": "C1",
        "oldbalanceOrg": 20.0, "newbalanceOrig": 10.0, "nameDest": "M2",
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "isFraud": 0,
        "isFlaggedFraud": 0,
    }])
    transformed = transform_transactions(frame, row_offset=10)
    assert transformed.loc[0, "source_row_number"] == 12
    assert len(transformed.loc[0, "source_row_hash"]) == 64
    assert set(customer_records(transformed)["account_id"]) == {"C1", "M2"}


def feature_rows():
    return pd.DataFrame([
        {"step": 1, "type": "TRANSFER", "amount": 100.0, "nameOrig": "C1", "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0, "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 100.0},
        {"step": 1, "type": "CASH_OUT", "amount": 300.0, "nameOrig": "C1", "oldbalanceOrg": 500.0, "newbalanceOrig": 200.0, "nameDest": "C3", "oldbalanceDest": 10.0, "newbalanceDest": 310.0},
        {"step": 2, "type": "PAYMENT", "amount": 50.0, "nameOrig": "C1", "oldbalanceOrg": 200.0, "newbalanceOrig": 150.0, "nameDest": "M4", "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
    ])


def test_features_are_temporal_and_business_aware():
    features = FraudFeatureEngineer(large_transaction_threshold=250).transform(feature_rows())
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features["account_transaction_count"].tolist() == [0, 1, 2]
    assert features["average_account_amount"].tolist() == [0.0, 100.0, 200.0]
    assert features["deviation_from_account_average"].tolist() == [0.0, 200.0, -150.0]
    assert features["transaction_velocity"].tolist() == [1, 2, 1]
    assert features["full_balance_transfer"].tolist() == [1, 0, 0]
    assert features["large_transaction"].tolist() == [0, 1, 0]
    assert features["destination_balance_error"].tolist() == [0.0, 0.0, 0.0]


def test_state_is_preserved_across_chunks_without_current_row_leakage():
    engineer = FraudFeatureEngineer()
    first = engineer.transform(feature_rows().iloc[:2])
    second = engineer.transform(feature_rows().iloc[2:])
    assert first["account_transaction_count"].tolist() == [0, 1]
    assert second["account_transaction_count"].tolist() == [2]
    assert second["average_account_amount"].iloc[0] == 200.0


def test_out_of_order_batch_is_rejected():
    engineer = FraudFeatureEngineer()
    try:
        engineer.transform(feature_rows().iloc[::-1])
    except ValueError as error:
        assert "ordered by step" in str(error)
    else:
        raise AssertionError("out-of-order features would introduce leakage")
