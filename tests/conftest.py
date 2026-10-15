"""Shared deterministic Pytest fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def valid_paysim_transaction() -> dict:
    return {
        "step": 1, "type": "TRANSFER", "amount": 300_000.0,
        "nameOrig": "C1001", "oldbalanceOrg": 300_000.0,
        "newbalanceOrig": 0.0, "nameDest": "C2002",
        "oldbalanceDest": 10_000.0, "newbalanceDest": 310_000.0,
        "isFraud": 1, "isFlaggedFraud": 0,
    }


@pytest.fixture
def valid_paysim_frame(valid_paysim_transaction) -> pd.DataFrame:
    return pd.DataFrame([valid_paysim_transaction])
