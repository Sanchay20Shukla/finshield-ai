"""Canonical transformations applied after PaySim validation."""

from __future__ import annotations

import hashlib

import pandas as pd


RENAME_MAP = {
    "nameOrig": "origin_account_id",
    "oldbalanceOrg": "origin_old_balance",
    "newbalanceOrig": "origin_new_balance",
    "nameDest": "destination_account_id",
    "oldbalanceDest": "destination_old_balance",
    "newbalanceDest": "destination_new_balance",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}


def transform_transactions(frame: pd.DataFrame, row_offset: int = 0) -> pd.DataFrame:
    transformed = frame.rename(columns=RENAME_MAP).copy()
    transformed.columns = [column.lower() for column in transformed.columns]
    transformed["type"] = transformed["type"].astype("string").str.upper()
    transformed["step"] = transformed["step"].astype("int32")
    transformed["is_fraud"] = transformed["is_fraud"].astype(bool)
    transformed["is_flagged_fraud"] = transformed["is_flagged_fraud"].astype(bool)
    transformed["source_row_number"] = range(row_offset + 2, row_offset + len(transformed) + 2)

    hash_columns = [
        "step", "type", "amount", "origin_account_id", "origin_old_balance",
        "origin_new_balance", "destination_account_id", "destination_old_balance",
        "destination_new_balance", "is_fraud", "is_flagged_fraud",
    ]
    transformed["source_row_hash"] = transformed[hash_columns].astype(str).agg("|".join, axis=1).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    return transformed


def customer_records(transactions: pd.DataFrame) -> pd.DataFrame:
    origins = transactions[["origin_account_id"]].rename(columns={"origin_account_id": "account_id"})
    destinations = transactions[["destination_account_id"]].rename(columns={"destination_account_id": "account_id"})
    customers = pd.concat([origins, destinations], ignore_index=True).drop_duplicates()
    customers["account_type"] = customers["account_id"].str[0].map({"C": "CUSTOMER", "M": "MERCHANT"})
    return customers
