"""Leakage-safe transaction and behavioral fraud features.

The stateful transformer accepts chronologically ordered chunks. History features
exclude the current transaction and preserve state between chunks, matching how
features would be calculated by an online scoring service.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


RAW_TO_CANONICAL = {
    "nameOrig": "origin_account_id",
    "oldbalanceOrg": "origin_old_balance",
    "newbalanceOrig": "origin_new_balance",
    "nameDest": "destination_account_id",
    "oldbalanceDest": "destination_old_balance",
    "newbalanceDest": "destination_new_balance",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}

REQUIRED_COLUMNS = {
    "step", "type", "amount", "origin_account_id", "origin_old_balance",
    "origin_new_balance", "destination_account_id", "destination_old_balance",
    "destination_new_balance",
}

FEATURE_COLUMNS = [
    "amount", "transaction_type", "origin_old_balance", "origin_new_balance",
    "destination_old_balance", "destination_new_balance", "origin_balance_delta",
    "destination_balance_delta", "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio", "origin_balance_error",
    "destination_balance_error", "zero_balance_after_transaction",
    "full_balance_transfer", "large_transaction", "transaction_velocity",
    "account_transaction_count", "average_account_amount",
    "deviation_from_account_average", "destination_frequency",
    "high_risk_transaction_type",
]


@dataclass
class FeatureState:
    """Minimal online history required for behavioral features."""

    origin_counts: dict[str, int] = field(default_factory=dict)
    origin_amount_sums: dict[str, float] = field(default_factory=dict)
    destination_counts: dict[str, int] = field(default_factory=dict)
    origin_step_counts: dict[tuple[str, int], int] = field(default_factory=dict)


class FraudFeatureEngineer:
    def __init__(self, large_transaction_threshold: float = 200_000.0) -> None:
        if large_transaction_threshold <= 0:
            raise ValueError("large_transaction_threshold must be positive")
        self.large_transaction_threshold = large_transaction_threshold
        self.state = FeatureState()
        self._last_step: int | None = None

    def reset(self) -> None:
        self.state = FeatureState()
        self._last_step = None

    def transform(self, frame: pd.DataFrame, *, update_state: bool = True) -> pd.DataFrame:
        """Add fraud features to a chronologically ordered transaction batch.

        The input may use raw PaySim or canonical FinShield column names. Rows must
        be ordered by ``step``; stable source order breaks ties within a step.
        """
        work = frame.rename(columns=RAW_TO_CANONICAL).copy()
        missing = sorted(REQUIRED_COLUMNS - set(work.columns))
        if missing:
            raise ValueError(f"missing feature inputs: {missing}")
        if not work["step"].is_monotonic_increasing:
            raise ValueError("feature batches must be ordered by step to prevent leakage")
        if self._last_step is not None and len(work) and int(work["step"].iloc[0]) < self._last_step:
            raise ValueError("batch starts before previously processed history")
        if work.empty:
            return work.assign(**{column: pd.Series(dtype="float64") for column in FEATURE_COLUMNS if column not in work})

        work["transaction_type"] = work["type"].astype("string").str.upper()
        work["origin_balance_delta"] = work["origin_old_balance"] - work["origin_new_balance"]
        work["destination_balance_delta"] = work["destination_new_balance"] - work["destination_old_balance"]
        work["amount_to_origin_balance_ratio"] = work["amount"] / (work["origin_old_balance"] + 1.0)
        work["amount_to_destination_balance_ratio"] = work["amount"] / (work["destination_old_balance"] + 1.0)

        incoming_origin = work["transaction_type"].eq("CASH_IN")
        expected_origin = np.where(
            incoming_origin,
            work["origin_old_balance"] + work["amount"],
            work["origin_old_balance"] - work["amount"],
        )
        expected_destination = work["destination_old_balance"] + work["amount"]
        merchant_payment = work["transaction_type"].eq("PAYMENT") & work["destination_account_id"].str.startswith("M")
        work["origin_balance_error"] = (work["origin_new_balance"] - expected_origin).abs()
        work["destination_balance_error"] = (work["destination_new_balance"] - expected_destination).abs()
        # PaySim does not update merchant balances, so their apparent error is structural rather than suspicious.
        work.loc[merchant_payment, "destination_balance_error"] = 0.0

        outgoing = ~incoming_origin
        work["zero_balance_after_transaction"] = work["origin_new_balance"].eq(0).astype("int8")
        work["full_balance_transfer"] = (
            outgoing
            & work["origin_new_balance"].eq(0)
            & np.isclose(work["origin_old_balance"], work["amount"], rtol=1e-5, atol=0.01)
        ).astype("int8")
        work["large_transaction"] = work["amount"].ge(self.large_transaction_threshold).astype("int8")
        work["high_risk_transaction_type"] = work["transaction_type"].isin(["TRANSFER", "CASH_OUT"]).astype("int8")

        origin = work["origin_account_id"].astype("string")
        destination = work["destination_account_id"].astype("string")
        prior_in_batch = work.groupby(origin, sort=False).cumcount()
        starting_count = origin.map(self.state.origin_counts).fillna(0).astype("int64")
        work["account_transaction_count"] = starting_count + prior_in_batch

        within_origin_sum = work["amount"].groupby(origin, sort=False).cumsum() - work["amount"]
        starting_sum = origin.map(self.state.origin_amount_sums).fillna(0.0)
        historical_sum = starting_sum + within_origin_sum
        history_count = work["account_transaction_count"]
        work["average_account_amount"] = np.where(history_count > 0, historical_sum / history_count, 0.0)
        work["deviation_from_account_average"] = np.where(
            history_count > 0, work["amount"] - work["average_account_amount"], 0.0
        )

        work["destination_frequency"] = (
            destination.map(self.state.destination_counts).fillna(0).astype("int64")
            + work.groupby(destination, sort=False).cumcount()
        )
        velocity_key = pd.Series(list(zip(origin, work["step"].astype(int))), index=work.index)
        work["transaction_velocity"] = (
            velocity_key.map(self.state.origin_step_counts).fillna(0).astype("int64")
            + work.groupby([origin, work["step"]], sort=False).cumcount()
            + 1
        )

        if update_state:
            self._update_state(work)
        return work

    def feature_matrix(self, frame: pd.DataFrame, *, update_state: bool = True) -> pd.DataFrame:
        return self.transform(frame, update_state=update_state)[FEATURE_COLUMNS]

    def _update_state(self, work: pd.DataFrame) -> None:
        origin_summary = work.groupby("origin_account_id", sort=False)["amount"].agg(["count", "sum"])
        for account, row in origin_summary.iterrows():
            self.state.origin_counts[account] = self.state.origin_counts.get(account, 0) + int(row["count"])
            self.state.origin_amount_sums[account] = self.state.origin_amount_sums.get(account, 0.0) + float(row["sum"])
        for account, count in work.groupby("destination_account_id", sort=False).size().items():
            self.state.destination_counts[account] = self.state.destination_counts.get(account, 0) + int(count)
        for key, count in work.groupby(["origin_account_id", "step"], sort=False).size().items():
            self.state.origin_step_counts[key] = self.state.origin_step_counts.get(key, 0) + int(count)
        # Steps never recur after chronological advancement; discard stale velocity state.
        self._last_step = int(work["step"].iloc[-1])
        self.state.origin_step_counts = {
            key: count for key, count in self.state.origin_step_counts.items() if key[1] >= self._last_step
        }
