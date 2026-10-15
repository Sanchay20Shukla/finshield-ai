"""Explainable fraud rules and hybrid rules-plus-ML decisioning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    rule: str
    severity: Severity
    score: int
    reason: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["severity"] = self.severity.value
        return value


@dataclass(frozen=True)
class RuleConfig:
    currency: str = "INR"
    large_transfer_threshold: float = 200_000.0
    high_velocity_threshold: int = 4
    repeated_destination_threshold: int = 3
    unusual_amount_multiplier: float = 3.0
    unusual_amount_min_history: int = 3
    zero_tolerance: float = 0.01

    def __post_init__(self) -> None:
        numeric = (
            self.large_transfer_threshold, self.high_velocity_threshold,
            self.repeated_destination_threshold, self.unusual_amount_multiplier,
            self.unusual_amount_min_history,
        )
        if any(value <= 0 for value in numeric) or self.zero_tolerance < 0:
            raise ValueError("rule thresholds must be positive")


class FraudRulesEngine:
    """Evaluate deterministic policy rules against one featured transaction."""

    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig()

    def evaluate(self, transaction: Mapping[str, Any]) -> list[RuleHit]:
        tx = transaction
        hits: list[RuleHit] = []
        tx_type = str(tx.get("transaction_type", tx.get("type", ""))).upper()
        amount = _number(tx, "amount")
        origin_new = _number(tx, "origin_new_balance", "newbalanceOrig")
        origin_old = _number(tx, "origin_old_balance", "oldbalanceOrg")
        velocity = _number(tx, "transaction_velocity", default=1)
        destination_frequency = _number(tx, "destination_frequency", default=0)
        history_count = _number(tx, "account_transaction_count", default=0)
        historical_average = _number(tx, "average_account_amount", default=0)
        outgoing = tx_type != "CASH_IN"

        if tx_type == "TRANSFER" and amount >= self.config.large_transfer_threshold:
            hits.append(RuleHit(
                "RULE_001", "LARGE_TRANSFER", Severity.HIGH, 20,
                f"Transfer amount exceeds the configured {self.config.currency} threshold.",
                {"amount": amount, "threshold": self.config.large_transfer_threshold, "currency": self.config.currency},
            ))

        if outgoing and origin_new <= self.config.zero_tolerance and origin_old > self.config.zero_tolerance:
            hits.append(RuleHit(
                "RULE_002", "ORIGIN_BALANCE_DEPLETED", Severity.MEDIUM, 15,
                "The transaction leaves the funded origin account with no balance.",
                {"origin_old_balance": origin_old, "origin_new_balance": origin_new},
            ))

        complete_balance = (
            outgoing and origin_old > self.config.zero_tolerance
            and abs(amount - origin_old) <= max(self.config.zero_tolerance, origin_old * 1e-5)
            and origin_new <= self.config.zero_tolerance
        )
        if bool(tx.get("full_balance_transfer", False)) or complete_balance:
            hits.append(RuleHit(
                "RULE_003", "FULL_BALANCE_TRANSFER", Severity.HIGH, 25,
                "Transaction amount approximately equals the complete available origin balance.",
                {"amount": amount, "origin_old_balance": origin_old, "origin_new_balance": origin_new},
            ))

        if velocity >= self.config.high_velocity_threshold:
            hits.append(RuleHit(
                "RULE_004", "HIGH_TRANSACTION_VELOCITY", Severity.HIGH, 20,
                "Origin transaction velocity exceeds the configured review threshold.",
                {"transaction_velocity": int(velocity), "threshold": self.config.high_velocity_threshold},
            ))

        if tx_type in {"TRANSFER", "CASH_OUT"} and destination_frequency >= self.config.repeated_destination_threshold:
            hits.append(RuleHit(
                "RULE_005", "REPEATED_DESTINATION", Severity.MEDIUM, 15,
                "The destination has appeared repeatedly in prior transaction history.",
                {"prior_destination_frequency": int(destination_frequency), "threshold": self.config.repeated_destination_threshold},
            ))

        amount_multiple = amount / historical_average if historical_average > 0 else 0.0
        if history_count >= self.config.unusual_amount_min_history and amount_multiple >= self.config.unusual_amount_multiplier:
            hits.append(RuleHit(
                "RULE_006", "UNUSUAL_ACCOUNT_AMOUNT", Severity.MEDIUM, 15,
                "Amount is unusually high relative to the origin account's prior average.",
                {"amount": amount, "historical_average": historical_average, "multiple": amount_multiple, "history_count": int(history_count)},
            ))
        return hits


@dataclass(frozen=True)
class FraudDecision:
    risk_score: float
    model_score: float | None
    rules_score: int
    severity: Severity
    action: str
    rule_hits: list[dict[str, Any]]
    decision_reason: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["severity"] = self.severity.value
        return value


class FraudDecisionEngine:
    """Combine model probability and rule evidence into an operational decision.

    With a model, probability contributes up to 70 points and rules up to 30.
    Without a model, the uncapped rule score supports rules-only fallback mode.
    """

    def __init__(self, rules_engine: FraudRulesEngine | None = None) -> None:
        self.rules_engine = rules_engine or FraudRulesEngine()

    def decide(self, transaction: Mapping[str, Any], model_probability: float | None = None) -> FraudDecision:
        if model_probability is not None and not 0 <= model_probability <= 1:
            raise ValueError("model_probability must be between 0 and 1")
        hits = self.rules_engine.evaluate(transaction)
        raw_rule_score = min(100, sum(hit.score for hit in hits))
        risk_score = raw_rule_score if model_probability is None else min(100.0, model_probability * 70 + min(raw_rule_score, 30))

        if risk_score >= 75:
            severity, action = Severity.CRITICAL, "BLOCK_AND_REVIEW"
        elif risk_score >= 50:
            severity, action = Severity.HIGH, "MANUAL_REVIEW"
        elif risk_score >= 30:
            severity, action = Severity.MEDIUM, "MONITOR"
        else:
            severity, action = Severity.LOW, "APPROVE"

        # Strong deterministic evidence must reach an investigator even when the model is uncertain.
        if any(hit.severity is Severity.CRITICAL for hit in hits):
            severity, action = Severity.CRITICAL, "BLOCK_AND_REVIEW"
        elif len([hit for hit in hits if hit.severity is Severity.HIGH]) >= 2 and risk_score < 50:
            severity, action = Severity.HIGH, "MANUAL_REVIEW"

        reason = (
            f"Model probability {model_probability:.1%} combined with {len(hits)} rule hit(s)."
            if model_probability is not None
            else f"Rules-only fallback produced {len(hits)} rule hit(s)."
        )
        return FraudDecision(
            round(risk_score, 4), model_probability, raw_rule_score, severity, action,
            [hit.to_dict() for hit in hits], reason,
        )


def _number(transaction: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = transaction.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{key} must be numeric") from error
    return default
