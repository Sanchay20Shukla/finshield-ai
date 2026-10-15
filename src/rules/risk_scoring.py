"""Unified ML, rules, and behavioral fraud-risk scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.rules.fraud_rules import FraudRulesEngine


@dataclass(frozen=True)
class RiskScoreConfig:
    ml_weight: float = 0.65
    rule_weight: float = 0.20
    behavioral_weight: float = 0.15
    approve_max: float = 30.0
    monitor_max: float = 60.0
    manual_review_max: float = 80.0

    def __post_init__(self) -> None:
        weights = (self.ml_weight, self.rule_weight, self.behavioral_weight)
        if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("risk-score weights must be non-negative and sum to 1")
        if not 0 <= self.approve_max < self.monitor_max < self.manual_review_max < 100:
            raise ValueError("decision thresholds must be strictly increasing within 0-100")


@dataclass(frozen=True)
class BehavioralSignal:
    signal: str
    score: float
    reason: str
    evidence: dict[str, Any]


class BehavioralRiskScorer:
    """Score non-model account and transaction behavior on a 0-100 scale."""

    def score(self, transaction: Mapping[str, Any]) -> tuple[float, list[BehavioralSignal]]:
        signals: list[BehavioralSignal] = []
        velocity = _number(transaction, "transaction_velocity", default=1)
        if velocity > 1:
            points = min(25.0, (velocity - 1) * 6.25)
            signals.append(BehavioralSignal(
                "VELOCITY", points, "Origin activity is elevated within the transaction window.",
                {"transaction_velocity": velocity},
            ))

        count = _number(transaction, "account_transaction_count")
        average = _number(transaction, "average_account_amount")
        amount = _number(transaction, "amount")
        multiple = amount / average if average > 0 else 0.0
        if count >= 3 and multiple > 1:
            points = min(25.0, (multiple - 1) * 6.25)
            signals.append(BehavioralSignal(
                "AMOUNT_DEVIATION", points, "Amount exceeds the origin account's prior average.",
                {"amount": amount, "historical_average": average, "multiple": multiple, "history_count": count},
            ))

        destination_frequency = _number(transaction, "destination_frequency")
        if destination_frequency > 0:
            points = min(15.0, destination_frequency * 3.0)
            signals.append(BehavioralSignal(
                "DESTINATION_REUSE", points, "Destination has appeared in prior transaction history.",
                {"prior_destination_frequency": destination_frequency},
            ))

        depleted = bool(transaction.get("zero_balance_after_transaction", False))
        full_balance = bool(transaction.get("full_balance_transfer", False))
        if depleted or full_balance:
            points = 20.0 if full_balance else 12.0
            signals.append(BehavioralSignal(
                "BALANCE_DEPLETION", points, "Transaction depletes all or nearly all origin funds.",
                {"zero_balance_after_transaction": depleted, "full_balance_transfer": full_balance},
            ))

        origin_error = _number(transaction, "origin_balance_error")
        destination_error = _number(transaction, "destination_balance_error")
        relative_error = (origin_error + destination_error) / max(amount, 1.0)
        if relative_error > 0.01:
            points = min(15.0, relative_error * 15.0)
            signals.append(BehavioralSignal(
                "BALANCE_INCONSISTENCY", points, "Recorded balances do not reconcile with the transaction amount.",
                {"origin_balance_error": origin_error, "destination_balance_error": destination_error, "relative_error": relative_error},
            ))
        return round(min(100.0, sum(signal.score for signal in signals)), 4), signals


@dataclass(frozen=True)
class UnifiedRiskDecision:
    final_risk_score: float
    decision: str
    ml_score: float
    rule_score: float
    behavioral_score: float
    weights: dict[str, float]
    rule_hits: list[dict[str, Any]]
    behavioral_signals: list[dict[str, Any]]
    score_formula: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnifiedRiskScoringEngine:
    def __init__(
        self, rules_engine: FraudRulesEngine | None = None,
        behavioral_scorer: BehavioralRiskScorer | None = None,
        config: RiskScoreConfig | None = None,
    ) -> None:
        self.rules_engine = rules_engine or FraudRulesEngine()
        self.behavioral_scorer = behavioral_scorer or BehavioralRiskScorer()
        self.config = config or RiskScoreConfig()

    def score(self, transaction: Mapping[str, Any], ml_probability: float) -> UnifiedRiskDecision:
        if not 0 <= ml_probability <= 1:
            raise ValueError("ml_probability must be between 0 and 1")
        ml_score = ml_probability * 100.0
        rule_hits = self.rules_engine.evaluate(transaction)
        rule_score = float(min(100, sum(hit.score for hit in rule_hits)))
        behavioral_score, behavioral_signals = self.behavioral_scorer.score(transaction)
        final = (
            self.config.ml_weight * ml_score
            + self.config.rule_weight * rule_score
            + self.config.behavioral_weight * behavioral_score
        )
        final = round(min(100.0, max(0.0, final)), 4)
        if final <= self.config.approve_max:
            decision = "APPROVE"
        elif final <= self.config.monitor_max:
            decision = "MONITOR"
        elif final <= self.config.manual_review_max:
            decision = "MANUAL_REVIEW"
        else:
            decision = "BLOCK"
        return UnifiedRiskDecision(
            final_risk_score=final,
            decision=decision,
            ml_score=round(ml_score, 4),
            rule_score=rule_score,
            behavioral_score=behavioral_score,
            weights={
                "ml": self.config.ml_weight,
                "rules": self.config.rule_weight,
                "behavioral": self.config.behavioral_weight,
            },
            rule_hits=[hit.to_dict() for hit in rule_hits],
            behavioral_signals=[asdict(signal) for signal in behavioral_signals],
            score_formula="ml_score × ml_weight + rule_score × rule_weight + behavioral_score × behavioral_weight",
        )


def _number(transaction: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = transaction.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be numeric") from error
