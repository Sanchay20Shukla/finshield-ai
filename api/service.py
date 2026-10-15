"""Compose feature engineering, ML, SHAP, rules, and unified scoring."""

from __future__ import annotations

import os
import hashlib
from dataclasses import asdict
import uuid
from functools import lru_cache
from pathlib import Path

import pandas as pd

from api.schemas import TransactionScoreRequest, TransactionScoreResponse
from src.features.fraud_features import FEATURE_COLUMNS, FraudFeatureEngineer
from src.investigation.risk_explanation import ShapRiskExplainer
from src.models.predict import FraudModel
from src.rules.risk_scoring import UnifiedRiskScoringEngine
from src.monitoring.telemetry import MonitoringRepository


class FraudScoringService:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise RuntimeError(f"model not found at {model_path}; run python -m src.models.train first")
        self.model = FraudModel(model_path)
        self.model_version = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
        self.explainer = ShapRiskExplainer(model_path)
        self.risk_engine = UnifiedRiskScoringEngine()

    def score(self, request: TransactionScoreRequest) -> TransactionScoreResponse:
        raw = pd.DataFrame([{
            "step": request.step, "type": request.type, "amount": request.amount,
            "nameOrig": request.nameOrig, "oldbalanceOrg": request.oldbalanceOrg,
            "newbalanceOrig": request.newbalanceOrig, "nameDest": request.nameDest,
            "oldbalanceDest": request.oldbalanceDest, "newbalanceDest": request.newbalanceDest,
        }])
        featured = FraudFeatureEngineer().transform(raw).iloc[0].to_dict()
        featured.update({
            "transaction_velocity": request.transaction_velocity,
            "account_transaction_count": request.account_transaction_count,
            "average_account_amount": request.average_account_amount,
            "destination_frequency": request.destination_frequency,
        })
        featured["deviation_from_account_average"] = (
            request.amount - request.average_account_amount
            if request.account_transaction_count and request.average_account_amount else 0.0
        )
        prediction = self.model.predict({column: featured[column] for column in FEATURE_COLUMNS})
        probability = prediction["fraud_probability"]
        decision = self.risk_engine.score(featured, probability)
        explanation = self.explainer.explain(featured, top_n=5)
        risk_factors = [factor.description for factor in explanation.primary_contributing_factors]
        if not risk_factors:
            risk_factors = [signal["reason"] for signal in decision.behavioral_signals[:5]]
        response = TransactionScoreResponse(
            transaction_id=f"TX-{uuid.uuid4().hex[:12].upper()}",
            risk_score=round(decision.final_risk_score, 2),
            fraud_probability=round(probability, 6), decision=decision.decision,
            triggered_rules=[hit["rule"] for hit in decision.rule_hits],
            risk_factors=risk_factors, rule_details=decision.rule_hits,
            score_breakdown={"ml_score": decision.ml_score, "rule_score": decision.rule_score, "behavioral_score": decision.behavioral_score},
            model_threshold=self.model.threshold, explanation_method=explanation.explanation_method,
            model_version=self.model_version,
            explanation_units="log_odds" if hasattr(self.model.pipeline.named_steps["model"], "get_booster") else "model_output",
            shap_factors=[asdict(factor) for factor in explanation.primary_contributing_factors + explanation.protective_factors],
        )
        MonitoringRepository.default().record_prediction(
            response.transaction_id, request.type, request.amount,
            response.fraud_probability, response.risk_score, response.decision,
        )
        from api.audit import record_prediction
        record_prediction(request, response, self.model_version)
        return response


@lru_cache(maxsize=1)
def get_scoring_service() -> FraudScoringService:
    from src.config import MODEL_PATH
    default_path = MODEL_PATH
    return FraudScoringService(Path(os.getenv("MODEL_PATH", str(default_path))))
