"""SHAP-based model explanations translated for fraud investigators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class RiskFactor:
    feature: str
    description: str
    contribution: float
    direction: str
    value: Any


@dataclass(frozen=True)
class RiskExplanation:
    fraud_risk: float
    predicted_fraud: bool
    decision_threshold: float
    primary_contributing_factors: list[RiskFactor]
    protective_factors: list[RiskFactor]
    explanation_method: str = "SHAP"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ShapRiskExplainer:
    """Explain a persisted FinShield model artifact at transaction level."""

    def __init__(self, artifact_path: Path) -> None:
        artifact = joblib.load(artifact_path)
        self.pipeline = artifact["pipeline"]
        self.threshold = float(artifact["threshold"])
        self.features = list(artifact["features"])
        self.preprocessor = self.pipeline.named_steps["preprocess"]
        self.estimator = self.pipeline.named_steps["model"]
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        self.explainer = self._build_explainer()

    def _build_explainer(self):
        if hasattr(self.estimator, "get_booster"):
            # Native TreeSHAP avoids SHAP/XGBoost vector-base-score incompatibility.
            return None
        if isinstance(self.estimator, LogisticRegression):
            background = np.zeros((1, len(self.feature_names)), dtype=float)
            return shap.LinearExplainer(self.estimator, background)
        return shap.TreeExplainer(self.estimator)

    def explain(self, features: Mapping[str, Any], *, top_n: int = 5) -> RiskExplanation:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        missing = sorted(set(self.features) - set(features))
        if missing:
            raise ValueError(f"missing explanation features: {missing}")
        frame = pd.DataFrame([{column: features[column] for column in self.features}])
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        transformed = self.preprocessor.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        if self.explainer is None:
            import xgboost as xgb
            contributions = self.estimator.get_booster().predict(
                xgb.DMatrix(np.asarray(transformed, dtype=float)), pred_contribs=True
            )[0, :-1]
        else:
            shap_values = self.explainer(np.asarray(transformed, dtype=float)).values
            contributions = _fraud_class_values(shap_values)

        factors = [
            _factor(feature_name, float(contribution), features)
            for feature_name, contribution in zip(self.feature_names, contributions)
        ]
        contributing = sorted(
            (factor for factor in factors if factor.contribution > 0),
            key=lambda factor: factor.contribution, reverse=True,
        )[:top_n]
        protective = sorted(
            (factor for factor in factors if factor.contribution < 0),
            key=lambda factor: factor.contribution,
        )[: min(3, top_n)]
        return RiskExplanation(
            fraud_risk=probability,
            predicted_fraud=probability >= self.threshold,
            decision_threshold=self.threshold,
            primary_contributing_factors=contributing,
            protective_factors=protective,
        )


def _fraud_class_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 3:
        # sklearn binary classifiers expose one SHAP vector per output class.
        return values[0, :, 1]
    if values.ndim == 2:
        return values[0]
    if values.ndim == 1:
        return values
    raise ValueError(f"unsupported SHAP output shape: {values.shape}")


def _factor(transformed_name: str, contribution: float, features: Mapping[str, Any]) -> RiskFactor:
    canonical = transformed_name.split("__", 1)[-1]
    if canonical.startswith("transaction_type_"):
        tx_type = canonical.removeprefix("transaction_type_")
        active = str(features.get("transaction_type", "")).upper() == tx_type
        description = f"{tx_type} transaction type" if active else f"not a {tx_type} transaction"
        value: Any = active
        feature = "transaction_type"
    else:
        feature = canonical
        value = features.get(canonical)
        description = _human_description(canonical, value)
    return RiskFactor(
        feature=feature,
        description=description,
        contribution=contribution,
        direction="INCREASES_RISK" if contribution > 0 else "REDUCES_RISK",
        value=_json_value(value),
    )


def _human_description(feature: str, value: Any) -> str:
    if feature == "origin_balance_error":
        return "origin balance reconciliation pattern" if float(value or 0) <= 0.01 else "origin balance does not reconcile with the transaction"
    if feature == "destination_balance_error":
        return "destination balance reconciliation pattern" if float(value or 0) <= 0.01 else "destination balance does not reconcile with the transaction"
    if feature == "zero_balance_after_transaction":
        return "complete origin balance depletion" if bool(value) else "origin account retains a balance"
    if feature == "full_balance_transfer":
        return "transaction uses the complete origin balance" if bool(value) else "partial-balance transaction pattern"
    if feature == "large_transaction":
        return "large transaction indicator" if bool(value) else "transaction below the large-amount threshold"
    if feature == "high_risk_transaction_type":
        return "high-risk transaction category" if bool(value) else "lower-risk transaction category"
    labels = {
        "amount": "unusually high transaction amount",
        "origin_old_balance": "origin account balance before the transaction",
        "origin_new_balance": "origin account balance after the transaction",
        "destination_old_balance": "destination account starting balance",
        "destination_new_balance": "destination account ending balance",
        "origin_balance_delta": "large origin balance reduction",
        "destination_balance_delta": "unusual destination balance change",
        "amount_to_origin_balance_ratio": "amount is large relative to the origin balance",
        "amount_to_destination_balance_ratio": "amount is large relative to the destination balance",
        "transaction_velocity": "abnormally high transaction velocity",
        "account_transaction_count": "origin account activity history",
        "average_account_amount": "origin account historical average amount",
        "deviation_from_account_average": "amount is unusual compared with account history",
        "destination_frequency": "destination account pattern",
    }
    return labels.get(feature, feature.replace("_", " "))


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
