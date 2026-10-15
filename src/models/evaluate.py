"""Fraud model metrics and business-cost threshold optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    true_negative: float
    false_positive: float
    false_negative: float
    true_positive: float
    false_positive_rate: float
    false_negative_rate: float
    expected_loss: float
    fraud_cost: float
    review_cost: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_probabilities(
    y_true, probabilities, threshold: float, *, sample_weight=None,
    fraud_cost: float = 10_000.0, review_cost: float = 50.0,
) -> EvaluationResult:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[0, 1], sample_weight=sample_weight
    ).ravel()
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return EvaluationResult(
        threshold=float(threshold),
        precision=float(precision_score(y_true, predicted, sample_weight=sample_weight, zero_division=0)),
        recall=float(recall_score(y_true, predicted, sample_weight=sample_weight, zero_division=0)),
        f1=float(f1_score(y_true, predicted, sample_weight=sample_weight, zero_division=0)),
        pr_auc=float(average_precision_score(y_true, probabilities, sample_weight=sample_weight)),
        roc_auc=float(roc_auc_score(y_true, probabilities, sample_weight=sample_weight)),
        true_negative=float(tn), false_positive=float(fp), false_negative=float(fn), true_positive=float(tp),
        false_positive_rate=float(fpr), false_negative_rate=float(fnr),
        expected_loss=float(fn * fraud_cost + fp * review_cost),
        fraud_cost=float(fraud_cost), review_cost=float(review_cost),
    )


def optimize_threshold(
    y_true, probabilities, *, sample_weight=None, fraud_cost: float = 10_000.0,
    review_cost: float = 50.0, thresholds=None,
) -> EvaluationResult:
    if thresholds is None:
        thresholds = np.unique(np.concatenate(([0.001], np.linspace(0.005, 0.995, 199), [0.999])))
    results = [
        evaluate_probabilities(
            y_true, probabilities, threshold, sample_weight=sample_weight,
            fraud_cost=fraud_cost, review_cost=review_cost,
        )
        for threshold in thresholds
    ]
    return min(results, key=lambda result: (result.expected_loss, -result.recall, result.false_positive))
