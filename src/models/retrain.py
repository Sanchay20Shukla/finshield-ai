"""Train and evaluate a feedback-enriched Random Forest candidate model."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from src.features.fraud_features import FEATURE_COLUMNS
from src.models.evaluate import evaluate_probabilities, optimize_threshold
from src.models.feedback import FeedbackRepository
from src.models.train import build_models, prepare_sampled_data


def retrain_with_feedback(
    csv_path: Path, feedback_database: Path, output_dir: Path, *,
    minimum_feedback: int = 25, negative_sample_rate: float = 0.03,
    fraud_cost: float = 10_000.0, review_cost: float = 50.0,
    promote: bool = False,
) -> dict:
    feedback = FeedbackRepository(feedback_database).training_frame()
    if len(feedback) < minimum_feedback:
        raise ValueError(f"at least {minimum_feedback} reviewed cases are required; found {len(feedback)}")
    data = prepare_sampled_data(csv_path, negative_sample_rate=negative_sample_rate)
    train = data["train"]
    X_train = train[FEATURE_COLUMNS]
    y_train = train["is_fraud"].astype(int)
    X_combined = __import__("pandas").concat([X_train, feedback[FEATURE_COLUMNS]], ignore_index=True)
    y_combined = __import__("pandas").concat([y_train, feedback["is_fraud"].astype(int)], ignore_index=True)
    sample_weight = np.concatenate([np.ones(len(train)), np.full(len(feedback), 5.0)])
    model = build_models(y_combined)["random_forest"]
    model.fit(X_combined, y_combined, model__sample_weight=sample_weight)

    validation = data["validation"]
    validation_probability = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    optimum = optimize_threshold(
        validation["is_fraud"], validation_probability, sample_weight=validation["evaluation_weight"],
        fraud_cost=fraud_cost, review_cost=review_cost,
    )
    test = data["test"]
    test_probability = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    metrics = evaluate_probabilities(
        test["is_fraud"], test_probability, optimum.threshold, sample_weight=test["evaluation_weight"],
        fraud_cost=fraud_cost, review_cost=review_cost,
    ).to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("feedback-%Y%m%dT%H%M%SZ")
    candidate_path = output_dir / f"random_forest_{version}.joblib"
    artifact = {
        "pipeline": model, "threshold": optimum.threshold, "features": FEATURE_COLUMNS,
        "metrics": metrics, "model_version": version, "feedback_records": len(feedback),
    }
    joblib.dump(artifact, candidate_path)
    report = {
        "candidate": str(candidate_path), "model_version": version,
        "feedback_records": len(feedback), "feedback_fraud": int(feedback["is_fraud"].sum()),
        "feedback_false_positive": int((feedback["is_fraud"] == 0).sum()),
        "threshold": optimum.threshold, "metrics": metrics, "promoted": bool(promote),
    }
    if promote:
        shutil.copy2(candidate_path, output_dir / "random_forest.joblib")
    (output_dir / f"retraining_{version}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    feedback.to_csv(output_dir / f"feedback_training_{version}.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain fraud model with investigator feedback")
    parser.add_argument("--csv", type=Path, default=Path("data/raw/paysim.csv"))
    parser.add_argument("--feedback-db", type=Path, default=Path("data/processed/feedback.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--minimum-feedback", type=int, default=25)
    parser.add_argument("--negative-sample-rate", type=float, default=0.03)
    parser.add_argument("--promote", action="store_true", help="Explicitly replace the serving Random Forest after evaluation")
    args = parser.parse_args()
    print(json.dumps(retrain_with_feedback(
        args.csv, args.feedback_db, args.output, minimum_feedback=args.minimum_feedback,
        negative_sample_rate=args.negative_sample_rate, promote=args.promote,
    ), indent=2))


if __name__ == "__main__":
    main()
