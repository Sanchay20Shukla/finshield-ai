"""Train and compare leakage-aware fraud baselines by business cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features.fraud_features import FEATURE_COLUMNS, FraudFeatureEngineer
from src.models.evaluate import evaluate_probabilities, optimize_threshold


CATEGORICAL_FEATURES = ["transaction_type"]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]


def temporal_cutoffs(csv_path: Path) -> tuple[int, int]:
    escaped = csv_path.resolve().as_posix().replace("'", "''")
    with duckdb.connect() as connection:
        row = connection.execute(
            f"SELECT quantile_disc(step, 0.70), quantile_disc(step, 0.85) FROM read_csv_auto('{escaped}', header=true)"
        ).fetchone()
    return int(row[0]), int(row[1])


def prepare_sampled_data(
    csv_path: Path, *, negative_sample_rate: float = 0.03, chunk_size: int = 100_000,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build chronological train/validation/test samples while keeping every fraud.

    Legitimate records are sampled after feature computation. Evaluation weights
    restore their original population count for cost and confusion-matrix metrics.
    """
    if not 0 < negative_sample_rate <= 1:
        raise ValueError("negative_sample_rate must be in (0, 1]")
    train_end, validation_end = temporal_cutoffs(csv_path)
    engineer = FraudFeatureEngineer()
    rng = np.random.default_rng(random_seed)
    pieces: dict[str, list[pd.DataFrame]] = {"train": [], "validation": [], "test": []}
    population: dict[str, dict[int, int]] = {name: {0: 0, 1: 0} for name in pieces}
    sampled: dict[str, dict[int, int]] = {name: {0: 0, 1: 0} for name in pieces}

    columns = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
    ]
    for raw in pd.read_csv(csv_path, usecols=columns, chunksize=chunk_size):
        featured = engineer.transform(raw)
        split = np.where(featured["step"] <= train_end, "train", np.where(featured["step"] <= validation_end, "validation", "test"))
        for name in pieces:
            subset = featured.loc[split == name]
            if subset.empty:
                continue
            labels = subset["is_fraud"].astype(int)
            population[name][0] += int((labels == 0).sum())
            population[name][1] += int((labels == 1).sum())
            keep = labels.eq(1).to_numpy() | (rng.random(len(subset)) < negative_sample_rate)
            selected = subset.loc[keep, FEATURE_COLUMNS + ["is_fraud"]].copy()
            for label in (0, 1):
                sampled[name][label] += int((selected["is_fraud"] == label).sum())
            pieces[name].append(selected)

    output: dict[str, pd.DataFrame] = {}
    for name, frames in pieces.items():
        data = pd.concat(frames, ignore_index=True)
        weights = {
            label: population[name][label] / sampled[name][label] if sampled[name][label] else 0.0
            for label in (0, 1)
        }
        data["evaluation_weight"] = data["is_fraud"].astype(int).map(weights)
        output[name] = data
    return output


def build_models(y_train: pd.Series, random_seed: int = 42) -> dict[str, Pipeline]:
    negatives, positives = np.bincount(y_train.astype(int), minlength=2)
    scale_pos_weight = negatives / max(positives, 1)
    scaled_preprocessor = ColumnTransformer([
        ("type", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ])
    tree_preprocessor = ColumnTransformer([
        ("type", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", "passthrough", NUMERIC_FEATURES),
    ])
    return {
        "logistic_regression": Pipeline([
            ("preprocess", scaled_preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", tree_preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=16, min_samples_leaf=3,
                class_weight="balanced_subsample", n_jobs=-1, random_state=random_seed,
            )),
        ]),
        "xgboost": Pipeline([
            ("preprocess", tree_preprocessor),
            ("model", XGBClassifier(
                n_estimators=350, max_depth=6, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85, scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr", n_jobs=-1, random_state=random_seed,
            )),
        ]),
    }


def train_and_compare(
    csv_path: Path, output_dir: Path, *, negative_sample_rate: float = 0.03,
    fraud_cost: float = 10_000.0, review_cost: float = 50.0,
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    data = prepare_sampled_data(csv_path, negative_sample_rate=negative_sample_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = data["train"][FEATURE_COLUMNS], data["train"]["is_fraud"].astype(int)
    results = []
    for name, model in build_models(y_train).items():
        model.fit(X_train, y_train)
        validation = data["validation"]
        validation_probability = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        optimum = optimize_threshold(
            validation["is_fraud"], validation_probability,
            sample_weight=validation["evaluation_weight"], fraud_cost=fraud_cost, review_cost=review_cost,
        )
        test = data["test"]
        test_probability = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        metrics = evaluate_probabilities(
            test["is_fraud"], test_probability, optimum.threshold,
            sample_weight=test["evaluation_weight"], fraud_cost=fraud_cost, review_cost=review_cost,
        )
        record = {"model": name, "validation_expected_loss": optimum.expected_loss, **metrics.to_dict()}
        results.append(record)
        joblib.dump(
            {"pipeline": model, "threshold": optimum.threshold, "features": FEATURE_COLUMNS, "metrics": record},
            output_dir / f"{name}.joblib",
        )
        if tracking_uri:
            from src.models.tracking import log_experiment
            log_experiment(name, model, record, output_dir / f"{name}.joblib",
                           {"negative_sample_rate": negative_sample_rate, "fraud_cost": fraud_cost,
                            "review_cost": review_cost, "threshold": optimum.threshold}, tracking_uri)
    comparison = pd.DataFrame(results).sort_values("validation_expected_loss", kind="stable")
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    (output_dir / "model_comparison.json").write_text(json.dumps(comparison.to_dict("records"), indent=2), encoding="utf-8")
    _write_report(comparison, output_dir, fraud_cost, review_cost, negative_sample_rate)
    _write_comparison_chart(comparison, output_dir)
    return comparison


def _write_report(comparison: pd.DataFrame, output_dir: Path, fraud_cost: float, review_cost: float, sample_rate: float) -> None:
    best = comparison.iloc[0]
    table = comparison[["model", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc", "false_positive_rate", "false_negative_rate", "expected_loss"]].to_markdown(index=False)
    report = f"""# Fraud model benchmark

## Selection

**{best['model']}** is selected by validation expected loss at threshold **{best['threshold']:.3f}**. Its final holdout expected loss is **INR {best['expected_loss']:,.0f}** under the stated cost assumptions.

{table}

## Business interpretation

The objective is `FN × INR {fraud_cost:,.0f} + FP × INR {review_cost:,.0f}`. A missed fraud therefore costs as much as {fraud_cost / review_cost:,.0f} false-positive reviews before customer-friction costs. This strongly favors recall, but the selected threshold is optimized on validation data rather than the test set. The test set is used once for final comparison.

Legitimate records were sampled at {sample_rate:.1%} after chronological feature computation; evaluation weights restore the original negative-class population for confusion counts and expected loss. Every fraud record is retained. Accuracy is intentionally excluded from model selection because a no-fraud classifier would appear strong on this imbalanced dataset.

PaySim is synthetic and its fraud mechanism is unusually separable. Near-perfect scores are evidence about this simulation, not proof of production performance; deployment requires calibration and monitoring on representative live traffic.
"""
    (output_dir / "model_report.md").write_text(report, encoding="utf-8")


def _write_comparison_chart(comparison: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    comparison.set_index("model")[["precision", "recall", "f1", "pr_auc"]].plot.bar(ax=axes[0])
    axes[0].set_title("Holdout fraud metrics")
    axes[0].set_ylim(0.9, 1.005)
    axes[0].tick_params(axis="x", rotation=20)
    comparison.set_index("model")["expected_loss"].plot.bar(ax=axes[1], color="#d9485f")
    axes[1].set_title("Expected loss at optimized threshold")
    axes[1].set_ylabel("INR")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FinShield fraud baselines")
    parser.add_argument("--csv", type=Path, default=Path("data/raw/paysim.csv"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--negative-sample-rate", type=float, default=0.03)
    parser.add_argument("--fraud-cost", type=float, default=10_000.0)
    parser.add_argument("--review-cost", type=float, default=50.0)
    parser.add_argument("--mlflow-uri", default=None, help="Optional MLflow URI, e.g. sqlite:///mlflow.db")
    args = parser.parse_args()
    comparison = train_and_compare(
        args.csv, args.output, negative_sample_rate=args.negative_sample_rate,
        fraud_cost=args.fraud_cost, review_cost=args.review_cost,
        tracking_uri=args.mlflow_uri,
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
