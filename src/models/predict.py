"""Persisted-model inference shared by future API scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd


class FraudModel:
    def __init__(self, artifact_path: Path) -> None:
        artifact = joblib.load(artifact_path)
        self.pipeline = artifact["pipeline"]
        self.threshold = float(artifact["threshold"])
        self.features = list(artifact["features"])
        self.metrics = artifact.get("metrics", {})

    def predict(self, features: Mapping[str, Any]) -> dict[str, Any]:
        missing = sorted(set(self.features) - set(features))
        if missing:
            raise ValueError(f"missing model features: {missing}")
        frame = pd.DataFrame([{column: features[column] for column in self.features}])
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        return {
            "fraud_probability": probability,
            "predicted_fraud": probability >= self.threshold,
            "threshold": self.threshold,
        }
