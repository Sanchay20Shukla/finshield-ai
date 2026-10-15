"""Export real XGBoost trees and SHAP explanations; no fabricated predictions.

Run from repository root: python -m scripts.export_demo
The small label-stratified sample is a UI fixture, never an evaluation set.
"""
import hashlib
import json
import math
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from src.features.fraud_features import FEATURE_COLUMNS, FraudFeatureEngineer
from src.investigation.risk_explanation import ShapRiskExplainer
from src.rules.risk_scoring import UnifiedRiskScoringEngine

ROOT = Path(__file__).resolve().parents[1]


def export():
    output = ROOT / "web/data"
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = ROOT / "models/xgboost.joblib"
    artifact = joblib.load(artifact_path)
    pipeline = artifact["pipeline"]
    booster = pipeline.named_steps["model"].get_booster()
    config = json.loads(booster.save_config())
    base = float(str(config["learner"]["learner_model_param"]["base_score"]).strip("[]"))
    def compact(node):
        if "leaf" in node:
            return float(node["leaf"])
        children = {child["nodeid"]: child for child in node["children"]}
        return [int(node["split"].removeprefix("f")), node["split_condition"],
                compact(children[node["yes"]]), compact(children[node["no"]])]
    model = {
        "name": "XGBoost", "version": hashlib.sha256(artifact_path.read_bytes()).hexdigest()[:12],
        "threshold": artifact["threshold"], "baseMargin": math.log(base / (1 - base)),
        "featureNames": list(pipeline.named_steps["preprocess"].get_feature_names_out()),
        "trees": [compact(json.loads(tree)) for tree in booster.get_dump(dump_format="json")],
        "metrics": artifact["metrics"],
    }
    (output / "model.json").write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    source = ROOT / "data/raw/paysim.csv"
    fixture = ROOT / "data/sample/paysim_demo.csv"
    if source.exists():
        with duckdb.connect() as con:
            path = source.as_posix().replace("'", "''")
            raw = con.execute(f"""SELECT * EXCLUDE(rn) FROM (
                SELECT *, row_number() OVER(PARTITION BY isFraud, type ORDER BY hash(nameOrig, step)) rn
                FROM read_csv_auto('{path}')) WHERE rn <= 24 ORDER BY step, nameOrig""").fetchdf()
        raw.to_csv(fixture, index=False)
    else:
        raw = pd.read_csv(fixture)
    explainer = ShapRiskExplainer(artifact_path)
    engine = UnifiedRiskScoringEngine()
    records = []
    for i, row in raw.iterrows():
        # Single-transaction contract: no unobserved account history is invented.
        featured = FraudFeatureEngineer().transform(pd.DataFrame([row])).iloc[0].to_dict()
        probability = float(pipeline.predict_proba(pd.DataFrame([featured])[FEATURE_COLUMNS])[0, 1])
        explanation = explainer.explain(featured, top_n=5).to_dict()
        records.append({"id": f"FS-{i+1:04d}", "transaction": row.to_dict(),
                        "probability": probability, "decision": engine.score(featured, probability).to_dict(),
                        "explanation": explanation})
    (output / "cases.json").write_text(json.dumps(records, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), separators=(",", ":")), encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps({
        "dataset": "PaySim synthetic mobile money dataset",
        "source": "https://www.kaggle.com/datasets/ealaxi/paysim1",
        "sample": "Up to 24 records per label/type stratum, deterministic hash selection. Not representative prevalence.",
        "rows": len(raw), "model_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "mode": "Post-transaction detection; post-event balances are required.",
        "evaluation": "Historical artifact metrics are not a freshly reproduced benchmark or production guarantee.",
    }, indent=2), encoding="utf-8")
    print(f"Exported {len(model['trees'])} trees and {len(records)} explained demo transactions")


if __name__ == "__main__":
    export()
