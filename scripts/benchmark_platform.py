"""Reproduce a chronological ML/rules/anomaly benchmark without replacing serving artifacts."""
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data.warehouse import ROOT, WAREHOUSE
from src.features.fraud_features import FEATURE_COLUMNS
from src.models.evaluate import evaluate_probabilities, optimize_threshold
from src.models.train import build_models


def run(output: Path = ROOT / "models/reproduced", sample_rate: float = .03):
    output.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(WAREHOUSE), read_only=True) as db:
        db.execute("SET memory_limit='1GB'")
        db.execute("SET threads=4")
        cutoffs = db.execute("SELECT quantile_disc(step,.70),quantile_disc(step,.85) FROM fact_transactions").fetchone()
        # Window features operate on every row BEFORE deterministic negative sampling.
        data = db.execute("""WITH history AS (
            SELECT *, count(*) OVER origin_prior AS account_transaction_count,
            coalesce(avg(amount) OVER origin_prior,0) AS average_account_amount,
            count(*) OVER (PARTITION BY nameOrig,step ORDER BY id ROWS UNBOUNDED PRECEDING) AS transaction_velocity,
            count(*) OVER (PARTITION BY nameDest ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS destination_frequency
            FROM fact_transactions
            WINDOW origin_prior AS (PARTITION BY nameOrig ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ) SELECT id, step, type AS transaction_type, amount,
            oldbalanceOrg AS origin_old_balance,newbalanceOrig AS origin_new_balance,
            oldbalanceDest AS destination_old_balance,newbalanceDest AS destination_new_balance,
            oldbalanceOrg-newbalanceOrig AS origin_balance_delta,
            newbalanceDest-oldbalanceDest AS destination_balance_delta,
            amount/(oldbalanceOrg+1) AS amount_to_origin_balance_ratio,
            amount/(oldbalanceDest+1) AS amount_to_destination_balance_ratio,
            abs(newbalanceOrig-(CASE WHEN type='CASH_IN' THEN oldbalanceOrg+amount ELSE oldbalanceOrg-amount END)) AS origin_balance_error,
            CASE WHEN type='PAYMENT' AND starts_with(nameDest,'M') THEN 0 ELSE abs(newbalanceDest-(oldbalanceDest+amount)) END AS destination_balance_error,
            (newbalanceOrig=0)::INT AS zero_balance_after_transaction,
            (type!='CASH_IN' AND newbalanceOrig=0 AND abs(oldbalanceOrg-amount)<=.01+.00001*abs(amount))::INT AS full_balance_transfer,
            (amount>=200000)::INT AS large_transaction,
            transaction_velocity,account_transaction_count,average_account_amount,
            CASE WHEN account_transaction_count>0 THEN amount-average_account_amount ELSE 0 END AS deviation_from_account_average,
            destination_frequency,(type IN ('TRANSFER','CASH_OUT'))::INT AS high_risk_transaction_type,
            isFraud AS is_fraud,rule_alert,
            CASE WHEN isFraud=1 THEN 1.0 ELSE 1.0/? END AS evaluation_weight
            FROM history WHERE isFraud=1 OR (hash(id)%10000)<(?*10000) ORDER BY id""", [sample_rate, sample_rate]).df()
    splits = {"train": data[data.step<=cutoffs[0]],
              "validation": data[(data.step>cutoffs[0]) & (data.step<=cutoffs[1])],
              "test": data[data.step>cutoffs[1]]}
    train, val, test = [splits[k] for k in ["train", "validation", "test"]]
    results = []
    def evaluate(name, pval, ptest, artifact=None):
        optimum=optimize_threshold(val.is_fraud,pval,sample_weight=val.evaluation_weight)
        metrics=evaluate_probabilities(test.is_fraud,ptest,optimum.threshold,sample_weight=test.evaluation_weight)
        record={"model":name,"validation_expected_loss":optimum.expected_loss,**metrics.to_dict()}
        results.append(record)
        if artifact is not None:
            joblib.dump({"pipeline":artifact,"threshold":optimum.threshold,"features":FEATURE_COLUMNS,"metrics":record},output/f"{name}.joblib")
        print(name, json.dumps(record), flush=True)
    for name, model in build_models(train.is_fraud).items():
        print(f"Fitting {name} on {len(train)} records", flush=True)
        model.fit(train[FEATURE_COLUMNS],train.is_fraud)
        evaluate(name,model.predict_proba(val[FEATURE_COLUMNS])[:,1],model.predict_proba(test[FEATURE_COLUMNS])[:,1],model)
    # Anomaly detector sees legitimate training data only; scaling parameters never use future rows.
    preprocessor=build_models(train.is_fraud)["random_forest"].named_steps["preprocess"]
    xtrain=preprocessor.fit_transform(train.loc[train.is_fraud==0,FEATURE_COLUMNS])
    anomaly=IsolationForest(n_estimators=100,max_samples=4096,random_state=42,n_jobs=-1)
    anomaly.fit(xtrain)
    # Monotone fixed mapping is an anomaly index, not a calibrated fraud probability.
    def anomaly_index(frame):
        return np.clip(-anomaly.score_samples(preprocessor.transform(frame[FEATURE_COLUMNS])),0,1)
    evaluate("isolation_forest",anomaly_index(val),anomaly_index(test))
    joblib.dump({"preprocess":preprocessor,"model":anomaly},output/"isolation_forest.joblib")
    rule_metrics=evaluate_probabilities(test.is_fraud,test.rule_alert.astype(float),.5,sample_weight=test.evaluation_weight)
    results.append({"model":"rules_baseline","validation_expected_loss":None,**rule_metrics.to_dict()})
    selection=min([r for r in results if r["model"] not in {"rules_baseline","isolation_forest"}],key=lambda r:r["validation_expected_loss"])
    report={"source":"Local full PaySim; deterministic 3% negative sample after full history windows",
            "train_end_step":int(cutoffs[0]),"validation_end_step":int(cutoffs[1]),
            "sample_counts":{name:len(frame) for name,frame in splits.items()},
            "sampling":"All fraud retained; sampled negatives carry inverse-probability weights (1/rate). Population estimates are approximate.",
            "selected_model":selection["model"],"serving_model_changed":False,
            "cost":"10000 per missed fraud + 50 per false positive, in hypothetical cost units",
            "scope":"New chronological benchmark. Serving remains the original XGBoost artifact. Isolation score is an anomaly index, not a probability.",
            "results":results}
    (output/"benchmark.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    pd.DataFrame(results).to_csv(output/"benchmark.csv",index=False)
    return report


if __name__ == "__main__":
    run()
