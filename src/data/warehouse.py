"""Reproducible full-PaySim analytics store. Never substitutes a curated fixture."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = Path(os.getenv("FINSHIELD_WAREHOUSE", str(ROOT / "data/processed/analytics.duckdb")))


def build(csv: Path, target: Path = WAREHOUSE) -> dict:
    if not csv.is_file():
        raise FileNotFoundError(f"Download PaySim and supply its CSV: {csv}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Build alongside the current database; replace only after a successful transaction.
    staging = target.with_suffix(".building.duckdb")
    if staging.exists():
        staging.unlink()
    with duckdb.connect(str(staging)) as db:
        db.execute("SET memory_limit='1GB'")
        db.execute("SET threads=4")
        db.execute("CREATE TABLE raw AS SELECT row_number() OVER ()::BIGINT AS id, * FROM read_csv_auto(?, header=true)", [str(csv.resolve())])
        invalid = db.execute("""SELECT count(*) FROM raw WHERE step IS NULL OR step < 1
            OR type IS NULL OR type NOT IN ('TRANSFER','CASH_OUT','CASH_IN','PAYMENT','DEBIT')
            OR amount IS NULL OR NOT isfinite(amount) OR amount < 0
            OR oldbalanceOrg IS NULL OR oldbalanceOrg < 0 OR NOT isfinite(oldbalanceOrg)
            OR newbalanceOrig IS NULL OR newbalanceOrig < 0 OR NOT isfinite(newbalanceOrig)
            OR oldbalanceDest IS NULL OR oldbalanceDest < 0 OR NOT isfinite(oldbalanceDest)
            OR newbalanceDest IS NULL OR newbalanceDest < 0 OR NOT isfinite(newbalanceDest)
            OR nameOrig IS NULL OR nameDest IS NULL OR nameOrig = nameDest
            OR isFraud IS NULL OR isFraud NOT IN (0,1)
            OR isFlaggedFraud IS NULL OR isFlaggedFraud NOT IN (0,1)""").fetchone()[0]
        if invalid:
            raise ValueError(f"Data quality failed: {invalid} invalid rows. Source was not published.")
        out_of_order = db.execute("""SELECT count(*) FROM (
            SELECT step, lag(step) OVER (ORDER BY id) AS previous_step FROM raw
            ) WHERE step < previous_step""").fetchone()[0]
        if out_of_order:
            raise ValueError("Source is not chronological; prior-history features would be invalid")
        db.execute("""CREATE TABLE fact_transactions AS SELECT *,
            (floor((step-1)/24)+1)::INTEGER AS simulation_day,
            type IN ('TRANSFER','CASH_OUT') AND amount >= 200000 AS rule_alert,
            CASE WHEN type IN ('TRANSFER','CASH_OUT') AND amount >= 200000 THEN 'HIGH'
                 WHEN type IN ('TRANSFER','CASH_OUT') THEN 'MEDIUM' ELSE 'LOW' END AS risk_band
            FROM raw""")
        db.execute("DROP TABLE raw")
        db.execute("CREATE INDEX transaction_id ON fact_transactions(id)")
        db.execute("CREATE VIEW dim_customer AS SELECT DISTINCT nameOrig AS account_id FROM fact_transactions")
        db.execute("CREATE VIEW dim_merchant AS SELECT DISTINCT nameDest AS account_id FROM fact_transactions WHERE nameDest LIKE 'M%'")
        db.execute("CREATE VIEW dim_geography AS SELECT 'UNKNOWN' AS geography_key, 'Not supplied by PaySim' AS geography")
        db.execute("CREATE VIEW dim_date AS SELECT DISTINCT step AS simulation_hour, simulation_day FROM fact_transactions")
        db.execute("""CREATE TABLE daily AS SELECT simulation_day, count(*) AS transactions,
            sum(amount) AS volume, sum(isFraud) AS fraud_count,
            sum(CASE WHEN isFraud=1 THEN amount ELSE 0 END) AS fraud_value,
            sum(rule_alert::INT) AS alerts FROM fact_transactions GROUP BY 1 ORDER BY 1""")
        db.execute("""CREATE TABLE segments AS SELECT type, count(*) AS transactions, sum(amount) AS volume,
            sum(isFraud) AS fraud_count, sum(CASE WHEN isFraud=1 THEN amount ELSE 0 END) AS fraud_value,
            sum(rule_alert::INT) AS alerts FROM fact_transactions GROUP BY 1 ORDER BY 2 DESC""")
        metadata = {"source": str(csv.resolve()), "source_bytes": csv.stat().st_size,
                    "source_mtime_ns": csv.stat().st_mtime_ns, "dataset": "PaySim full local CSV",
                    "quality": "passed", "invalid_rows": invalid,
                    "rules": "TRANSFER or CASH_OUT with amount >= 200000; independent of ground truth",
                    "model_scope": "Existing trained artifacts; analytics risk bands use rules, not ML",
                    "date_scope": "Simulation days, not calendar dates"}
        db.execute("CREATE TABLE metadata AS SELECT ? AS payload", [json.dumps(metadata)])
    staging.replace(target)
    return metadata


def connect():
    if not WAREHOUSE.is_file():
        raise FileNotFoundError("Run python -m src.data.warehouse --csv data/raw/paysim.csv first")
    return duckdb.connect(str(WAREHOUSE), read_only=True)


def records(db, sql, params=None):
    cursor = db.execute(sql, params or [])
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "data/raw/paysim.csv")
    parser.add_argument("--output", type=Path, default=WAREHOUSE)
    args = parser.parse_args()
    print(json.dumps(build(args.csv, args.output), indent=2))
