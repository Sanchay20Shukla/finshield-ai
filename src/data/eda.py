"""Full-dataset, business-oriented exploratory analysis for PaySim."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


QUERIES = {
    "summary": """
        SELECT count(*) transactions,
               sum(isFraud) fraud_transactions,
               avg(isFraud) fraud_rate,
               sum(amount) total_amount,
               sum(amount * isFraud) fraud_amount,
               sum(amount * isFraud) / sum(amount) fraud_amount_rate,
               avg(amount) average_amount,
               avg(amount) FILTER (WHERE isFraud = 1) average_fraud_amount
        FROM paysim
    """,
    "fraud_by_type": """
        SELECT type, count(*) transactions, sum(isFraud) fraud_transactions,
               avg(isFraud) fraud_rate, sum(amount) total_amount,
               sum(amount * isFraud) fraud_amount,
               avg(amount) FILTER (WHERE isFraud = 1) average_fraud_amount
        FROM paysim GROUP BY type ORDER BY fraud_rate DESC, transactions DESC
    """,
    "fraud_by_size": """
        SELECT CASE WHEN amount < 1000 THEN '01_<1K'
                    WHEN amount < 10000 THEN '02_1K-10K'
                    WHEN amount < 100000 THEN '03_10K-100K'
                    WHEN amount < 1000000 THEN '04_100K-1M'
                    ELSE '05_1M+' END amount_band,
               count(*) transactions, sum(isFraud) fraud_transactions,
               avg(isFraud) fraud_rate, sum(amount * isFraud) fraud_amount
        FROM paysim GROUP BY amount_band ORDER BY amount_band
    """,
    "fraud_by_hour": """
        SELECT ((step - 1) % 24)::INTEGER hour_of_day, count(*) transactions,
               sum(isFraud) fraud_transactions, avg(isFraud) fraud_rate,
               sum(amount * isFraud) fraud_amount
        FROM paysim GROUP BY hour_of_day ORDER BY hour_of_day
    """,
    "origin_balance_behavior": """
        SELECT CASE WHEN oldbalanceOrg = 0 THEN 'zero_starting_balance'
                    WHEN newbalanceOrig = 0 THEN 'fully_depleted'
                    WHEN amount / NULLIF(oldbalanceOrg, 0) >= 0.9 THEN '90%+_depleted'
                    WHEN amount / NULLIF(oldbalanceOrg, 0) >= 0.5 THEN '50-90%_depleted'
                    ELSE 'under_50%_depleted' END behavior,
               count(*) transactions, sum(isFraud) fraud_transactions,
               avg(isFraud) fraud_rate, avg(amount) average_amount
        FROM paysim GROUP BY behavior ORDER BY fraud_rate DESC
    """,
    "destination_balance_behavior": """
        SELECT CASE WHEN nameDest LIKE 'M%' THEN 'merchant_destination'
                    WHEN oldbalanceDest = 0 AND newbalanceDest = 0 THEN 'zero_unchanged'
                    WHEN newbalanceDest < oldbalanceDest THEN 'balance_decreased'
                    WHEN newbalanceDest = oldbalanceDest THEN 'balance_unchanged'
                    ELSE 'balance_increased' END behavior,
               count(*) transactions, sum(isFraud) fraud_transactions,
               avg(isFraud) fraud_rate, avg(amount) average_amount
        FROM paysim GROUP BY behavior ORDER BY fraud_rate DESC
    """,
    "velocity": """
        WITH velocities AS (
            SELECT nameOrig, step, count(*) transaction_velocity,
                   max(isFraud) has_fraud, sum(amount) step_amount
            FROM paysim GROUP BY nameOrig, step
        )
        SELECT CASE WHEN transaction_velocity = 1 THEN '1'
                    WHEN transaction_velocity BETWEEN 2 AND 3 THEN '2-3'
                    WHEN transaction_velocity BETWEEN 4 AND 5 THEN '4-5'
                    ELSE '6+' END velocity_band,
               count(*) account_steps, sum(has_fraud) fraudulent_account_steps,
               avg(has_fraud) fraud_rate, avg(step_amount) average_step_amount,
               max(transaction_velocity) maximum_velocity
        FROM velocities GROUP BY velocity_band ORDER BY min(transaction_velocity)
    """,
    "high_risk_accounts": """
        SELECT nameOrig origin_account, count(*) transactions,
               sum(isFraud) fraud_transactions, avg(isFraud) fraud_rate,
               sum(amount) total_amount, sum(amount * isFraud) fraud_amount,
               max(amount) maximum_amount
        FROM paysim GROUP BY nameOrig HAVING sum(isFraud) > 0
        ORDER BY fraud_amount DESC LIMIT 100
    """,
    "repeated_destinations": """
        SELECT nameDest destination_account, count(*) transactions,
               count(DISTINCT nameOrig) distinct_origins,
               sum(isFraud) fraud_transactions, avg(isFraud) fraud_rate,
               sum(amount * isFraud) fraud_amount
        FROM paysim GROUP BY nameDest
        HAVING count(*) > 1
        ORDER BY fraud_transactions DESC, distinct_origins DESC LIMIT 100
    """,
    "large_depletion": """
        SELECT type, count(*) FILTER (WHERE oldbalanceOrg > 0) eligible_transactions,
               count(*) FILTER (WHERE oldbalanceOrg > 0 AND newbalanceOrig = 0) full_depletions,
               sum(isFraud) FILTER (WHERE oldbalanceOrg > 0 AND newbalanceOrig = 0) fraudulent_full_depletions,
               avg(isFraud) FILTER (WHERE oldbalanceOrg > 0 AND newbalanceOrig = 0) full_depletion_fraud_rate
        FROM paysim GROUP BY type ORDER BY full_depletion_fraud_rate DESC NULLS LAST
    """,
}


def run_analysis(csv_path: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    escaped_path = csv_path.resolve().as_posix().replace("'", "''")
    connection.execute(f"CREATE VIEW paysim AS SELECT * FROM read_csv_auto('{escaped_path}', header=true)")
    results = {name: connection.execute(query).fetchdf() for name, query in QUERIES.items()}
    for name, frame in results.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    _create_charts(results, output_dir)
    (output_dir / "business_findings.md").write_text(_findings(results), encoding="utf-8")
    connection.close()
    return results


def _create_charts(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plots = (
        ("fraud_by_type", "type", "fraud_rate", "Fraud rate by transaction type", "type_fraud_rate.png"),
        ("fraud_by_size", "amount_band", "fraud_rate", "Fraud rate by transaction size", "size_fraud_rate.png"),
        ("fraud_by_hour", "hour_of_day", "fraud_rate", "Fraud rate by simulated hour", "hour_fraud_rate.png"),
    )
    for table, x, y, title, filename in plots:
        fig, axis = plt.subplots(figsize=(10, 5))
        sns.barplot(data=results[table], x=x, y=y, color="#d9485f", ax=axis)
        axis.set_title(title, weight="bold")
        axis.set_ylabel("Fraud rate")
        axis.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def _findings(results: dict[str, pd.DataFrame]) -> str:
    summary = results["summary"].iloc[0]
    by_type = results["fraud_by_type"].set_index("type")
    by_size = results["fraud_by_size"].set_index("amount_band")
    hours = results["fraud_by_hour"]
    peak = hours.loc[hours["fraud_rate"].idxmax()]
    depletion = results["origin_balance_behavior"].set_index("behavior")
    velocity = results["velocity"].set_index("velocity_band")
    destinations = results["repeated_destinations"]
    top_destination = destinations.iloc[0]
    return f"""# PaySim business findings

## Executive summary

- PaySim contains **{int(summary.transactions):,} transactions** and **{int(summary.fraud_transactions):,} frauds**, an event fraud rate of **{summary.fraud_rate:.3%}**.
- Fraud represents **{summary.fraud_amount_rate:.3%} of transaction value**. Average fraud value is **{summary.average_fraud_amount:,.2f}**, compared with **{summary.average_amount:,.2f}** overall. Class imbalance makes accuracy unsuitable as the primary model metric.
- Fraud occurs in **TRANSFER and CASH_OUT**, with rates of **{by_type.loc['TRANSFER', 'fraud_rate']:.3%}** and **{by_type.loc['CASH_OUT', 'fraud_rate']:.3%}**. Their different base rates justify transaction-type-specific features and decision thresholds; the remaining types have no labelled fraud in this simulation and must not be interpreted as inherently safe in production.
- The **1M+** band has a **{by_size.loc['05_1M+', 'fraud_rate']:.3%}** fraud rate. Amount is strongly informative, but a raw cutoff would create avoidable false positives; it should interact with type, balance depletion, and account history.
- Simulated hour **{int(peak.hour_of_day):02d}:00** has the highest fraud rate at **{peak.fraud_rate:.3%}**, but only **{int(peak.transactions):,} transactions** occur in that bucket. This is a low-volume simulation effect, so time should be encoded cyclically and monitored rather than converted into a hard blocking rule.
- Transactions that fully deplete a funded origin have a **{depletion.loc['fully_depleted', 'fraud_rate']:.3%}** fraud rate. Balance depletion is a high-value behavioral feature, while zero starting balances require separate treatment because PaySim includes balance-recording irregularities.
- Within-step origin velocity is almost absent: **{int(velocity.loc['1', 'account_steps']):,}** account-steps contain one transaction and only **{int(velocity.loc['2-3', 'account_steps']):,}** contain two or three. Production velocity features therefore need event timestamps and rolling windows; PaySim's hourly step is too coarse for strong conclusions.
- Repeated destinations can aggregate many distinct origins. The leading fraud-linked repeated destination receives **{int(top_destination.transactions)} transactions from {int(top_destination.distinct_origins)} origins**, including **{int(top_destination.fraud_transactions)} frauds**. Destination fan-in is useful context, but repeated use alone is not a fraud verdict.

## Feature and policy implications

1. Train and calibrate with stratification by transaction type; compare global and type-specific thresholds.
2. Add amount bands, log amount, amount-to-origin-balance ratio, full-depletion flags, and origin/destination balance deltas.
3. Add rolling origin velocity and amount features over multiple windows, not only the current PaySim step.
4. Add destination fan-in, repeated-origin-to-destination interactions, and prior fraud/alert history.
5. Keep merchant destinations distinct from customer destinations because their recorded balance behavior differs structurally.
6. Evaluate with precision-recall AUC, recall at review capacity, false-positive rate, captured fraud value, and investigation workload—not accuracy alone.
7. Treat PaySim labels and zero-fraud transaction types as simulation artifacts and document the production monitoring risk.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full PaySim business EDA")
    parser.add_argument("--csv", type=Path, default=Path("data/raw/paysim.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/eda"))
    args = parser.parse_args()
    run_analysis(args.csv, args.output)
    print(f"EDA outputs written to {args.output}")


if __name__ == "__main__":
    main()
