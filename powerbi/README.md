# FinShield Power BI reporting pack

Generate files with `python -m scripts.export_reporting`. The command reads the
complete local analytics warehouse; it does not export the 168-row UI sample.
The frontend also downloads these same daily and channel aggregate tables.

## Quick report

1. Power BI Desktop → Get data → Text/CSV → `outputs/powerbi/daily.csv` (name it **Daily**).
2. Import `segments.csv` as **Channels**. Keep these aggregate tables unrelated.
3. Copy the first measure block from `Measures.dax`, one measure at a time.
4. Overview: cards for Transactions, Fraud Event Rate, Labeled Fraud Value, Rule Alerts;
   line chart with simulation_day and transactions; line chart with simulation_day
   and fraud_value; bar chart of Channels[type] and Channels[fraud_count].
5. Loss trends: simulation_day, Labeled Fraud Value, Transaction Volume, Fraud Loss Rate.
6. Risk segmentation: channel, fraud_count, fraud_value, alerts and transaction counts.

## PostgreSQL star model

Ingest the source using `python -m src.data.ingestion`, then apply
`sql/star_schema.sql` to that database. Import the analytics views via the PostgreSQL
connector. Use single-direction, one-to-many relationships:

| Dimension key | Fact foreign key |
|---|---|
| dim_customer.customer_key | fact_transaction.customer_key |
| dim_merchant.merchant_key | fact_transaction.merchant_key |
| dim_date.date_key | fact_transaction.date_key |
| dim_geography.geography_key | fact_transaction.geography_key |

For destination analysis, use a second role-playing copy of dim_customer.
Name the fact **FactTransaction** for the second DAX measure block. A merchant
dimension key is null when the destination is a customer account. Geography is
UNKNOWN for every record; no map is meaningful. PaySim steps are simulated hours,
so use simulation_day instead of presenting them as real calendar dates.

The baseline is TRANSFER/CASH_OUT with amount ≥200,000; it is independent of
the isFraud label. The exported labeled fraud value is retrospective exposure,
not evidence of prevented loss. A native `.pbix` has not been authored or verified.
