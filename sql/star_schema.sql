-- PostgreSQL reporting star over idempotently ingested PaySim events.
-- Use simulation hour/day, not invented calendar dates. Geography is unknown.
CREATE SCHEMA IF NOT EXISTS analytics;

CREATE OR REPLACE VIEW analytics.dim_customer AS
SELECT customer_id AS customer_key, account_id, account_type FROM customers;

CREATE OR REPLACE VIEW analytics.dim_merchant AS
SELECT customer_id AS merchant_key, account_id FROM customers WHERE account_type='MERCHANT';

CREATE OR REPLACE VIEW analytics.dim_geography AS
SELECT 0 AS geography_key, 'UNKNOWN'::text AS country, 'Not supplied by PaySim'::text AS description;

CREATE OR REPLACE VIEW analytics.dim_date AS
SELECT DISTINCT step AS date_key, step AS simulation_hour,
       ((step-1)/24)+1 AS simulation_day, (step-1)%24 AS hour_of_day FROM transactions;

CREATE OR REPLACE VIEW analytics.fact_transaction AS
SELECT t.transaction_id, t.step AS date_key, origin.customer_id AS customer_key,
       destination.customer_id AS destination_customer_key,
       CASE WHEN destination.account_type='MERCHANT' THEN destination.customer_id END AS merchant_key,
       0 AS geography_key, t.type, t.amount, t.is_fraud, t.is_flagged_fraud,
       t.type IN ('TRANSFER','CASH_OUT') AND t.amount>=200000 AS rule_alert,
       t.origin_old_balance, t.origin_new_balance,
       t.destination_old_balance, t.destination_new_balance
FROM transactions t
JOIN customers origin ON origin.account_id=t.origin_account_id
JOIN customers destination ON destination.account_id=t.destination_account_id;

CREATE OR REPLACE VIEW analytics.daily_risk AS
SELECT d.simulation_day, count(*) AS transactions, sum(f.amount) AS volume,
       count(*) FILTER (WHERE f.is_fraud) AS fraud_count,
       coalesce(sum(f.amount) FILTER (WHERE f.is_fraud),0) AS fraud_value,
       count(*) FILTER (WHERE f.rule_alert) AS alerts
FROM analytics.fact_transaction f JOIN analytics.dim_date d ON d.date_key=f.date_key
GROUP BY d.simulation_day;
