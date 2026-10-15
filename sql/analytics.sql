-- Fraud rate and value by transaction type.
SELECT type,
       count(*) AS transaction_count,
       count(*) FILTER (WHERE is_fraud) AS fraud_count,
       avg(is_fraud::int) AS fraud_rate,
       sum(amount) FILTER (WHERE is_fraud) AS fraud_amount
FROM transactions
GROUP BY type
ORDER BY fraud_rate DESC;

-- Origin account velocity within each PaySim step.
SELECT origin_account_id, step, count(*) AS transaction_velocity, sum(amount) AS total_amount
FROM transactions
GROUP BY origin_account_id, step
HAVING count(*) > 1
ORDER BY transaction_velocity DESC, total_amount DESC;

-- Repeated destinations and their origin fan-in.
SELECT destination_account_id, count(*) AS transaction_count,
       count(DISTINCT origin_account_id) AS distinct_origins,
       count(*) FILTER (WHERE is_fraud) AS fraud_count
FROM transactions
GROUP BY destination_account_id
HAVING count(*) > 1
ORDER BY fraud_count DESC, distinct_origins DESC;
