CREATE TABLE IF NOT EXISTS customers (
    customer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    account_id VARCHAR(32) NOT NULL UNIQUE,
    account_type VARCHAR(16) NOT NULL CHECK (account_type IN ('CUSTOMER', 'MERCHANT')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    step INTEGER NOT NULL CHECK (step >= 1),
    type VARCHAR(16) NOT NULL CHECK (type IN ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')),
    amount NUMERIC(20, 2) NOT NULL CHECK (amount >= 0),
    origin_account_id VARCHAR(32) NOT NULL REFERENCES customers(account_id),
    origin_old_balance NUMERIC(20, 2) NOT NULL CHECK (origin_old_balance >= 0),
    origin_new_balance NUMERIC(20, 2) NOT NULL CHECK (origin_new_balance >= 0),
    destination_account_id VARCHAR(32) NOT NULL REFERENCES customers(account_id),
    destination_old_balance NUMERIC(20, 2) NOT NULL CHECK (destination_old_balance >= 0),
    destination_new_balance NUMERIC(20, 2) NOT NULL CHECK (destination_new_balance >= 0),
    is_fraud BOOLEAN NOT NULL,
    is_flagged_fraud BOOLEAN NOT NULL,
    source_row_number BIGINT NOT NULL,
    source_row_hash CHAR(64) NOT NULL UNIQUE,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (origin_account_id <> destination_account_id)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    transaction_id BIGINT NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    risk_score NUMERIC(8, 7) NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    predicted_fraud BOOLEAN NOT NULL,
    threshold NUMERIC(8, 7) NOT NULL CHECK (threshold BETWEEN 0 AND 1),
    feature_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    explanation JSONB NOT NULL DEFAULT '{}'::jsonb,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (transaction_id, model_name, model_version)
);

CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    transaction_id BIGINT NOT NULL REFERENCES transactions(transaction_id),
    prediction_id BIGINT REFERENCES model_predictions(prediction_id),
    severity VARCHAR(12) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'ASSIGNED', 'CLOSED', 'ESCALATED')),
    rule_hits JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS investigations (
    investigation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    case_id VARCHAR(50) NOT NULL UNIQUE,
    transaction_id BIGINT NOT NULL REFERENCES transactions(transaction_id),
    alert_id BIGINT REFERENCES fraud_alerts(alert_id),
    risk_score NUMERIC(6, 3) NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    assigned_investigator VARCHAR(150),
    status VARCHAR(24) NOT NULL DEFAULT 'NEW' CHECK (status IN ('NEW', 'ASSIGNED', 'INVESTIGATING', 'CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'CLOSED')),
    notes TEXT NOT NULL DEFAULT '',
    decision VARCHAR(24) CHECK (decision IN ('CONFIRMED_FRAUD', 'FALSE_POSITIVE')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS investigator_actions (
    action_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    investigation_id BIGINT NOT NULL REFERENCES investigations(investigation_id) ON DELETE CASCADE,
    actor VARCHAR(150) NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    notes TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS investigation_reports (
    report_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    investigation_id BIGINT REFERENCES investigations(investigation_id) ON DELETE CASCADE,
    case_reference VARCHAR(50) NOT NULL,
    transaction_id BIGINT REFERENCES transactions(transaction_id),
    version INTEGER NOT NULL DEFAULT 1,
    status VARCHAR(16) NOT NULL DEFAULT 'DRAFT' CHECK (status IN ('DRAFT', 'APPROVED', 'SUPERSEDED')),
    analyst VARCHAR(150),
    report_content TEXT NOT NULL,
    evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    generator VARCHAR(100) NOT NULL,
    content_sha256 CHAR(64) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL,
    approved_at TIMESTAMPTZ,
    UNIQUE (case_reference, version)
);

CREATE INDEX IF NOT EXISTS idx_reports_case_version ON investigation_reports(case_reference, version DESC);

CREATE TABLE IF NOT EXISTS investigator_feedback (
    feedback_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    case_id VARCHAR(50) NOT NULL UNIQUE,
    transaction_id BIGINT REFERENCES transactions(transaction_id),
    prediction_id BIGINT REFERENCES model_predictions(prediction_id),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    fraud_probability NUMERIC(8, 7) NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    risk_score NUMERIC(6, 3) NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    investigator VARCHAR(150) NOT NULL,
    final_outcome VARCHAR(24) NOT NULL CHECK (final_outcome IN ('CONFIRMED_FRAUD', 'FALSE_POSITIVE')),
    final_label SMALLINT NOT NULL CHECK (final_label IN (0, 1)),
    feature_snapshot JSONB NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_outcome_created ON investigator_feedback(final_outcome, created_at);

CREATE TABLE IF NOT EXISTS api_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    method VARCHAR(12) NOT NULL,
    endpoint TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms NUMERIC(12, 3) NOT NULL,
    error_type VARCHAR(150)
);

CREATE TABLE IF NOT EXISTS prediction_events (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    transaction_id VARCHAR(50) NOT NULL,
    transaction_type VARCHAR(16) NOT NULL,
    amount NUMERIC(20, 2) NOT NULL,
    fraud_probability NUMERIC(8, 7) NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    risk_score NUMERIC(6, 3) NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    decision VARCHAR(24) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_api_events_time ON api_events(occurred_at);
CREATE INDEX IF NOT EXISTS idx_prediction_events_time ON prediction_events(occurred_at);

CREATE INDEX IF NOT EXISTS idx_transactions_origin_step ON transactions(origin_account_id, step);
CREATE INDEX IF NOT EXISTS idx_transactions_destination_step ON transactions(destination_account_id, step);
CREATE INDEX IF NOT EXISTS idx_transactions_fraud ON transactions(is_fraud) WHERE is_fraud;
CREATE INDEX IF NOT EXISTS idx_alerts_queue ON fraud_alerts(status, severity, created_at);
CREATE INDEX IF NOT EXISTS idx_investigations_assignee ON investigations(assigned_investigator, status);
CREATE INDEX IF NOT EXISTS idx_investigations_queue ON investigations(status, risk_score DESC, created_at);
