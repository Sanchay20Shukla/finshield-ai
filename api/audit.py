"""Optional durable PostgreSQL audit for successful inference requests."""
import os


DDL = """CREATE TABLE IF NOT EXISTS inference_audit (
    transaction_id TEXT PRIMARY KEY,
    model_version TEXT NOT NULL,
    request JSONB NOT NULL,
    response JSONB NOT NULL,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT now()
)"""


def record_prediction(request, response, model_version):
    url = os.getenv("DATABASE_URL")
    if not url:
        return
    import psycopg
    from psycopg.types.json import Jsonb
    # If configured persistence fails, do not silently report a durable success.
    with psycopg.connect(url, connect_timeout=5) as connection:
        connection.execute(DDL)
        connection.execute(
            "INSERT INTO inference_audit(transaction_id, model_version, request, response) VALUES (%s,%s,%s,%s)",
            (response.transaction_id, model_version, Jsonb(request.model_dump()), Jsonb(response.model_dump())),
        )
