"""Chunked, idempotent PaySim-to-PostgreSQL ingestion pipeline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

from src.data.preprocessing import customer_records, transform_transactions
from src.data.validation import REQUIRED_COLUMNS, validate_batch


@dataclass
class IngestionStats:
    rows_read: int = 0
    rows_valid: int = 0
    rows_rejected: int = 0
    rows_inserted: int = 0
    duplicate_rows: int = 0


TRANSACTION_COLUMNS = (
    "step", "type", "amount", "origin_account_id", "origin_old_balance",
    "origin_new_balance", "destination_account_id", "destination_old_balance",
    "destination_new_balance", "is_fraud", "is_flagged_fraud",
    "source_row_number", "source_row_hash",
)


def apply_schema(connection: psycopg.Connection, schema_path: Path) -> None:
    with connection.cursor() as cursor:
        cursor.execute(schema_path.read_text(encoding="utf-8"))
    connection.commit()


def _upsert_customers(connection: psycopg.Connection, customers: pd.DataFrame) -> None:
    with connection.cursor() as cursor:
        cursor.executemany(
            """INSERT INTO customers (account_id, account_type)
               VALUES (%s, %s)
               ON CONFLICT (account_id) DO NOTHING""",
            customers.itertuples(index=False, name=None),
        )


def _copy_transactions(connection: psycopg.Connection, frame: pd.DataFrame) -> int:
    before = connection.execute("SELECT count(*) FROM transactions").fetchone()[0]
    with connection.cursor() as cursor:
        cursor.execute("CREATE TEMP TABLE transaction_batch (LIKE transactions INCLUDING DEFAULTS) ON COMMIT DROP")
        columns_sql = ", ".join(TRANSACTION_COLUMNS)
        with cursor.copy(f"COPY transaction_batch ({columns_sql}) FROM STDIN") as copy:
            for row in frame[list(TRANSACTION_COLUMNS)].itertuples(index=False, name=None):
                copy.write_row(row)
        cursor.execute(
            f"""INSERT INTO transactions ({columns_sql})
                SELECT {columns_sql} FROM transaction_batch
                ON CONFLICT (source_row_hash) DO NOTHING"""
        )
    after = connection.execute("SELECT count(*) FROM transactions").fetchone()[0]
    return after - before


def ingest_csv(csv_path: Path, database_url: str, schema_path: Path, chunk_size: int = 100_000) -> IngestionStats:
    if not csv_path.is_file():
        raise FileNotFoundError(f"PaySim CSV not found: {csv_path}")
    stats = IngestionStats()
    with psycopg.connect(database_url) as connection:
        apply_schema(connection, schema_path)
        for raw in pd.read_csv(csv_path, chunksize=chunk_size, dtype={"type": "string", "nameOrig": "string", "nameDest": "string"}):
            result = validate_batch(raw)
            transformed = transform_transactions(result.valid, row_offset=stats.rows_read)
            stats.rows_read += len(raw)
            stats.rows_valid += len(transformed)
            stats.rows_rejected += len(result.rejected)
            stats.duplicate_rows += result.issue_counts.get("duplicate_in_batch", 0)
            if transformed.empty:
                continue
            _upsert_customers(connection, customer_records(transformed))
            inserted = _copy_transactions(connection, transformed)
            stats.rows_inserted += inserted
            stats.duplicate_rows += len(transformed) - inserted
            connection.commit()
    return stats


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest PaySim into FinShield PostgreSQL")
    parser.add_argument("--csv", type=Path, default=Path(os.getenv("PAYSIM_CSV_PATH", "data/raw/paysim.csv")))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("INGESTION_CHUNK_SIZE", "100000")))
    args = parser.parse_args()
    if not args.database_url:
        parser.error("DATABASE_URL is required")
    project_root = Path(__file__).resolve().parents[2]
    stats = ingest_csv(args.csv, args.database_url, project_root / "sql" / "schema.sql", args.chunk_size)
    print(json.dumps(stats.__dict__, indent=2))


if __name__ == "__main__":
    main()
