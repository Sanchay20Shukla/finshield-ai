"""Export full-dataset aggregate reporting files and provenance."""
import csv
import json
from pathlib import Path

from src.data.warehouse import ROOT, connect, records


def export(output: Path = ROOT / "outputs/powerbi"):
    output.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        for table in ["daily", "segments"]:
            rows = records(db, f"SELECT * FROM {table}")
            with (output / f"{table}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (output / "provenance.json").write_text(db.execute("SELECT payload FROM metadata").fetchone()[0], encoding="utf-8")
    print(json.dumps({"output": str(output), "tables": ["daily", "segments"]}))


if __name__ == "__main__":
    export()
