"""Document chunking and persistent local vector retrieval for fraud policy."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from src.rag.embeddings import hash_embedding, vector_cosine
from src.rag.retriever import EvidenceDocument


SUPPORTED_DOCUMENTS = (
    "fraud_policy.pdf", "investigation_guidelines.pdf", "risk_rules.md",
    "transaction_typologies.md", "escalation_policy.md",
)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    heading: str
    content: str


class FraudKnowledgeBase:
    def __init__(self, document_dir: Path, database_path: Path) -> None:
        self.document_dir = document_dir
        self.database_path = database_path

    @classmethod
    def default(cls) -> "FraudKnowledgeBase":
        root = Path(__file__).resolve().parents[2]
        return cls(root / "docs" / "knowledge_base", root / "data" / "processed" / "rag" / "fraud_kb.sqlite3")

    def build(self, force: bool = False) -> int:
        paths = [self.document_dir / name for name in SUPPORTED_DOCUMENTS]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"knowledge base documents missing: {missing}")
        fingerprint = hashlib.sha256("".join(_file_hash(path) for path in paths).encode()).hexdigest()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("""CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, source TEXT NOT NULL, heading TEXT NOT NULL,
                content TEXT NOT NULL, embedding TEXT NOT NULL
            )""")
            current = connection.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
            if current and current[0] == fingerprint and not force:
                return connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
            connection.execute("DELETE FROM chunks")
            chunks = [chunk for path in paths for chunk in chunk_document(path)]
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [(chunk.chunk_id, chunk.source, chunk.heading, chunk.content, json.dumps(hash_embedding(chunk.content))) for chunk in chunks],
            )
            connection.execute("INSERT OR REPLACE INTO metadata VALUES ('fingerprint', ?)", (fingerprint,))
            connection.commit()
            return len(chunks)

    def retrieve(self, question: str, top_k: int = 5) -> list[EvidenceDocument]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.build()
        query_vector = hash_embedding(question)
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute("SELECT chunk_id, source, heading, content, embedding FROM chunks").fetchall()
        lowered = question.lower()
        def relevance(row) -> float:
            score = vector_cosine(query_vector, json.loads(row[4]))
            source = row[1]
            if "escalat" in lowered and source == "escalation_policy.md":
                score += 0.6
            if any(term in lowered for term in ("evidence", "collect", "notes")) and source == "investigation_guidelines.pdf":
                score += 0.45
            if "rule" in lowered and source == "risk_rules.md":
                score += 0.45
            if "typolog" in lowered and source == "transaction_typologies.md":
                score += 0.45
            if "policy" in lowered and source in {"fraud_policy.pdf", "escalation_policy.md"}:
                score += 0.3
            return score
        ranked = sorted(rows, key=relevance, reverse=True)[:top_k]
        documents = []
        for row in ranked:
            content = row[3].strip()
            heading = row[2].strip()
            display = content if content.lower().startswith(heading.lower()) else f"{heading}: {content}"
            documents.append(EvidenceDocument(f"KB:{row[0]}", "policy", display, row[1]))
        return documents


def chunk_document(path: Path, chunk_size: int = 900, overlap: int = 120) -> list[DocumentChunk]:
    text = _read_document(path)
    sections = [section.strip() for section in re.split(r"(?m)^#+\s+", text) if section.strip()]
    chunks: list[DocumentChunk] = []
    for section in sections:
        heading = section.splitlines()[0][:120] if section.splitlines() else path.stem
        start = 0
        while start < len(section):
            end = min(len(section), start + chunk_size)
            content = section[start:end].strip()
            if content:
                chunks.append(DocumentChunk(f"{path.stem}#{len(chunks) + 1}", path.name, heading, content))
            if end == len(section):
                break
            start = max(start + 1, end - overlap)
    return chunks


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        pages = PdfReader(path).pages
        # The first page is a cover. Retrieval indexes substantive policy pages only.
        substantive = pages[1:] if len(pages) > 1 else pages
        text = "\n\n".join(page.extract_text() or "" for page in substantive)
        text = re.sub(r"FinShield AI - Internal demonstration policy\s+Page \d+", "", text)
        return text.strip()
    return path.read_text(encoding="utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
