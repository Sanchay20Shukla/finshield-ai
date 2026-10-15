"""Retrieve the most relevant evidence without exposing unrelated case data."""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.embeddings import cosine_similarity, embed_text


@dataclass(frozen=True)
class EvidenceDocument:
    evidence_id: str
    category: str
    content: str
    source: str | None = None


class EvidenceRetriever:
    def retrieve(self, question: str, documents: list[EvidenceDocument], top_k: int = 8) -> list[EvidenceDocument]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = embed_text(question)
        scored = sorted(
            documents,
            key=lambda document: cosine_similarity(query, embed_text(document.content)),
            reverse=True,
        )
        # Core decision evidence is always included, even when the question uses different wording.
        mandatory = [document for document in documents if document.category in {"transaction", "risk", "recommendation"}]
        selected: list[EvidenceDocument] = []
        for document in mandatory + scored:
            if document not in selected:
                selected.append(document)
            if len(selected) >= top_k:
                break
        return selected
