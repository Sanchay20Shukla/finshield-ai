"""Evidence-grounded AI fraud investigation copilot."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from src.rag.retriever import EvidenceDocument, EvidenceRetriever
from src.rag.knowledge_base import FraudKnowledgeBase


SYSTEM_INSTRUCTIONS = """You are FinShield's fraud investigation copilot.
Answer only from the supplied evidence. Never invent account history, timing,
rules, model values, customer identity, or causal claims. If evidence is absent,
say it is unavailable. Be concise. Cite supporting evidence IDs in square brackets.
End with a recommended action only when the evidence packet contains one."""


@dataclass(frozen=True)
class CopilotAnswer:
    answer: str
    evidence_ids: list[str]
    provider: str
    grounded: bool = True

    def to_dict(self):
        return asdict(self)


class FraudInvestigationCopilot:
    def __init__(self, retriever: EvidenceRetriever | None = None, knowledge_base: FraudKnowledgeBase | None = None) -> None:
        self.retriever = retriever or EvidenceRetriever()
        self.knowledge_base = knowledge_base

    def answer(self, question: str, documents: list[EvidenceDocument]) -> CopilotAnswer:
        if not question.strip():
            raise ValueError("question cannot be empty")
        retrieved = self.retriever.retrieve(question, documents)
        if self.knowledge_base is not None:
            retrieved.extend(self.knowledge_base.retrieve(question, top_k=5))
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL")
        if api_key and model:
            return self._openai_answer(question, retrieved, api_key, model)
        return self._grounded_fallback(retrieved)

    def _openai_answer(self, question: str, evidence: list[EvidenceDocument], api_key: str, model: str) -> CopilotAnswer:
        from openai import OpenAI

        packet = "\n".join(f"[{item.evidence_id}] {item.content}" for item in evidence)
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=f"Investigator question: {question}\n\nRetrieved evidence:\n{packet}",
        )
        return CopilotAnswer(response.output_text, [item.evidence_id for item in evidence], f"openai:{model}")

    def _grounded_fallback(self, evidence: list[EvidenceDocument]) -> CopilotAnswer:
        by_category = {item.category: item for item in evidence}
        transaction = by_category.get("transaction")
        risk = by_category.get("risk")
        recommendation = by_category.get("recommendation")
        observations = [
            item for item in evidence
            if item.category in {"behavior", "rules", "model_explanation", "relationships", "history", "policy"}
        ][:7]
        heading = transaction.content if transaction else "The requested transaction was retrieved."
        lines = [f"**{heading}**"]
        if risk:
            lines.extend(["", risk.content])
        lines.extend(["", "Key observations:"])
        if observations:
            lines.extend(f"{index}. {item.content} [{item.evidence_id}]" for index, item in enumerate(observations, 1))
        else:
            lines.append("1. No additional supporting evidence was retrieved.")
        if recommendation:
            lines.extend(["", f"**{recommendation.content}** [{recommendation.evidence_id}]"])
        return CopilotAnswer("\n".join(lines), [item.evidence_id for item in evidence], "grounded-local")
