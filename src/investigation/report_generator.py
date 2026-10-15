"""Grounded, editable investigation report generation and versioned saving."""

from __future__ import annotations

import hashlib
import json
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.rag.knowledge_base import FraudKnowledgeBase


@dataclass(frozen=True)
class InvestigationReportDraft:
    case_id: str
    transaction_id: str
    content: str
    generated_at: str
    generator: str
    evidence_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InvestigationReportGenerator:
    def __init__(self, knowledge_base: FraudKnowledgeBase | None = None) -> None:
        self.knowledge_base = knowledge_base or FraudKnowledgeBase.default()

    def generate(
        self, case: Mapping[str, Any], decision, explanation,
        context: Mapping[str, Any], analyst: str = "Pending",
    ) -> InvestigationReportDraft:
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        case_id = _formal_case_id(str(case["Case"]))
        profile = context["origin_profile"].iloc[0]
        destination = context["destination_profile"].iloc[0]
        old_balance = float(case["Origin balance before"])
        new_balance = float(case["Origin balance after"])
        depletion = (old_balance - new_balance) / old_balance if old_balance else 0.0
        action = _recommended_action(decision.final_risk_score)
        policy = self.knowledge_base.retrieve(
            f"What policy and escalation guidance applies to a {decision.decision} transaction with score {decision.final_risk_score}?",
            top_k=4,
        )
        rule_lines = "\n".join(f"- {hit['rule']} ({hit['severity']}, +{hit['score']} points)" for hit in decision.rule_hits) or "- None"
        factor_lines = "\n".join(
            f"- {factor.description}: value={factor.value}, SHAP contribution={factor.contribution:+.4f}"
            for factor in explanation.primary_contributing_factors
        ) or "- No positive SHAP factors returned"
        policy_lines = "\n".join(
            f"- {item.source}: {textwrap.shorten(item.content, width=260, placeholder='...')} [{item.evidence_id}]"
            for item in policy
        )
        related = context["related"]
        related_summary = (
            f"{len(related)} related origin accounts were retrieved for destination {case['Destination']}. "
            f"The destination has {int(destination.incoming_transactions)} incoming transactions from "
            f"{int(destination.unique_origins)} unique origins and {int(destination.labelled_frauds)} PaySim-labelled frauds."
        )
        content = f"""CASE: {case_id}
TRANSACTION: {case['Transaction']}

Risk Level:
{_risk_level(decision.final_risk_score)}

Fraud Probability:
{explanation.fraud_risk:.1%}

Final Risk Score:
{decision.final_risk_score:.2f}/100

Triggered Rules:
{rule_lines}

Transaction Summary:
{case['Type']} transaction for INR {float(case['Amount']):,.2f} from {case['Origin']} to {case['Destination']} at PaySim step {int(case['Step'])}. Origin balance changed from INR {old_balance:,.2f} to INR {new_balance:,.2f}.

Behavior Analysis:
The transaction depleted {depletion:.1%} of the recorded origin balance. The origin has {int(profile.lifetime_transactions)} observed transaction(s), an average amount of INR {float(profile.average_amount):,.2f}, and {int(profile.unique_destinations)} unique destination(s). {len(context['history'])} prior origin transaction(s) were retrieved.

Risk Indicators:
{rule_lines}

Related Transactions:
{related_summary}

Model Explanation:
{factor_lines}

Applicable Policy:
{policy_lines}

Recommended Action:
{action}

Analyst:
{analyst.strip() or 'Pending'}

Generated:
{generated_at}

Human Review Status:
DRAFT - Requires analyst verification and approval before final disposition.
"""
        evidence_ids = [item.evidence_id for item in policy]
        return InvestigationReportDraft(case_id, str(case["Transaction"]), content, generated_at, "grounded-template+vector-policy", evidence_ids)


def save_report(draft: InvestigationReportDraft, edited_content: str, output_dir: Path, analyst: str) -> Path:
    if not edited_content.strip():
        raise ValueError("report content cannot be empty")
    if not analyst.strip() or analyst.strip().lower() == "pending":
        raise ValueError("an analyst name is required before saving")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"{draft.case_id}_{timestamp}.md"
    report_path.write_text(edited_content, encoding="utf-8")
    metadata = {
        **draft.to_dict(),
        "analyst": analyst.strip(),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_sha256": hashlib.sha256(edited_content.encode("utf-8")).hexdigest(),
        "report_path": str(report_path),
    }
    report_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return report_path


def _formal_case_id(case_id: str) -> str:
    digits = "".join(character for character in case_id if character.isdigit()) or "0"
    return f"FIN-{datetime.now(timezone.utc).year}-{int(digits):05d}"


def _risk_level(score: float) -> str:
    if score > 80:
        return "CRITICAL"
    if score > 60:
        return "HIGH"
    if score > 30:
        return "MEDIUM"
    return "LOW"


def _recommended_action(score: float) -> str:
    if score > 80:
        return "BLOCK AND ESCALATE FOR IMMEDIATE INVESTIGATION"
    if score > 60:
        return "MANUAL REVIEW"
    if score > 30:
        return "MONITOR WITH ENHANCED CONTROLS"
    return "APPROVE AND RETAIN FOR MONITORING"
