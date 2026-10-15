"""Build a bounded, auditable evidence packet for one investigation case."""

from __future__ import annotations

from typing import Any, Mapping

from src.rag.retriever import EvidenceDocument


def build_case_evidence(
    case: Mapping[str, Any], decision, explanation, context: Mapping[str, Any],
    recommended_action: str,
) -> list[EvidenceDocument]:
    old_balance = float(case["Origin balance before"])
    new_balance = float(case["Origin balance after"])
    depletion = (old_balance - new_balance) / old_balance if old_balance else 0.0
    profile = context["origin_profile"].iloc[0]
    destination = context["destination_profile"].iloc[0]
    documents = [
        EvidenceDocument(
            "E-TX", "transaction",
            f"Transaction {case['Transaction']} is a {case['Type']} for INR {float(case['Amount']):,.2f} from {case['Origin']} to {case['Destination']} at PaySim step {int(case['Step'])}.",
        ),
        EvidenceDocument(
            "E-RISK", "risk",
            f"Final risk score is {decision.final_risk_score:.2f}/100. ML fraud probability is {explanation.fraud_risk:.2%}. The operational decision is {decision.decision}.",
        ),
        EvidenceDocument(
            "E-BALANCE", "behavior",
            f"Origin balance changed from INR {old_balance:,.2f} to INR {new_balance:,.2f}, a depletion of {depletion:.2%}. Destination balance changed from INR {float(case['Destination balance before']):,.2f} to INR {float(case['Destination balance after']):,.2f}.",
        ),
        EvidenceDocument(
            "E-HISTORY", "history",
            f"Origin account has {int(profile.lifetime_transactions)} observed transaction(s), average amount INR {float(profile.average_amount):,.2f}, maximum amount INR {float(profile.maximum_amount):,.2f}, and {int(profile.unique_destinations)} unique destination(s). There are {len(context['history'])} transactions before this event in the retrieved history.",
        ),
        EvidenceDocument(
            "E-DEST", "relationships",
            f"Destination account has {int(destination.incoming_transactions)} incoming transaction(s) from {int(destination.unique_origins)} unique origin(s), including {int(destination.labelled_frauds)} PaySim-labelled fraud transaction(s). {len(context['related'])} related origin account(s) were retrieved.",
        ),
        EvidenceDocument(
            "E-RECOMMEND", "recommendation",
            f"Recommended action: {recommended_action.rstrip('.')}.",
        ),
    ]
    for index, hit in enumerate(decision.rule_hits, start=1):
        documents.append(EvidenceDocument(
            f"E-RULE-{index}", "rules",
            f"Triggered rule {hit['rule']} ({hit['severity']}, {hit['score']} points): {hit['reason']}",
        ))
    for index, factor in enumerate(explanation.primary_contributing_factors, start=1):
        documents.append(EvidenceDocument(
            f"E-SHAP-{index}", "model_explanation",
            f"SHAP risk factor: {factor.description}; feature value {factor.value}; model contribution {factor.contribution:+.4f}.",
        ))
    return documents
