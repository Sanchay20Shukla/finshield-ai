"""Generate the two styled policy PDFs used by the local fraud knowledge base."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "knowledge_base"


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#2b4664"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.setFillColor(colors.HexColor("#51677f"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 10 * mm, "FinShield AI - Internal demonstration policy")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(name="Cover", parent=base["Title"], fontName="Helvetica-Bold", fontSize=27, leading=32, textColor=colors.HexColor("#102a43"), alignment=TA_CENTER, spaceAfter=14))
    base.add(ParagraphStyle(name="Kicker", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=colors.HexColor("#008f79"), alignment=TA_CENTER, spaceAfter=8))
    base.add(ParagraphStyle(name="H1x", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=colors.HexColor("#102a43"), spaceBefore=8, spaceAfter=5))
    base.add(ParagraphStyle(name="H2x", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=colors.HexColor("#176b87"), spaceBefore=6, spaceAfter=3))
    base.add(ParagraphStyle(name="Bodyx", parent=base["BodyText"], fontName="Helvetica", fontSize=9, leading=12, textColor=colors.HexColor("#243b53"), spaceAfter=4))
    base.add(ParagraphStyle(name="Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=9.5, leading=13, textColor=colors.HexColor("#8b2635"), backColor=colors.HexColor("#fff1f2"), borderPadding=8, spaceBefore=5, spaceAfter=7))
    return base


def build_pdf(filename: str, title: str, subtitle: str, sections: list[tuple[str, list[tuple[str, str]]]], callout: str):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT / filename), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=18 * mm, bottomMargin=22 * mm, title=title, author="FinShield AI")
    style = styles()
    story = [Spacer(1, 42 * mm), Paragraph("FINSHIELD AI", style["Kicker"]), Paragraph(title, style["Cover"]), Paragraph(subtitle, ParagraphStyle(name="Sub", parent=style["Bodyx"], alignment=TA_CENTER, fontSize=11, leading=16)), Spacer(1, 12 * mm)]
    story.append(Table([["Version", "1.0"], ["Classification", "Internal demonstration"], ["Owner", "Fraud Operations"]], colWidths=[42 * mm, 90 * mm], style=TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f4f2")), ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#243b53")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#b8c8d8")), ("PADDING", (0, 0), (-1, -1), 7)])))
    story.extend([PageBreak(), Paragraph(callout, style["Callout"])])
    for section, items in sections:
        story.append(Paragraph(section, style["H1x"]))
        for heading, body in items:
            story.append(Paragraph(heading, style["H2x"]))
            story.append(Paragraph(body, style["Bodyx"]))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    build_pdf("fraud_policy.pdf", "Fraud Risk Decision Policy", "Controls for explainable hybrid transaction risk decisions", [
        ("Purpose and scope", [("Objective", "Prioritize suspicious digital payments using model probability, deterministic policy rules, behavioral signals, and investigator review. The system supports decisions; it does not establish that fraud occurred."), ("Data boundary", "PaySim is synthetic. Production use requires representative data, calibration, monitoring, access controls, and governance approval.")]),
        ("Decision framework", [("Unified score", "Default weighting is 65 percent ML, 20 percent rules, and 15 percent behavioral risk. Scores 0-30 approve, above 30-60 monitor, above 60-80 manual review, and above 80 block."), ("Evidence standard", "Every decision must retain model version, probability, threshold, feature snapshot, triggered rules, SHAP factors, behavioral signals, final score, and action."), ("Overrides", "Investigators may override an automated action only with a documented reason and supporting evidence. Overrides must remain auditable.")]),
        ("Risk controls", [("False positives", "Review trusted beneficiaries, known customer behavior, balance corrections, and successful step-up verification before maintaining restrictions."), ("False negatives", "Monitor missed-fraud rate, captured fraud value, emerging typologies, threshold performance, and drift by transaction type."), ("Explainability", "SHAP describes model behavior and is not causal proof. Correlated balance features can redistribute attribution.")]),
        ("Governance", [("Review cadence", "Fraud operations should review thresholds, rule hit rates, model calibration, investigation capacity, and customer friction on an approved schedule."), ("Prohibited use", "Do not use PaySim labels, synthetic relationships, or this demonstration policy for autonomous real-world adverse action.")]),
    ], "Policy principle: a high score prioritizes investigation; it is not a substitute for verified evidence.")

    build_pdf("investigation_guidelines.pdf", "Fraud Investigation Guidelines", "Evidence collection and case documentation playbook", [
        ("Case intake", [("Confirm the event", "Verify transaction identifier, amount, type, origin, destination, time, channel, and processing status."), ("Preserve decision evidence", "Capture model probability and version, unified score, rule hits, behavioral signals, SHAP factors, and the exact evidence available at decision time.")]),
        ("Evidence checklist", [("Origin account", "Review prior amounts, velocity, balance changes, beneficiary history, device, IP, authentication, channel, profile changes, and customer contact."), ("Destination account", "Review fan-in, repeated origins, downstream movement, linked alerts, account age, identity, device overlap, and merchant context when available."), ("Transaction context", "Check whether the amount is unusual, whether funds were fully depleted, whether timestamps indicate rapid activity, and whether data-quality issues explain balance errors.")]),
        ("Analysis and disposition", [("Avoid confirmation bias", "Consider legitimate explanations and protective model factors. Do not treat a triggered rule or SHAP factor as proof."), ("Disposition", "Record confirmed fraud, false positive, or inconclusive. Summarize facts, missing evidence, customer impact, action taken, and rationale."), ("Escalation", "Escalate scores above 80 and strong rapid-depletion patterns. For scores above 60, complete manual review before resolution.")]),
        ("Minimum investigation note", [("Required content", "State why the alert was generated, key evidence reviewed, contradictory or missing evidence, related-account findings, customer contact outcome if any, recommended action, and final disposition."), ("Grounding", "AI-generated notes must cite retrieved case or policy evidence and must be reviewed by the investigator before saving.")]),
    ], "Investigation standard: distinguish observed facts, model interpretation, policy guidance, and missing evidence.")


if __name__ == "__main__":
    main()
