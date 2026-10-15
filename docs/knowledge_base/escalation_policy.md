# Fraud Alert Escalation Policy

Version: 1.0

## Decision bands

- 0 through 30: APPROVE. Retain the score and evidence for monitoring.
- Above 30 through 60: MONITOR. Apply enhanced monitoring and review repeated activity.
- Above 60 through 80: MANUAL REVIEW. Pause automated resolution and assign an investigator.
- Above 80 through 100: BLOCK. Hold the transaction and escalate immediately.

Thresholds are configurable and must ultimately be validated against loss, review capacity, customer friction, and calibrated production data.

## Immediate escalation indicators

Escalate urgently when the final score is above 80, when strong evidence indicates rapid full-balance evacuation, or when multiple high-severity signals combine with destination or authentication risk. A high-severity rule alone does not automatically confirm fraud.

## Investigator actions

1. Preserve the transaction, feature, rule, model, and explanation snapshots.
2. Review origin history and destination relationships.
3. Collect device, IP, authentication, channel, beneficiary, and customer-contact evidence when available.
4. Check prior alerts and linked investigations.
5. Record a reasoned disposition: confirmed fraud, false positive, or inconclusive.

## Customer-friction safeguard

Before maintaining a block, consider trusted-beneficiary history, known customer behavior, recent legitimate balance changes, data-quality issues, and successful step-up verification. Document any override and the evidence supporting it.

## Demonstration limitation

FinShield policies are portfolio examples, not legal or regulatory advice. Production escalation procedures require approval from the institution's fraud, compliance, legal, and operations owners.
