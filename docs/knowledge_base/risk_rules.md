# FinShield Risk Rules Reference

Version: 1.0  
Scope: Investigator decision support for the FinShield demonstration environment.

## RULE_001 - Large transfer

Trigger when a `TRANSFER` equals or exceeds the configured INR 200,000 threshold. Default severity is HIGH and the rule contributes 20 points. Investigators should verify whether the amount is normal for the origin account and whether the destination is trusted. Amount alone is not sufficient to confirm fraud.

## RULE_002 - Origin balance depleted

Trigger when an outgoing transaction leaves a previously funded origin account at or near zero. Default severity is MEDIUM and the rule contributes 15 points. Check balance-recording quality, recent inbound funding, and whether depletion matches prior customer behavior.

## RULE_003 - Full balance transfer

Trigger when the outgoing amount approximately equals the complete origin balance and the ending balance is near zero. Default severity is HIGH and the rule contributes 25 points. This pattern is associated with account takeover and rapid fund evacuation, but may also represent a legitimate account closure or treasury movement.

## RULE_004 - High transaction velocity

Trigger when origin activity reaches the configured velocity threshold within the scoring window. Default severity is HIGH and the rule contributes 20 points. Collect exact timestamps, channel, device, and authentication events because PaySim steps do not provide production-grade velocity resolution.

## RULE_005 - Repeated destination

Trigger when a destination appears at least three times in prior transaction history for the evaluated context. Default severity is MEDIUM and the rule contributes 15 points. Investigate destination fan-in, shared origin relationships, and prior alerts; repeated use can also be legitimate.

## RULE_006 - Unusual account amount

Trigger when the amount is at least three times the origin account average after a minimum of three prior observations. Default severity is MEDIUM and the rule contributes 15 points. Confirm that history is sufficiently representative and check seasonal or customer-specific explanations.

## Rule interpretation

Rules provide evidence, not a fraud label. Correlated rules may describe the same underlying behavior. Investigators must use the unified score, model explanation, account context, and policy before disposition.
