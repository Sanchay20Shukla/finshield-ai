# Digital Payment Fraud Typologies

## Account takeover and rapid depletion

An attacker gains account access and quickly transfers or cashes out most available funds. Common signals include a new or unusual destination, complete balance depletion, large deviation from normal amounts, unfamiliar device or channel, and authentication anomalies. Collect login, device, IP, authentication, beneficiary-creation, and customer-contact evidence.

## Mule or funnel destination

A destination receives funds from many unrelated origin accounts and then moves or withdraws them rapidly. Useful evidence includes destination fan-in, repeated destination use, shared devices or contact details, timing concentration, downstream movement, and prior alerts. High fan-in is not conclusive because merchants and aggregators can exhibit similar behavior.

## Transaction laundering or merchant misuse

Payments are processed through a merchant account for activity inconsistent with the stated business. Investigators should compare merchant category, ticket size, refund behavior, customer concentration, geography, and settlement flows. PaySim merchant balance fields are structurally incomplete and should not be treated as production merchant-ledger evidence.

## Structuring and threshold avoidance

Multiple transactions are kept just below a control threshold. Look for repeated amounts, short intervals, shared destinations, cumulative value, channel switching, and related accounts. A single below-threshold transaction is insufficient evidence.

## Authorized push-payment deception

The customer is manipulated into authorizing a transfer. Authentication may appear valid. Collect customer communications, beneficiary history, payment purpose, device continuity, warnings shown, confirmation behavior, and destination intelligence.

## Data limitation

PaySim is synthetic, fraud labels occur only for `TRANSFER` and `CASH_OUT`, and true device, identity, geography, channel, and timestamp evidence is absent. Typologies guide questions; they do not fill missing facts.
