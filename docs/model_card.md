# Model card

## Intended use

FinShield prioritizes transactions for review using PaySim-trained baseline models combined with deterministic fraud rules. It is a portfolio demonstration and must not be deployed for autonomous financial decisions without production data validation, calibration, fairness review, and monitoring.

## Evaluation design

- Chronological 70%/15%/15% train, validation, and test split by PaySim step.
- All fraud records retained. Legitimate records are sampled only after chronological feature computation; evaluation weights restore population counts.
- Threshold selected on validation expected loss, never on the test set.
- Metrics: precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, false-positive rate, false-negative rate, and expected loss.
- Default costs: INR 10,000 per missed fraud and INR 50 per false-positive review, excluding unquantified customer friction.

## Limitations

PaySim is synthetic, has fraud labels only for `TRANSFER` and `CASH_OUT`, lacks true timestamps and customer attributes, and does not represent current production behavior. Model probabilities require calibration on representative production data. Business costs are configurable assumptions and should be replaced by measured loss, review, churn, and friction costs.

## Explainability

Transaction-level explanations use SHAP values from the persisted model and its fitted preprocessing pipeline. Positive contributions are ranked as primary risk factors; negative contributions are retained as protective context. Technical feature names are translated into investigator language while the original feature value and numeric contribution remain available for audit.

SHAP describes the model's behavior, not the true cause of fraud. Correlated balance features may divide or redistribute attribution, so investigators should treat explanations as decision support rather than causal evidence.

## Serving and artifact provenance

XGBoost is the default serving artifact. Existing model artifacts and historical metrics predate this release and were not retrained on the full dataset during release verification. Historical model comparison ranked on test loss; new training selects on validation loss. The sample training run is a pipeline smoke test, not a performance benchmark.

The public demo is static browser inference. Its 168 stratified examples include precomputed native TreeSHAP (log-odds) explanations. Arbitrary-input TreeSHAP and optional PostgreSQL auditing are available through the Python service. Model probability is uncalibrated; the composite policy score is not a probability.
