# Interview walkthrough

## 60–90 second introduction

“FinShield is a post-transaction fraud detection and investigation project built around the synthetic PaySim dataset. I combine engineered transaction and balance features with XGBoost, then expose the prediction and SHAP evidence through a Python scoring service and investigation interface.

The project goes beyond a classifier: it separates model output from a configurable policy score, shows triggered rules, supports analyst notes and downloadable evidence, and includes API validation and reproducible deployment files. The GitHub demo runs the actual exported model in the browser, so reviewers can try it immediately. The full Python version provides arbitrary-input TreeSHAP, optional PostgreSQL auditing, and MLflow tracking during training.

A key limitation is that the model uses post-event balances, so I describe it as retrospective detection. I also explicitly distinguish synthetic-data results from production performance and uncalibrated probabilities from real-world fraud likelihood.”

## Demonstrate, do not just describe

1. Load a fraud example. Explain probability versus policy score.
2. Inspect positive and protective SHAP contributions. Explain log-odds and non-causality.
3. Change inputs. Explain why custom browser results have policy evidence but no new browser SHAP calculation.
4. Investigate a case, save a local review, download its JSON report.
5. Open methodology. Explain the difference between Pages and the Python backend.
6. Show parity tests and real model API tests in the repository.

## Likely follow-up questions

**Why not accuracy?** Fraud is rare. Discuss precision, recall, PR-AUC, false positives, and expected cost. The demo fixture's prevalence is intentionally enriched.

**Why XGBoost?** It handles nonlinear interactions in tabular data. The project also includes Logistic Regression and Random Forest; do not claim XGBoost won the historical comparison.

**How do you avoid leakage?** Chronological splitting and history excluding the current record. Post-event features are deliberately allowed only because the task is retrospective. Future-account aggregates must not enter a prediction.

**Why not always 0.5?** Threshold selection uses validation false-negative and review costs. Costs are assumptions, not measured production economics.

**Does 0.9 mean 90% fraud?** Not necessarily. Weighted and sampled training needs calibration on representative held-out data.

**Is this production-ready?** It is a working portfolio implementation. Auth, representative validation, calibration, load testing, monitoring, migrations, and operational controls are still required.

**Did you train these models in this release?** Existing local model artifacts were retained. This release tests inference and explanations, adds deployment and parity checks, and corrects validation-based selection for future training. Historical metrics are labelled accordingly.
