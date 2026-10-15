# FinShield — Explainable Fraud Intelligence

A working fraud-detection portfolio project that connects **XGBoost inference, TreeSHAP explanations, policy rules, and an investigation UI**. Built for post-transaction analysis on synthetic PaySim data.

## Try it in 90 seconds

1. Open the **live demo** (repository website link).
2. Choose **Score a transaction → Load fraud example**. Inspect probability, policy score, rules, and SHAP contributions.
3. Load a legitimate example, then change its amount or balances and run the real model again.
4. Open **Transaction queue**, inspect a case, save a local analyst review, and download the evidence report.
5. Read **Model & methodology** for the assumptions and limitations.

The public demo needs **no login, API key, server startup, or dataset download**. It runs the actual exported 350-tree XGBoost model in the browser, not randomized scores or a rules-only substitute.

## What is implemented

| Component | Implementation |
|---|---|
| Public UI | Responsive browser dashboard, searchable/sortable queue, custom inference, sample SHAP explanations, local reviews, JSON exports |
| ML | Logistic Regression, Random Forest, XGBoost; chronological splitting; class imbalance handling; cost-based thresholds |
| Explainability | Native XGBoost TreeSHAP; numeric contributions and units in the Python API |
| API | FastAPI request validation, inference, model fingerprint, health/readiness, OpenAPI |
| Investigation | Streamlit workspace, policy evidence, case lifecycle, report generation, local grounded copilot |
| Persistence | PostgreSQL ingestion and optional inference audit; SQLite investigation/feedback/telemetry repositories |
| Experiment tracking | Optional MLflow parameters, metrics, and model artifact logging during training |
| Delivery | Docker Compose, Python tests, browser/Python parity tests, GitHub Actions CI and Pages deployment |

### Public demo versus full platform

GitHub Pages serves a static application. It has **no live Python API or PostgreSQL server**. Notes persist only in the visitor's browser. The included cases have precomputed real TreeSHAP values; arbitrary-input SHAP runs in the Python API. The browser UI labels these distinctions explicitly.

## Run locally

Python 3.10 or 3.11 recommended. Trained models and a small sample are included.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m streamlit rundashboard/app.py
```

Open http://localhost:8501. In another terminal with the environment activated:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000/docs for interactive API requests. `/health` is process liveness; `/ready` checks that the model and explanation engine load.

Example `POST /api/v1/transactions/score`:

```json
{
  "type": "TRANSFER",
  "amount": 250000,
  "oldbalanceOrg": 250000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 0,
  "newbalanceDest": 0
}
```

The response includes model probability, policy score/decision, rules, numeric SHAP factors, explanation units, threshold, and model fingerprint. The API defaults to cold-start history; callers may supply prior-count, prior-average, velocity, and destination-frequency fields.

To run the browser UI locally:

```bash
python -m http.server 8765 --directory web
# Open http://localhost:8765
node --test tests/test_browser_parity.mjs
```

## Docker and PostgreSQL

```bash
docker compose up --build
```

This starts PostgreSQL, FastAPI on port 8000, and Streamlit on port 8501. Models ship with the repo; missing full PaySim data falls back to the included demonstration sample. With `DATABASE_URL` configured, successful inference requests persist their input, response, timestamp, and model fingerprint in `inference_audit`. Core scoring works without PostgreSQL when this variable is unset.

The Compose password is a **local demo default**. Do not expose this stack publicly without replacing credentials and adding authentication and network controls. Docker was not available on the authoring machine; PostgreSQL round-trip verification is configured in CI.

## Training and MLflow

Download the full dataset from [PaySim on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it at `data/raw/paysim.csv`. The full dataset is intentionally excluded from Git.

```bash
python -m src.models.train --csv data/raw/paysim.csv
# Optional tracking client:
pip install -r requirements-tracking.txt
python -m src.models.train --csv data/raw/paysim.csv --mlflow-uri ./mlruns
# Use an MLflow server URI if you run a tracking server separately.
python -m scripts.export_demo
```

Thresholds and future model selection use validation expected loss. Evaluation restores sampled-negative population weights. Re-training is intentionally separate from deployment; do not treat the 168-row UI fixture as a benchmark.

**Artifact provenance:** the supplied model files and historical comparison metrics pre-existed this release. This release verifies loading, inference, SHAP additivity, and browser parity; it does not claim a freshly reproduced full-dataset benchmark. The historical comparison ranked models on test results, so those results should not be presented as an untouched final model-selection evaluation. The training code now ranks by validation loss. XGBoost is the serving default for this project, not a claim that it beat the historical Random Forest.

## Engineering decisions and limitations

- **Timing:** new balances are post-event features. This is retrospective detection, not a validated pre-authorization payment blocker.
- **Probability:** class weighting and negative sampling affect calibration. Probability output is explicitly uncalibrated.
- **Policy:** `65% model + 20% rules + 15% behavior` and bands 30/60/80 are illustrative; the classifier threshold is separate.
- **Sample:** up to 24 records per type/label stratum. The displayed fraud/alert rate is not population prevalence.
- **Explanations:** XGBoost SHAP contributions are log-odds, not percentage points or causal evidence.
- **History:** production needs a chronological feature store. The browser uses cold-start history; the Streamlit sample history is bounded to the sample.
- **Persistence:** browser reviews are local; Python case/feedback state uses SQLite; configured API audit and ingestion use PostgreSQL. These are not a single production system of record.
- **Production gaps:** authentication, authorization, rate limits, secure secrets, calibration, representative validation, load testing, migrations, and operational monitoring remain deployment work.

## Repository map

```text
web/                 Public interviewer demo and portable tree inference
api/                 FastAPI schemas, scoring service, PostgreSQL audit
src/features/        Shared chronological feature engineering
src/models/          Training, metrics, inference, MLflow, feedback
src/rules/           Deterministic and hybrid policy scoring
src/investigation/   SHAP, cases, reports, evidence
src/monitoring/      Local prediction and API telemetry
src/rag/             Grounded local / optional API copilot
dashboard/          Streamlit investigation UI
models/              Included trained artifacts and historical metrics
data/sample/         Attributed synthetic UI fixture
scripts/             Reproducible browser model/data export
tests/               Unit, API, persistence, artifact and parity checks
```

See [architecture](docs/architecture.md), [deployment](docs/deployment.md), [model card](docs/model_card.md), and [interview walkthrough](docs/interview.md).

## Data attribution

PaySim dataset by Edgar Lopez-Rojas and collaborators: [source](https://www.kaggle.com/datasets/ealaxi/paysim1). The included sample retains synthetic account identifiers and is for demonstration. See [data notes](data/sample/README.md). Dependency and dataset licenses remain with their respective owners.
