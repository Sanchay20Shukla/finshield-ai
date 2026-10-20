# FinShield — Explainable Fraud Intelligence

## Full PaySim platform — connected frontend

Open **[the local platform](http://127.0.0.1:8000/platform/)** after starting it:

```powershell
cd finshield-ai
.\start.ps1
```

The new frontend extends the existing project with full-dataset analytics,
searchable transactions, live XGBoost + SHAP, prior account history, persistent
case reviews and audit events, evidence summaries, drift monitoring and exports.
The source contains **6,362,620 transactions and 8,213 labeled fraud events** over
743 simulated hours. The script builds a DuckDB analytics warehouse on first use.
Download PaySim into `data/raw/paysim.csv` if it is absent; the source is not tracked.

On another OS, install `requirements.txt`, run
`python -m src.data.warehouse --csv data/raw/paysim.csv`, then
`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`.

Portfolio risk bands and alerts use a documented rule baseline; individual
investigations execute the original trained XGBoost model with historical account
features from the full dataset. Analyst reviews persist locally in SQLite with
an event history. This workspace is designed for a single user on localhost.

Evidence summaries work without a key and are labeled **local, non-LLM** output.
Set `OPENAI_API_KEY` and `OPENAI_MODEL` in the process environment to enable the
existing optional LLM provider. Dataset fraud labels are excluded from its evidence.

### Reproduce the benchmark and reporting exports

```bash
python -m scripts.benchmark_platform
python -m scripts.export_reporting
```

The benchmark computes history over the entire dataset before deterministic 3%
negative sampling, retains all fraud, and splits at simulation hours 323/378.
Thresholds minimize validation financial cost. Holdout metrics use inverse
probability weights; population estimates are approximate. Models include Logistic
Regression, Random Forest, XGBoost, Isolation Forest and a rule baseline.
Fresh results are in `models/reproduced/benchmark.json` and Model intelligence.
The benchmark preserves the original serving model. The three supervised models
tied at zero estimated validation cost; the deterministic first-model tie winner
does not demonstrate better holdout performance. Isolation scores are anomaly
indices, not calibrated probabilities. Synthetic performance is not a production claim.

Power BI: [reporting guide](powerbi/README.md), [DAX measures](powerbi/Measures.dax),
[PostgreSQL star views](sql/star_schema.sql). Daily/channel CSVs use the full source.
A native `.pbix` is not included. Geography is UNKNOWN; dates are simulation-relative.

Stop the API before rebuilding after a CSV replacement. Run `start.ps1 -RebuildData`
to rebuild and restart, refreshing its cached analytics snapshot.

---

**[Open the live FinShield AI Streamlit app](https://sanchay-finshield-ai.streamlit.app/)** — executive charts, alert queue, transaction scoring, SHAP evidence and investigation workspace. No installation or sign-in required.

**[Working Model and reviewer guide](Working%20Model.md)**

A working fraud-detection portfolio project that connects **XGBoost inference, TreeSHAP explanations, policy rules, and an investigation UI**. Built for post-transaction analysis on synthetic PaySim data.

## Try it in 90 seconds

1. Open the **[live Streamlit app](https://sanchay-finshield-ai.streamlit.app/)**.
2. Review the **Executive dashboard** charts and sample-data caveat.
3. Choose **Score transaction**, adjust amount or balances, and submit to run the trained model.
4. Open **Alert queue** or **Transaction 360°** to inspect model drivers, policy evidence and account context.
5. Explore **Investigations** and **Model monitoring**. All workflow decisions are simulated.

The Streamlit demo needs **no login, API key, server startup, or dataset download**. It runs the included trained XGBoost model in Python with the curated synthetic PaySim sample. The separate `web/` application runs an exported 350-tree model in the browser.

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

The live Streamlit application runs Python scoring directly without a separate API service. It uses the included curated 168-row sample, which is enriched by label and type and is not representative population data. Its case notes and monitoring state are shared demonstration state, resettable when hosting restarts; use sample information only.

The separate `web/` output is suitable for GitHub Pages. It has no live Python API or PostgreSQL server. Its notes persist only in the visitor's browser. Its included cases have precomputed real TreeSHAP values; arbitrary-input SHAP runs in the Python platform.

## Run locally

Python 3.10 or 3.11 recommended. Trained models and a small sample are included.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m streamlit run dashboard/app.py
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
