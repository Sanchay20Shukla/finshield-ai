# Deployment

## Public interviewer demo

The `web/` directory is self-contained static output, including real XGBoost trees and supplied-case TreeSHAP explanations. GitHub Actions publishes it through GitHub Pages on pushes to `main`. In repository Settings → Pages, source must be **GitHub Actions**. The public URL is recorded in the repository website field after successful deployment.

No backend services or secrets are needed. The demo runs model inference in the browser. Custom-input explanations from Python, shared cases, and PostgreSQL are not hosted by Pages. Reviews are browser-local.

## Python platform

Install `requirements.txt`, then run `python -m streamlit run dashboard/app.py` and `python -m uvicorn api.main:app --port 8000` in separate terminals. Included models and `data/sample/paysim_demo.csv` make the default clone runnable. If `data/raw/paysim.csv` exists it is used instead; set `PAYSIM_PATH` to override.

`docker compose up --build` adds PostgreSQL and mounts models and local state. An empty full-data mount is fine: the image includes the sample. Docker must be installed separately. The authoring machine did not have Docker, so container startup has not been verified locally; CI exercises PostgreSQL auditing.

`DATABASE_URL` is optional. When set, scoring writes an audit record to PostgreSQL. Failure to persist fails the request rather than silently dropping the audit. Streamlit cases and telemetry use SQLite under `data/processed` and are separate from PostgreSQL.

`/health` reports process liveness; `/ready` loads the serving model and explanation engine. No public authenticated API is configured by this project.

## Rebuilding browser assets

Run `python -m scripts.export_demo`, then `node --test tests/test_browser_parity.mjs`. Review provenance changes before committing. The script uses a full CSV when available, otherwise the supplied sample. Exported XGBoost split thresholds and inputs use float32 to match native prediction semantics.

## Security boundary

The public frontend has no credentials. Never commit `.env`, Streamlit secrets, personal data, raw production transactions, runtime SQLite stores, or MLflow logs. The Compose database credentials are local demonstration defaults. Production hosting requires authentication, authorization, TLS, network restrictions, secure secrets, resource limits, and a database migration strategy.
