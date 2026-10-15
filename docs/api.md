# API

## Run locally

Train the models first, then start the API:

```bash
uvicorn api.main:app --reload
```

Interactive OpenAPI documentation is available at `/docs`.

## Score a transaction

`POST /api/v1/transactions/score`

Required fields are transaction type, amount, and origin/destination balances. Optional account-history fields accept online feature-store context. The response contains the unified risk score, fraud probability, operational decision, triggered rules, SHAP risk factors, evidence details, and component breakdown.

Invalid transaction types, negative amounts or balances, unknown fields, and identical origin/destination accounts return HTTP 422.
