# API documentation

Interactive OpenAPI docs: `http://localhost:8000/docs`. Base path `/api/v1`. JSON in/out; errors are `{"detail": "..."}` with 400/404/409/413/415/422. NaN/Inf are returned as `null`. Single local user, no authentication.

Long-running work returns `202 {job_id}`; poll `GET /jobs/{job_id}` → `{status: queued|running|succeeded|failed, progress, message, result}`.

| Area | Endpoint | Notes |
|---|---|---|
| Health | `GET /health`, `GET /config`, `GET /metadata` | metadata includes the sign convention and metric definitions |
| Datasets | `POST /datasets/generate` (job) | `{name, size|n_customers, scenario, seed, train}`; analysis seed = configured default |
| | `GET /datasets`, `GET /datasets/{id}`, `DELETE /datasets/{id}?confirm=true` | delete is permanent and needs `confirm` |
| | `POST /datasets/upload` (multipart `file`) → token + inspection + suggested mapping | `.csv` only, size limit (413), parse errors (422) |
| | `POST /datasets/upload/commit` `{upload_token, name, mapping, covariates?, control_label?}` | returns validation report with row exclusions and capability matrix |
| | `POST /datasets/{id}/validate`, `GET .../profile`, `GET .../quality-report[?download=true]` | |
| Portfolio | `GET /customers` (filters, `sort`, `page`, `page_size≤200`) | filters: loan_type, risk_band, customer_segment, response_segment, eligible_action, min_arr, uncertainty, region_group, min_delinq, split, cohort_from/to |
| | `GET /customers/{id}?dataset_id=`, `/repayment-history`, `/recommendations`, `/explanation` | explanation = structured decision object (predictive / causal / optimisation parts) |
| Models | `POST /models/train` (job), `GET /models`, `/models/{id}`, `/models/{id}/metrics`, `/models/overview/{dataset_id}` | predictive and uplift metrics are returned in separate families |
| Causal | `POST /causal/estimate`, `/causal/compare`, `/causal/diagnostics`, `/causal/sensitivity-analysis` | body: dataset_id, treatment, comparison, outcome, covariates?, cohorts?, estimator, bootstrap, trim_low?, ps_model, subgroup? |
| | `GET/POST /causal/experiments` | saved config + results + manifest hash |
| Uplift | `POST /uplift/train` (job; same pipeline), `GET /uplift/metrics`, `GET /uplift/segments` | thresholds high/moderate/z configurable |
| Optimisation | `POST /optimization/run` | constraints, objective, `z`, coverage, `compare`, `save`; infeasible runs return status + advice and are not saved |
| | `GET /optimization/runs`, `/runs/{id}`, `/runs/{id}/allocation` | |
| Scenarios | `GET /scenarios/catalogue`, `POST /scenarios/run`, `POST /scenarios/advanced`, `GET /scenarios`, `GET /scenarios/{id}` | advanced = frontiers, stress test, optimiser's curse, EVPI |
| Policies | `POST /policies`, `GET /policies`, `POST /policies/evaluate` | off-policy (DR) evaluation on held-out cohorts |
| Experiments | `POST /experiments/sample-size`, `POST /experiments/design` | |
| Governance | `GET /audit/decisions`, `POST /audit/decisions/{id}/review`, `GET /audit/log`, `GET /audit/decisions/export` (CSV) | overrides must be to an eligible action and need a reason |
| Monitoring | `GET /monitoring/summary`, `GET /monitoring/drift` | |
| Overview/Reports | `GET /overview`, `POST /reports/generate`, `GET /reports`, `GET /reports/{name}` | md/html/pdf |
| Assistant | `POST /assistant/parse` | keyword rules → *proposed* configuration only, `executed: false` |

## Example requests

```bash
curl -X POST localhost:8000/api/v1/datasets/generate -H 'content-type: application/json' \
  -d '{"name":"demo_obs","size":"medium","scenario":"observational","seed":42,"train":true}'
curl localhost:8000/api/v1/jobs/job-XXXXXXXXXX

curl -X POST localhost:8000/api/v1/causal/estimate -H 'content-type: application/json' \
  -d '{"dataset_id":1,"treatment":"standard_reminder","comparison":"none","outcome":"delinquency_30d","estimator":"aipw"}'

curl -X POST localhost:8000/api/v1/optimization/run -H 'content-type: application/json' \
  -d '{"dataset_id":1,"budget":120000,"agent_capacity":60,"costs":{"counselling":250},"z":0,"compare":true,"save":true}'

curl -X POST localhost:8000/api/v1/audit/decisions/run-abc-0001234/review -H 'content-type: application/json' \
  -d '{"reviewer":"j.analyst","outcome":"override","final_action":"none","reason":"customer on hardship arrangement"}'
```

Unavailable analyses return explicit statuses instead of values: `409` (models not trained), `422` (analysis unsupported for this dataset, with the requirement), `status: "insufficient_evidence"` (too few units).
