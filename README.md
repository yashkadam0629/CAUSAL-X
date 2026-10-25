# CAUSAL-X — Financial Decision Intelligence & Causal Impact Optimization

> **Predict outcomes. Estimate impact. Optimize decisions.**

CAUSAL-X is a local, end-to-end decision-support platform for a lender that has to decide *which customer should receive which repayment intervention* under a limited budget and limited agent capacity. It deliberately separates what a conventional risk dashboard blurs together:

| Layer | Question | Where |
|---|---|---|
| A. Predictive risk | Who is likely to miss a payment? | `predictive/`, Model Laboratory |
| B. Causal effects | What would an intervention change on average? (ATE/ATT, diagnostics, sensitivity) | `causal/`, Causal Lab |
| C. Heterogeneous effects | For whom does it change the outcome? (CATE, uplift segments, Qini) | `uplift/`, Uplift Intelligence |
| D/E. Decision & optimisation | Which action for whom under budget, capacity, eligibility, uncertainty? | `optimization/`, Optimizer |
| F. Financial impact | What is it worth, how does it respond to assumptions, how reliable is the estimate? | `simulation/`, `policy_evaluation/`, Scenario Studio, Policy Evaluation |
| G. Governance | Can a human review, override, audit and monitor it? | `governance/`, `monitoring/`, Audit & Governance |

> **Status: a working, tested vertical system for synthetic data — not production software.** All bundled data are synthetic; they validate the methodology, not real-world effectiveness. See [Known limitations](#known-limitations) and `BUILD_STATUS.md`.

## Why prediction alone is not enough

The highest-risk customer is not necessarily the customer who benefits most. Some will repay regardless of contact, some will default despite it, some respond, and some are *burdened* by an inappropriate action. In the demo data (40,000 customers, observational assignment), baseline risk correlates only 0.57 with the estimated best-action benefit, and risk is nearly uncorrelated (0.17) with the true benefit of a reminder. A confounded comparison of treated vs untreated customers even has the *wrong sign*: lenders contact riskier customers, so reminders look harmful (+4.1 pp) when their true effect is −5.4 pp. CAUSAL-X shows this, shows how adjustment recovers the effect when confounding is measured, and shows how an unmeasured confounder defeats it — and how to detect that.

## Key features

* **Synthetic portfolio generator** with an explicit causal data-generating process, three assignment regimes (randomised, confounded-observable, confounded-by-hidden-variable) and a *separate* ground-truth table used only for benchmarks.
* **Causal engine**: difference in means, regression adjustment, IPW, cross-fitted AIPW; balance (SMD/Love), overlap, ESS and weight diagnostics; sensitivity (covariate sets, propensity models, trimming, placebo outcome, alternative outcomes, E-value).
* **Uplift**: cross-fitted DR-learner CATE with bootstrap uncertainty, T-learner comparison, Qini/AUUC with bootstrap CIs, quintile calibration, configurable response segments, risk-vs-benefit view.
* **Optimizer**: sparse MILP (HiGHS) with budget, agent/channel capacity, per-action caps, eligibility, exclusivity, uncertainty cap, minimum benefit, coverage minima, shadow prices; eight comparison strategies (including highest-risk-first and an optimiser fed naive effects).
* **Reliability tooling**: conservative (lower-bound) objective, optimiser's-curse simulation, budget/capacity frontiers, stress tests, Monte Carlo propagation, value-of-information upper bound, off-policy (doubly robust) policy evaluation vs ground truth.
* **Governance**: eligibility rules recorded, explanation stored with every recommendation, human review with reasons, original-vs-final decision, audit log, versions (model/policy/data), PSI/KS drift, fairness audit with minimum-n suppression.
* **Data management**: CSV upload with column mapping, validation, leakage warnings, explicit row-exclusion counts, capability matrix (it never invents treatments or outcomes).
* **Experiment designer** with sample sizes for binary/continuous outcomes, attrition, non-compliance, clustering and multiple comparisons; protocol export (always "draft — not approved").
* **12-page React dashboard**, Markdown/HTML/PDF executive report.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) (diagram, decisions, scored brainstorm of 20 additions).

```
frontend (React/TS/Vite/Tailwind/Recharts) ──/api/v1──> FastAPI ──> analytic modules ──> Parquet (data) + SQLite/PostgreSQL (metadata, results, audit)
```

## Technology stack

Python 3.12 · FastAPI · Pydantic 2 · SQLAlchemy 2 · pandas / NumPy / SciPy (HiGHS MILP) · scikit-learn · LightGBM · SHAP · ReportLab · React 19 · TypeScript · Vite · Tailwind 4 · Recharts · lucide-react · pytest · vitest. Deviations from the original brief (and why): no EconML/DoWhy/CVXPY/MLflow/Alembic/shadcn — see `docs/architecture.md`.

## Install and run

Requirements: Python ≥ 3.11 (tested 3.12), Node ≥ 20 (tested 24). No Docker, PostgreSQL or paid service is needed.

```bash
# 1. backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cp .env.example .env                                    # optional; defaults work

# 2. demo data + models (deterministic, seed 42, ~1.5 min for 40,000 customers)
python scripts/generate_demo_data.py --name demo_obs --size medium --scenario observational --seed 42 --train

# 3. API (http://localhost:8000/docs)
cd backend && python -m uvicorn app.main:app --port 8000

# 4. dashboard (http://localhost:5173) — in a second terminal
cd frontend && npm install && npm run dev
```

If port 8000 or 5173 is taken, run the API elsewhere and tell the dev server: `VITE_API_TARGET=http://127.0.0.1:8010 npx vite --port 5180`.
`make setup | data | api | web | test | demo | bench` wrap the same commands. Docker files exist (`docker-compose.yml`) but were **not run** (Docker was unavailable when building).

Data are stored in `data/` (Parquet per dataset, SQLite `data/causalx.db`); models and reports in `artifacts/`. All are git-ignored and regenerated by the commands above. To use PostgreSQL set `CAUSALX_DATABASE_URL` (untested).

You can also generate datasets and train from the UI (**Data & Quality → Generate & train**).

## Example workflow

1. *Data & Quality*: generate `demo_obs` (observational) and `demo_rct` (randomised); validate (score 100).
2. *Causal Lab*: reminder vs none. Compare raw difference (+0.041, wrong sign) with AIPW (−0.051, truth −0.054). Run sensitivity; open the Love plot and overlap chart.
3. *Uplift Intelligence*: scatter of risk vs benefit, segments (click a bar → Portfolio), Qini curves for DR-learner vs risk-only ranking.
4. *Intervention Optimizer*: set budget/agents, run. Compare with highest-risk-first; read binding constraints and shadow prices; open fairness table.
5. *Scenario Studio*: run scenarios A-J; advanced analyses (frontier, stress test, optimiser's curse, EVPI).
6. *Policy Evaluation*: doubly robust held-out evaluation vs ground truth.
7. *Audit & Governance*: review a recommendation, override with a reason, export CSV; *Monitoring*: drift.
8. *Executive Reports*: generate the report.

Same flow without the UI: `python scripts/run_demo_pipeline.py --n 20000`.

## Methodology (summary)

Potential outcomes `Y(a)`; adverse-outcome effect `τ = E[Y(treat) − Y(none)]` (negative is good) and benefit `ARR = −τ` everywhere. Estimand: ITT effect of assignment. AIPW:
`ψ_i = μ̂₁(x_i) − μ̂₀(x_i) + T_i(Y_i − μ̂₁)/ê(x_i) − (1−T_i)(Y_i − μ̂₀)/(1−ê(x_i))`, `τ̂ = mean ψ`, SE from the influence function (5-fold cross-fitting). DR-learner regresses `ψ_i` on `x_i` for CATE. Optimisation: `max Σ x_{ia}(V·(ARR_{ia} − z·SE_{ia}) − c_a)` s.t. one action per customer, budget, capacity, coverage. Details: [`causal_methodology`](docs/causal_methodology.md), [`optimization_methodology`](docs/optimization_methodology.md), [`experiment_design`](docs/experiment_design.md), [`data_dictionary`](docs/data_dictionary.md), [`model_card`](docs/model_card.md), [`api_documentation`](docs/api_documentation.md), [`testing_strategy`](docs/testing_strategy.md), [`assumptions_and_limitations`](docs/assumptions_and_limitations.md).

### Evidence classes

| Class | Meaning here |
|---|---|
| Synthetic benchmark | Known generator effects; verifies code and method behaviour. Labelled in the UI. |
| Observational estimate | Adjusted from logged assignment; conditional on exchangeability. |
| Randomised experimental evidence | **None in this repository.** The experiment designer plans how to get it. |

## Tests

```bash
cd backend && python -m pytest        # 61 tests incl. the 15-step end-to-end acceptance run
cd frontend && npm test && npm run build
```

Latest recorded results are in `BUILD_STATUS.md`.

## Screenshots checklist

Command Center · Portfolio + decision panel · Causal Lab (estimators, Love plot, overlap, sensitivity) · Uplift scatter and Qini · Optimizer comparison · Scenario frontier · Policy Evaluation · Audit review · Data quality report. (Not committed; run the app and capture.)

## Known limitations

See `docs/assumptions_and_limitations.md`. Highlights: hidden confounding defeats adjustment (detectable, not fixable); the optimiser's curse overstates benefit (≈1.8× in one demo run); CATE intervals are partial; one period/one loan per customer; uploads support causal estimation but not uplift/optimisation; PostgreSQL, Docker and Alembic untested/unwired; no authentication.

## Ethical and governance considerations

Decision support only; consequential actions are routed to human review; audit-only attributes never enter models or the optimiser; fairness tables are descriptive and suppress small groups; every recommendation is reproducible from stored versions. No claim of legal or regulatory compliance is made. Use synthetic data for demonstrations; do not upload real customer data to a deployment you have not secured.

## Future improvements

Empirical-Bayes shrinkage of CATEs, distributionally robust optimisation, DAG editor with adjustment-set checks, dynamic policies with contact fatigue, household-level interference, simulated bandit allocation, EVSI for specific experiments, Alembic migrations, authentication, component/E2E browser tests.

## Reproducibility

`GenConfig(seed)` + `CAUSALX_SEED` fix data and models; datasets carry SHA-256 fingerprints; model version = hash(dataset, fingerprint, seed, features); every run stores config, policy hash, model and data versions. Experiments store a manifest hash.

## Project structure

```
backend/app/{core,api,data,predictive,causal,uplift,optimization,policy_evaluation,experiment_design,simulation,explainability,monitoring,governance,reporting,services,database}
backend/tests   frontend/src/{app,components,pages,hooks,services,utils,styles}   scripts/   docs/   data/   artifacts/
```

## License

MIT (see `LICENSE`). Synthetic data and all indicators are fabricated for demonstration; they are not official statistics.
