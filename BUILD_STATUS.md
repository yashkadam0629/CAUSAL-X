# BUILD_STATUS

Last verified: 2026-10-26 on Windows 11, Python 3.12.10, Node 24.11, no Docker, no PostgreSQL.

## Verification results (actually executed)

| Command | Result |
|---|---|
| `cd backend && python -m pytest` | **61 passed**, exit code 0 (≈3.5 min). During the full run Python's faulthandler printed four `Windows fatal exception: access violation` lines in the middle of the run; the process continued and every test passed. They did not appear when test files were run individually, so they are most likely first-chance exceptions handled inside a native library (OpenMP/LightGBM or numba) and reported by faulthandler's vectored handler. Not investigated further. |
| `cd frontend && npm test` | 2 files, **5 passed** |
| `npx tsc --noEmit`, `vite build` | clean, build succeeds (single ~700 kB chunk; no code-splitting) |
| `python scripts/validate_dataset.py dev_obs` | quality score 100, no errors |
| `python scripts/run_demo_pipeline.py --n 20000` | all three scenarios ran end to end; output in `docs/model_card.md` and `artifacts/experiment_results/demo_pipeline.json` |
| `python scripts/benchmark.py --sizes 8000 40000` | table in `docs/testing_strategy.md` (200,000 preset not benchmarked) |
| Browser walk-through | Command Center, Causal Lab (full run), Uplift, Optimizer (full run + fairness), Policy Evaluation (full run) rendered with real data. Portfolio, Model Lab, Data, Governance, Experiment Designer, Reports, Scenarios loaded without errors; Scenario Studio *run buttons*, the Governance review dialog, CSV upload UI and report generation from the UI were **not clicked through** (their API endpoints are covered by `test_api.py`). |

## Completed (implemented, tested, wired to the UI)

* Synthetic generator: explicit DGP, 3 assignment scenarios, separate ground truth, reproducible fingerprints, MCAR missingness, economic indicators with drift, separate assignment/delivery/outcome.
* Validation: schema, referential integrity, temporal/outcome-window, duplicates, leakage warnings, capability detection, CSV upload with mapping, exclusions reported.
* Predictive layer: LR / RF / LightGBM, chronological split, validation-Brier selection, isotonic calibration choice, cost-aware threshold, SHAP.
* Causal layer: 4 estimators + diagnostics + sensitivity (covariate sets, PS models, trimming, placebo outcome, alt outcomes, E-value) + synthetic benchmark.
* Uplift: DR-learner CATE (cross-fitted, bootstrap), T-learner, Qini/AUUC with CIs, calibration, segments, risk-vs-benefit.
* Optimizer: sparse MILP with all listed constraint types, three objectives, conservative objective, 8 comparison strategies, shadow prices, exclusion reasons, infeasibility handling.
* Simulation: scenarios A-J, frontiers, Monte Carlo, stress tests, optimiser's-curse check, EVPI upper bound.
* Policy evaluation: doubly robust OPE on held-out cohorts with ground-truth comparison.
* Experiment designer: binary/continuous sample sizes with adjustments, protocol export.
* Governance: recommendation store with explanations and versions, review/override rules, audit log, CSV export, monitoring (PSI/KS, calibration by cohort, effect stability), fairness audit, reports (md/html/pdf).
* Frontend: 12 pages, light/dark theme, loading/empty/error states, provenance tags, modals with confirmation for destructive actions.

## Partial / known gaps

* **Uploaded datasets** support validation, profiling and causal estimation only (not predictive training, uplift, optimisation, policy evaluation, scenarios).
* **Causal methods not implemented**: causal forest, DML, X-/R-learner, multi-valued-treatment estimators, ATT for AIPW. CATE uncertainty excludes nuisance/confounding error.
* **Database**: schema created with `create_all`; **Alembic not wired**; PostgreSQL path untested; customers/loans/etc. live in Parquet, not DB tables (documented decision).
* **Docker / compose / nginx / CI workflow**: written, **never executed**.
* **Jobs** use an in-process thread pool; no cancellation.
* **Ideas not built**: causal-graph editor, dynamic policies, interference estimation, bandit simulation, DRO, EB shrinkage, EVSI (see `docs/architecture.md`).
* **Monitoring** snapshots are stored when the summary endpoint is called; no scheduler.
* **Frontend tests** cover the component kit only; page flows are untested automatically. No screenshots committed.
* **Scale**: 200,000-customer preset not exercised; the portfolio endpoint rebuilds/caches a full table in memory per model version.
* **Security**: single-user, no authentication, not reviewed.

## Notable findings from the demo runs (20,000 customers, seed 42)

* Randomised: all estimators recover the known effect (reminder: truth −0.054, AIPW −0.056).
* Confounded: naive difference +0.041 (wrong sign) vs adjusted −0.051.
* Hidden confounding: adjusted estimates are still wrong (−0.016; other arms flip sign); the placebo check fires; off-policy value of the optimised policy is wrong too (−9.7 vs truth +114.5 events prevented).
* Optimiser's curse is material: model-estimated net benefit of the optimised policy can be ~1.8× the synthetic-truth value, while still ranking first among compared strategies.
* A risk-only ranking is competitive with the DR-learner for counselling and repayment plans; uplift ranking helps most for reminders.

## Next tasks

1. Click-through browser tests (Playwright) for scenario run, review dialog, upload, report.
2. Empirical-Bayes shrinkage of CATEs and a DRO/aggregate-risk objective to shrink the optimism gap.
3. Wire Alembic; exercise PostgreSQL; run the Docker build.
4. Extend uploads to predictive modelling and effect estimation with user-mapped costs/eligibility.
5. Code-split the frontend bundle; add pagination server-side caching limits.
