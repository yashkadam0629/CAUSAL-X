# Testing strategy

```bash
cd backend && python -m pytest          # unit + integration + end-to-end acceptance (~3-4 min)
cd frontend && npm test && npm run build
```

| Suite | File | What it proves |
|---|---|---|
| Data | `test_data.py` | seeded reproducibility (fingerprints), schema/referential integrity, outcome windows after time zero, repayment history before time zero, assignment respects eligibility, logged propensities only in the randomised scenario, naive comparison has the wrong sign under confounding, ground truth never reaches features, upload validation (duplicates, bad dates, leakage, unsupported analyses) |
| Predictive | `test_predictive.py` | chronological split order, no outcome/treatment/audit features, preprocessing fitted on training only, metric + calibration output, threshold search, model serialisation round-trip |
| Causal | `test_causal.py` | all four estimators recover the known ITT effect under randomisation (tolerance 3.5 SE or 0.02), adjustment fixes confounding where difference-in-means has the wrong sign, hidden confounding flagged by the placebo outcome (and not flagged when absent), sensitivity structure, SMD/ESS/weight diagnostics, poor-overlap warning, `insufficient_evidence`, E-value known values |
| Uplift & experiment | `test_uplift_experiment.py` | segment definitions and sign convention, Qini of a perfect vs random ranker, unbiasedness of the IPW benefit signal, sample-size formulas vs hand calculation and the documented adjustments, invalid designs, PSI, fairness suppression/intervals |
| Optimisation | `test_optimization.py` | budget/agent/channel limits, eligibility, one action per customer, cost counted once, optimiser ≥ every heuristic, heuristics feasible, unavailable actions, infeasible coverage reported as infeasible, coverage met, SE cap and minimum benefit, conservative objective, min-cost objective, reproducibility, scenario directions, Monte Carlo seeding, frontier monotonicity, EVPI ≥ 0, optimiser's-curse gap smaller for the conservative policy |
| API / acceptance | `test_api.py` | validation errors, 404/409/422/413/415 paths, malformed uploads, destructive-action confirmation, pagination/filters, and the **15-step acceptance run**: generate → load → validate → train → estimate → diagnostics → sensitivity → uplift → optimise → compare policies → scenarios → save → audit/review (override rules, original recommendation preserved) → report (md/html/PDF) → retrieve |
| Frontend | `format.test.ts` | formatting helpers (missing values rendered `n/a`). The rest of the UI was exercised manually in a browser (see BUILD_STATUS); component and flow tests are a known gap. |

Tests use an isolated temporary data directory and SQLite file (`conftest.py`). Statistical tests use fixed seeds and tolerances justified by the sampling error; none requires exact recovery.

## Benchmarks (this machine: AMD Ryzen-class 8 threads, Python 3.12, Windows; two jobs were running concurrently so times are conservative)

| customers | decision cust. | MILP vars | generate (s) | write parquet (s) | train all models (s) | AIPW estimate (s) | MILP solve (s) |
|---|---|---|---|---|---|---|---|
| 8,000 | 1,374 | 1,546 | 0.6 | 1.2 | 23.0 | 0.9 | 0.0 |
| 40,000 | 6,521 | 7,789 | 3.0 | 5.2 | 63.8 | 3.1 | 1.2 |

The 200,000-customer preset was not benchmarked. Bottlenecks: cross-fitted nuisance models and the bootstrap in CATE training (≈ 85 % of training time), and the Python loop in the random/greedy baselines (≈ 5 s for 50 random draws on 6.5 k customers). A full optimisation + 9-strategy comparison + persistence of ≈ 3,000 recommendations takes ≈ 15-20 s at 40 k customers.
