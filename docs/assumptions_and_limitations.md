# Assumptions and limitations

## What CAUSAL-X is not

* Not a credit-decisioning, collections-compliance or suitability engine. Eligibility rules are simulated business rules. Nothing here establishes legal eligibility, financial suitability or regulatory compliance.
* Not evidence that any intervention works for real customers. All bundled data are synthetic and encode the generator's assumptions.
* Not a security-reviewed or production-ready system (single local user, no authentication, SQLite default).

## Statistical limits that matter

1. **Unmeasured confounding.** Adjusted estimates are valid only if the covariates capture what drove both assignment and outcome. The `observational_hidden` scenario shows an estimator failing silently (AIPW reminder effect −0.016 vs truth −0.054; counselling and repayment plan estimates flip sign) while the placebo check, E-value and unstable-spread indicator warn. In that scenario the *policy-evaluation* estimate of the optimised policy is wrong too (−9.7 vs +114.5 events prevented).
2. **Optimiser's curse.** Selecting the maximum of many noisy estimates overstates benefit. With 20,000 synthetic customers and randomised assignment the model-estimated net benefit of the optimised policy (≈ 960 k) is about 1.8× the synthetic-truth value (≈ 525 k), although the optimised policy is still the best of those compared. The conservative policy and larger samples narrow the gap; the headline numbers are *estimates*.
3. **CATE uncertainty is partial.** Individual intervals come from bootstrapping the final stage and exclude nuisance-model and confounding error. They should be read as lower bounds on the true uncertainty.
4. **Heterogeneity is hard to recover.** On held-out data the DR-learner's correlation with true individual ARR ranges from about 0.2 to 0.75 depending on arm and scenario, and for counselling and repayment plans a plain risk ranking is competitive (true ARR correlates ≈ 0.6 with baseline risk). The product reports this rather than assuming uplift modelling always wins; the clearest "risk ≠ benefit" case is the reminder.
5. **ITT estimand.** Effects are of assignment; delivery is imperfect and not separately modelled. No per-protocol or complier effects.
6. **Single period, one loan per customer.** No repeated-contact fatigue, dynamic eligibility, delayed outcomes beyond 30 days, or household/agent interference.
7. **Multi-arm.** Each action is compared only with "none"; actions are not compared with each other.
8. **Off-policy evaluation** uses estimated multinomial propensities in observational data (assumes exchangeability), clips weights at 0.02, and has high variance for policies that choose rarely-logged arms.
9. **Monitoring** is descriptive: PSI/KS thresholds are conventions; effect stability across three periods is indicative with wide intervals; concept drift is inferred from calibration/AUC by cohort only.
10. **Fairness audit** is descriptive (selection rates, calibration gaps, Wilson intervals, suppression below n=100). It cannot show absence of discrimination, and fairness criteria conflict.
11. **Financial values** (cost per action, value per avoided event, capacities) are analyst assumptions with default values; they are labelled "assumption" wherever they appear.

## Engineering limits

* Uploaded single-table CSVs support validation, profiling and causal estimation only. Predictive modelling, uplift, optimisation and policy evaluation require the standard multi-table schema (generated datasets).
* Alembic migrations are not wired; PostgreSQL support is configured but untested; Docker files are untested (Docker was unavailable).
* Background jobs use an in-process thread pool (state in the DB); there is no distributed queue.
* The portfolio table and default overview are cached in memory per model version.
* No authentication, rate limiting or per-user permissions.
