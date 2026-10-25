# Causal methodology

## Question definition

Every analysis records: target population (customers assigned either arm *and* eligible for both), treatment, comparison, outcome, horizon
(30 days after the decision date, which is time zero), covariates (pre-treatment only), assignment mechanism (randomised only if logged
propensities exist; otherwise observational), estimand, assumptions, estimator, and diagnostics. The population funnel
(rows in two arms → eligible → analysed → trimmed → clipped) is returned with each result.

## Potential outcomes and sign convention

For customer *i* let `Y_i(a)` be the 30-day adverse outcome (1 = delinquent) under action *a*; `Y_i(none)` is the comparison.

```
tau          = E[ Y(treat) - Y(none) ]         negative = fewer adverse events
ARR (benefit) = -tau                            positive = beneficial
ATE   : average over the analysed population
ATT   : average over those assigned the treatment (IPW, regression adjustment)
CATE  : tau(x) = E[ Y(treat) - Y(none) | X = x ]
ITE   : Y_i(treat) - Y_i(none) is latent; only one potential outcome is ever observed per customer-decision
```

Only CATE-level model estimates are shown for individuals, always as estimates with an interval, never as facts about the person.
The estimand is an **intention-to-treat (ITT) effect of assignment**: assignment, delivery and receipt are distinct fields, and delivery is imperfect (reminder 92 %, counselling 70 %, plan review 85 % in the synthetic data).

## Identification assumptions (shown in the UI with every estimate)

1. **Consistency / SUTVA**: the observed outcome equals the potential outcome under the assigned action; no interference between customers.
2. **Conditional exchangeability** given the covariates (untestable). Guaranteed only by randomisation.
3. **Positivity**: each compared arm has non-zero probability for every covariate profile in the population (checked by overlap diagnostics; eligibility restricts the population to where it can hold).
4. **Time zero**: covariates measured at/before the decision date; the outcome window starts the day after (validated).
5. **Correct model** for the outcome *or* the propensity score (doubly robust estimators only need one).

## Estimators

| Estimator | Use | Notes |
|---|---|---|
| Difference in means | Randomised data, or as the naive baseline | Labelled *association only* unless randomised. Welch-type SE. |
| Regression adjustment | Baseline g-computation | Separate logistic outcome models per arm, standardised over the population; bootstrap SE (B configurable). Assumes the outcome models are right. |
| IPW (Hajek) | ATE and ATT | Propensity by logistic regression (or gradient boosting). Weight clipping is explicit (`clip_ps`, default 0.01) and the clipped count is reported; overlap trimming is an opt-in estimand change, never silent. Bootstrap refits the propensity model. |
| AIPW (cross-fitted) | Default | 5-fold cross-fitting, influence-function SE. Doubly robust: consistent if either nuisance model is right, but not robust to unmeasured confounding or to both models failing. ATT not reported. |
| DR-learner (CATE) | Heterogeneous effects | Cross-fitted nuisance → pseudo-outcome `phi = mu1 - mu0 + T(Y-mu1)/e - (1-T)(Y-mu0)/(1-e)`, regressed on X with shallow, regularised LightGBM. Uncertainty by bootstrapping the final stage only (excludes nuisance and confounding error; stated everywhere). |
| T-learner | Comparison | Separate outcome models per arm; no cross-fitting; used as an alternative ranker. |

Not implemented (and why): causal forests / DML / X- / R-learners (would add EconML; the DR-learner covers the same semi-parametric idea without the dependency, and more learners would add comparison noise, not validity), multi-valued-treatment estimators (each active action is compared with "none" as a separate binary contrast; actions are never compared with each other because that population and overlap differ).

## Diagnostics (per estimate)

Treatment/comparison sizes, standardised mean differences before and after weighting (Love plot; 0.10 is a convention), propensity histogram by arm, share outside [0.05, 0.95], effective sample size, max and p99 weights, missingness by arm, raw outcome means. Warnings: small arm (<200), poor overlap (>10 % outside), residual imbalance, ESS<50 % of nominal, extreme weights. Fewer than 30 units in an arm returns `insufficient_evidence` with no number.

## Sensitivity analysis (what each check does and does not show)

| Check | Applies to | Reading |
|---|---|---|
| Covariate specifications (drop each feature group, minimal set, unadjusted) | any adjusted estimator | Shows dependence on measured confounders. |
| Alternative propensity models (logistic vs boosting) | IPW / AIPW | Model-dependence of the weights. |
| Overlap restriction (e in [0.02/0.05/0.10]) | IPW / AIPW | Different (trimmed) population: not the same estimand. |
| **Placebo outcome**: pre-treatment `recent_late_flag` (kept out of the covariates) | observational contrasts | A non-zero adjusted "effect" on an outcome the treatment cannot change signals residual confounding. A zero result does not prove its absence. |
| Alternative outcomes (missed payment, 60 d, default) | all | Consistency across definitions (derived outcomes are nested in the primary one). |
| **E-value** (VanderWeele-Ding) for the risk ratio | binary outcome | Minimum confounder strength (RR scale, beyond measured covariates) needed to explain the estimate away. |
| Bootstrap uncertainty | RA, IPW, CATE | Sampling variability only. |

The "stability indicator" compares the spread of ATEs across specifications with the sampling CI width. It is a summary of robustness, not a validity test.

## Evidence classes (kept distinct in the UI and reports)

* **Randomised experimental evidence**: none in this repository (the randomised scenario is *synthetic*).
* **Observational estimate**: adjusted effects from logged data; conditional on assumptions.
* **Synthetic benchmark**: comparison with the generator's known effects. It verifies code and method behaviour under a stated DGP; it is not evidence about real customers.
