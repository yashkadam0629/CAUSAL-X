# Model card

Generated from executed runs (`python scripts/run_demo_pipeline.py --n 20000`, seed 42, `artifacts/experiment_results/demo_pipeline.json`). Re-run to refresh; do not trust numbers copied from here for a different dataset.

## 1. Predictive risk model (Layer A)

* **Task**: probability that a customer has a 30-day delinquency (`delinquency_30d`) after the decision date, using pre-treatment features only (see data dictionary). The score describes risk under *historical practice*; it is not an intervention effect.
* **Candidates**: logistic regression, random forest, LightGBM gradient boosting; optional isotonic calibration (fit on cohort 8, judged on cohort 9, then re-fit on 8-9). Selection on **validation Brier score**, not accuracy. Threshold chosen to minimise expected cost on validation (FN cost 0.2×value, FP cost = reminder cost: analyst assumptions).
* **Splits**: chronological, train cohorts 0-7, validation 8-9, test 10-11; the preprocessing (median/mode imputation, scaling, one-hot) is fitted on training rows only.
* **Performance (held-out test cohorts, 20,000-customer demo runs)**: ROC-AUC ≈ 0.70-0.71, with the logistic model selected in all three scenarios; gradient boosting and random forest are within ≈ 0.01 AUC. The ceiling is set by the hidden latent traits (forgetfulness, income shock) that the features only proxy. Calibration and threshold metrics are on the Model Laboratory page.
* **Intended use**: ranking and baseline risk context in decision support. **Not** for credit approval, pricing or any use outside the synthetic setting.
* **Explainability**: SHAP values (log-odds) for the LightGBM model; they explain the *prediction*, not causal effects.
* **Known issues**: score drift is visible by design in cohorts 10-11 (economic stress rises from month 8); PSI of the score ≈ 0.6 in the demo data.

## 2. Treatment-effect (uplift) models (Layers B/C)

* **Method**: cross-fitted DR-learner per action vs "none", trained on cohorts 0-7 among customers eligible for the action; T-learner as a comparison ranker.
* **Evaluation**: IPW-weighted Qini/uplift curves on cohorts 10-11 with bootstrap CIs; quintile calibration of predicted vs IPW-realised benefit; for synthetic data only, correlation/RMSE against true individual effects (labelled benchmark).
* **Measured behaviour (synthetic, 40,000 customers, seed 42, observational scenario)**: correlation of DR-learner ARR with true ARR 0.34 (reminder), 0.29 (counselling), 0.67 (repayment plan); a baseline-risk ranking correlates 0.17 / 0.60 / 0.63 respectively. So uplift ranking adds value for reminders, is comparable to risk ranking for counselling, and the repayment-plan estimates are biased downward on average in the confounded scenario (mean predicted ARR ≈ 0 vs true 0.03).
* **Average effects (20,000 customers; truth → AIPW)**: randomised reminder −0.054 → −0.056; confounded reminder −0.054 → −0.051 (naive difference +0.041); hidden confounding reminder −0.054 → −0.016 (naive +0.061).
* **Limitations**: see `assumptions_and_limitations.md` (partial uncertainty, hidden confounding, ITT).

## 3. Governance

* Excluded from models and the optimiser: age band (audit-only), region group (coverage/audit only), outcome and post-treatment fields.
* Human review is flagged for low-confidence estimates (80 % interval includes zero), DR-vs-T-learner sign conflicts, potentially harmful estimates, missing income/credit history, top-1 % balances, and all agent-delivered actions.
* All recommendations store model, policy and data versions, the explanation, and any review (reviewer, outcome, reason, original and final action).
