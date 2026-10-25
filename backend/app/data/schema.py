"""Shared constants: interventions, feature groups, provenance and governance classification."""
from __future__ import annotations

NONE = "none"
TREATMENTS = ["standard_reminder", "counselling", "repayment_plan"]
ARMS = [NONE] + TREATMENTS

ARM_LABELS = {
    "none": "No additional intervention",
    "standard_reminder": "Standard reminder",
    "counselling": "Financial counselling",
    "repayment_plan": "Repayment-plan review",
}

# Base unit costs (currency units, synthetic / analyst-specified) and operational resource usage.
DEFAULT_COSTS = {"none": 0.0, "standard_reminder": 40.0, "counselling": 250.0, "repayment_plan": 500.0}
# Resource each action consumes: reminders use the automated channel, others use a trained human agent slot.
RESOURCE = {"standard_reminder": "channel", "counselling": "agent", "repayment_plan": "agent"}
AGENT_SLOTS = {"counselling": 1.0, "repayment_plan": 1.5}
DEFAULT_VALUE_PER_EVENT = 5000.0  # analyst assumption: loss avoided per delinquency event prevented

# Pre-treatment features available to risk / causal models, grouped for provenance.
FEATURE_GROUPS = {
    "financial_condition": [
        "log_monthly_income", "expense_ratio", "debt_to_income_ratio", "installment_to_income",
        "financial_stress_indicator",
    ],
    "repayment_history": [
        "historical_delinquency_count", "historical_on_time_payment_rate",
        "credit_history_length", "account_tenure_months", "prior_intervention_count",
    ],
    "loan_characteristics": [
        "principal_amount", "outstanding_balance", "interest_rate", "loan_term_months",
        "remaining_term_months", "loan_type", "employment_type",
    ],
    "macro_conditions": ["economic_stress_index"],
}
CATEGORICAL = ["loan_type", "employment_type"]
NUMERIC = [f for g in FEATURE_GROUPS.values() for f in g if f not in CATEGORICAL]
FEATURES = NUMERIC + CATEGORICAL

# Governance classification. Audit-only attributes are never model inputs or optimizer inputs.
AUDIT_ONLY = ["age_band"]
COVERAGE_GROUP = "region_group"  # operational coverage constraints (configurable), also fairness audit group
NEGATIVE_CONTROL_OUTCOME = "recent_late_flag"  # pre-treatment proxy kept out of covariates, used as placebo outcome

OUTCOMES = ["delinquency_30d", "missed_payment", "delinquency_60d", "default_event"]
PRIMARY_OUTCOME = "delinquency_30d"

# Cohort-based chronological roles (months 0..11)
SPLIT_TRAIN = range(0, 8)
SPLIT_VALID = range(8, 10)
SPLIT_TEST = range(10, 12)

PROVENANCE = {
    "measured": "Computed from observed data fields",
    "estimated": "Output of a fitted statistical model (carries uncertainty and assumptions)",
    "assumption": "Analyst-specified input, not measured from the data",
    "simulated": "Produced by simulation under stated assumptions",
    "synthetic_ground_truth": "Known only because data are synthetic; never a model input",
}
