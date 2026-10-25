"""Experiment design: sample-size calculators and protocol generation (planning aid, never an approval)."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def n_two_proportions(p_control: float, mde_abs: float, alpha=0.05, power=0.8, ratio=1.0, two_sided=True) -> tuple[int, int]:
    """Unpooled normal-approximation sample size; ratio = n_treat / n_control. mde_abs is the absolute risk change (can be negative)."""
    p1 = p_control + mde_abs
    if not (0 < p_control < 1 and 0 < p1 < 1) or mde_abs == 0:
        raise ValueError("p_control and p_control+mde must lie in (0,1) and mde must be non-zero")
    za = stats.norm.ppf(1 - alpha / (2 if two_sided else 1)); zb = stats.norm.ppf(power)
    var = p_control * (1 - p_control) + p1 * (1 - p1) / ratio
    n_c = (za + zb) ** 2 * var / mde_abs ** 2
    return math.ceil(n_c), math.ceil(n_c * ratio)


def n_two_means(sd: float, mde_abs: float, alpha=0.05, power=0.8, ratio=1.0, two_sided=True) -> tuple[int, int]:
    za = stats.norm.ppf(1 - alpha / (2 if two_sided else 1)); zb = stats.norm.ppf(power)
    n_c = (za + zb) ** 2 * sd ** 2 * (1 + 1 / ratio) / mde_abs ** 2
    return math.ceil(n_c), math.ceil(n_c * ratio)


def design(outcome_type: str, baseline: float, mde_abs: float, alpha=0.05, power=0.8, ratio=1.0, n_arms_vs_control=1,
           attrition=0.0, compliance=1.0, icc=0.0, cluster_size=1, sd=None, two_sided=True) -> dict:
    """Adjustments (each reported separately):
    multiple comparisons -> Bonferroni alpha/k; noncompliance -> ITT effect diluted to compliance*mde (n scales 1/compliance^2);
    attrition -> inflate by 1/(1-attrition); clustering/interference -> design effect 1+(m-1)*icc."""
    k = max(1, n_arms_vs_control)
    a_adj = alpha / k
    eff = mde_abs * compliance
    if outcome_type == "binary":
        n_c, n_t = n_two_proportions(baseline, eff, a_adj, power, ratio, two_sided)
    elif outcome_type == "continuous":
        if not sd:
            raise ValueError("sd is required for a continuous outcome")
        n_c, n_t = n_two_means(sd, eff, a_adj, power, ratio, two_sided)
    else:
        raise ValueError("outcome_type must be 'binary' or 'continuous'")
    deff = 1 + (cluster_size - 1) * icc
    infl = deff / max(1e-9, 1 - attrition)
    n_c_f, n_t_f = math.ceil(n_c * infl), math.ceil(n_t * infl)
    return {
        "outcome_type": outcome_type, "formula": ("unpooled two-proportion z-test" if outcome_type == "binary" else "two-sample z-test (known sd)"),
        "inputs": {"baseline": baseline, "mde_absolute": mde_abs, "alpha": alpha, "power": power, "allocation_ratio_treat_to_control": ratio,
                   "comparisons_vs_control": k, "attrition": attrition, "compliance": compliance, "icc": icc, "cluster_size": cluster_size},
        "adjustments": {"alpha_after_bonferroni": a_adj, "effective_mde_after_noncompliance": eff, "design_effect": deff,
                        "inflation_for_attrition_and_clustering": infl},
        "n_control": n_c_f, "n_treatment_per_arm": n_t_f, "n_total": n_c_f + n_t_f * k,
        "n_before_adjustments": {"control": n_c, "treatment": n_t},
        "notes": ["Normal approximations; for very small or very large rates consider exact methods or simulation.",
                  "ITT analysis: the MDE should be the effect of *assignment*, which is diluted by non-delivery.",
                  "Bonferroni is conservative; a pre-registered hierarchy or Holm procedure can reduce required n.",
                  "Interference/spillover between customers (households, shared agents) violates SUTVA; consider cluster randomisation."],
    }


def protocol(cfg: dict, calc: dict) -> dict:
    """Structured protocol document. Status is always 'draft - requires institutional review'."""
    return {
        "status": "DRAFT - not approved. Requires institutional, compliance and ethical review before any customer is affected.",
        "objective": cfg.get("objective"), "target_population": cfg.get("target_population"), "eligibility": cfg.get("eligibility"),
        "treatment": cfg.get("treatment"), "control": cfg.get("control", "No additional intervention (business-as-usual)"),
        "randomization": {"unit": cfg.get("randomization_unit", "customer"), "stratification": cfg.get("stratification", []),
                          "allocation_ratio": cfg.get("allocation_ratio", 1.0), "logging": "Record assignment probability for every unit."},
        "primary_outcome": cfg.get("primary_outcome"), "outcome_horizon_days": cfg.get("horizon_days", 30),
        "secondary_outcomes": cfg.get("secondary_outcomes", []), "guardrails": cfg.get("guardrails", [
            "Complaint rate", "Opt-out rate", "Customer contact fatigue (contacts per customer)", "Serious delinquency (60d) not increasing"]),
        "sample_size": calc, "duration_assumption_days": cfg.get("duration_days"),
        "stopping_rules": cfg.get("stopping_rules", ["Pre-specified interim look(s) with alpha spending; stop for harm on guardrail breach"]),
        "risks": {"contamination": "Control customers receiving the treatment through other channels", "missing_outcomes": "Define handling of unobserved outcomes before launch",
                  "noncompliance": "Measure delivery/receipt separately from assignment", "interference": "Household or agent-level spillovers"},
        "ethics_checklist": ["Treatment withholding is acceptable (no known harm from control)", "Customer communications reviewed by compliance",
                             "Fairness/coverage review of eligible population", "Data protection review", "Stopping rule for harm agreed"],
    }
