"""Decision explanations. Three distinct explanation types are kept separate:
  1. predictive (SHAP on the risk model; NOT causal),
  2. causal (estimator, interval, uncertainty, assumptions),
  3. optimisation (cost, marginal value, constraints, alternatives)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import ARM_LABELS, TREATMENTS
from ..optimization.engine import OptConfig
from ..predictive.risk import local_explanation

ELIGIBILITY_RULES = {
    "standard_reminder": "Customer is contactable (consent / valid channel on file).",
    "counselling": "Contactable AND (>=1 prior delinquency OR stress indicator > 0.5 OR debt-to-income > 0.40).",
    "repayment_plan": "Not a revolving credit line AND >=6 months remaining AND (DTI >= 0.35 OR >=2 prior delinquencies).",
}
Z_UNC = 1.28


def uncertainty_status(arr: float, se: float) -> str:
    if se <= 0:
        return "unavailable"
    lcb, ucb = arr - Z_UNC * se, arr + Z_UNC * se
    if lcb > 0:
        return "low" if se < 0.02 else "moderate"
    if ucb < 0:
        return "possibly_counterproductive"
    return "high_interval_spans_zero"


def review_flags(row: pd.Series, arr: float, se: float, arr_t: float, action: str, balance_q99: float) -> list[str]:
    flags = []
    if action != "none":
        if arr - Z_UNC * se <= 0:
            flags.append("Low confidence: the 80% interval for the estimated benefit includes zero.")
        if np.sign(arr) != np.sign(arr_t) and abs(arr) > 0.01:
            flags.append("Conflicting model outputs: DR-learner and T-learner disagree on the sign of the effect.")
        if arr + Z_UNC * se < 0:
            flags.append("Potentially harmful: the estimate suggests the action may increase the risk of an adverse outcome.")
        if action in ("counselling", "repayment_plan"):
            flags.append("Consequential action: involves a trained human agent and the customer's repayment terms; requires qualified decision-maker sign-off.")
    if pd.isna(row.get("monthly_income")) or pd.isna(row.get("credit_history_length")):
        flags.append("Incomplete data: income or credit history length is missing (imputed by the model).")
    if row.get("outstanding_balance", 0) > balance_q99:
        flags.append("High-impact case: outstanding balance in the top 1% of the portfolio.")
    return flags


def explain_customer(a, customer_id: str, cfg: OptConfig, selected_action: str | None = None, alloc_info: dict | None = None, with_shap: bool = True) -> dict:
    from ..services.pipeline import customer_segments
    fr = a.frame
    memo = a.__dict__.setdefault("_memo", {})
    if "cid" not in memo:
        memo["cid"] = {c: k for k, c in zip(fr.index, fr.customer_id)}
        memo["q99"] = float(fr["outstanding_balance"].quantile(0.99))
    if customer_id not in memo["cid"]:
        raise KeyError(customer_id)
    i = memo["cid"][customer_id]
    row = fr.loc[i]
    sc = a.scores.loc[i]
    risk = float(a.risk[i])
    cands = []
    for arm in TREATMENTS:
        elig = bool(row[f"elig_{arm}"])
        arr, se, arr_t = float(sc[f"arr_{arm}"]), float(sc[f"se_{arm}"]), float(sc[f"arr_tlearner_{arm}"])
        cost = float(cfg.costs[arm])
        gross = arr * cfg.value_per_event
        net = gross - cost
        cands.append({
            "action": arm, "label": ARM_LABELS[arm], "eligible": elig, "eligibility_rule": ELIGIBILITY_RULES[arm],
            "estimated_absolute_risk_reduction": {"value": arr, "interval_80": [arr - Z_UNC * se, arr + Z_UNC * se], "se": se,
                                                  "unit": "probability (positive = fewer adverse outcomes)", "t_learner_value": arr_t,
                                                  "provenance": "estimated"},
            "intervention_cost": {"value": cost, "provenance": "assumption"},
            "gross_benefit": {"value": gross, "formula": "ARR x value_per_event", "value_per_event": cfg.value_per_event, "provenance": "estimated x assumption"},
            "estimated_net_benefit": {"value": net, "formula": "gross - cost (cost subtracted once)"},
            "break_even_arr": cost / cfg.value_per_event, "uncertainty_status": uncertainty_status(arr, se),
            "available": arm in cfg.available,
        })
    el = [c for c in cands if c["eligible"] and c["available"]]
    best = max(el, key=lambda c: c["estimated_net_benefit"]["value"]) if el else None
    action = selected_action or (best["action"] if best and best["estimated_net_benefit"]["value"] > 0 else "none")
    chosen = next((c for c in cands if c["action"] == action), None)
    reasons = []
    if action == "none":
        if not el:
            reasons.append("No intervention is both eligible and available for this customer.")
        elif best["estimated_net_benefit"]["value"] <= 0:
            reasons.append(f"No eligible action has positive estimated net benefit; best is {best['label']} at {best['estimated_net_benefit']['value']:.0f}.")
        else:
            reasons.append("An action with positive net benefit exists but was not selected under the active budget/capacity/uncertainty constraints.")
    else:
        reasons.append(f"{chosen['label']} has the highest estimated net benefit among eligible actions ({chosen['estimated_net_benefit']['value']:.0f}) "
                       "and fits within the remaining budget and capacity." if (best and best["action"] == action)
                       else f"{chosen['label']} was selected by the optimiser to satisfy the joint budget/capacity constraints, although {best['label']} has a higher stand-alone net benefit.")
    alternatives = []
    for c in cands:
        if c["action"] == action:
            continue
        why = []
        if not c["eligible"]:
            why.append("not eligible: " + c["eligibility_rule"])
        elif not c["available"]:
            why.append("unavailable in this scenario")
        else:
            if c["estimated_net_benefit"]["value"] <= 0:
                why.append(f"estimated ARR {c['estimated_absolute_risk_reduction']['value']:.3f} is below break-even {c['break_even_arr']:.3f}")
            elif chosen and c["estimated_net_benefit"]["value"] < chosen["estimated_net_benefit"]["value"]:
                why.append("lower estimated net benefit than the selected action")
            else:
                why.append("higher stand-alone net benefit but displaced by a binding budget/capacity/uncertainty constraint")
        alternatives.append({"action": c["action"], "reason_not_selected": "; ".join(why)})
    flags = review_flags(row, chosen["estimated_absolute_risk_reduction"]["value"] if chosen else 0.0, chosen["estimated_absolute_risk_reduction"]["se"] if chosen else 0.0,
                         chosen["estimated_absolute_risk_reduction"]["t_learner_value"] if chosen else 0.0, action, memo["q99"])
    # what would need to change
    change = []
    if chosen is None or action == "none":
        if best:
            change.append(f"{best['label']} would be selected if its ARR exceeded {best['break_even_arr']:.3f} (currently {best['estimated_absolute_risk_reduction']['value']:.3f}) "
                          "or if its unit cost fell below the gross benefit.")
    else:
        change.append(f"{chosen['label']} stops being worthwhile if ARR falls below {chosen['break_even_arr']:.3f} or cost exceeds {chosen['gross_benefit']['value']:.0f}.")
    seg = customer_segments(a, a.frame.loc[[i]]).iloc[0]
    return {
        "customer_id": customer_id, "synthetic_dataset": a.dataset, "decision_cohort": int(row["cohort_month"]), "split": row["split"],
        "baseline_risk": {"value": risk, "target": a.risk_bundle.target, "model_version": a.model_version, "model": a.risk_bundle.model_name,
                          "meaning": "Predicted probability of the adverse outcome under historical practice (prediction, not causal effect)."},
        "recommended_action": action, "candidate_interventions": cands,
        "selection_reason": " ".join(reasons), "alternatives_not_selected": alternatives,
        "constraints": alloc_info or {"note": "No allocation run attached; showing stand-alone economics."},
        "what_would_change_the_decision": change, "response_segment": seg,
        "human_review_required": bool(flags), "human_review_reasons": flags,
        "explanations": {
            "predictive": {"type": "SHAP (log-odds) from gradient-boosting risk model - describes the prediction, NOT a causal effect",
                           "top_drivers": local_explanation(a.risk_bundle, a.frame.loc[[i]]) if with_shap else "computed on demand"},
            "causal": {"type": "DR-learner CATE (cross-fitted nuisance, bootstrap final stage) vs no additional intervention",
                       "assumptions": ["conditional exchangeability given pre-treatment covariates", "positivity within eligible population", "no interference"],
                       "limits": "Estimated conditional average effect for customers like this one; not a known fact about this individual. Interval reflects final-stage sampling error only."},
            "optimisation": {"objective": cfg.objective, "z": cfg.z, "budget": cfg.budget, "agent_capacity": cfg.agent_capacity,
                             "value_per_event": cfg.value_per_event, "costs": cfg.costs},
        },
        "disclaimer": "Decision-support only. Not a determination of legal eligibility, suitability or regulatory compliance.",
    }
