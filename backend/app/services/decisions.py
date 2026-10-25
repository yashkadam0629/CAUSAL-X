"""Decision workflow: optimisation runs, persistence of recommendations, policy evaluation, fairness audit."""
from __future__ import annotations

import hashlib
import json
import uuid

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..core.config import timed
from ..data.schema import ARM_LABELS, ARMS, COVERAGE_GROUP, TREATMENTS
from ..database.models import DecisionRecommendation, OptimizationAllocation, OptimizationRun, Dataset
from ..explainability.explain import explain_customer
from ..governance.fairness import audit as fairness_audit
from ..optimization.engine import (OptConfig, Problem, candidate_mask, compare_strategies, optimise, STRATEGY_LABELS)
from ..policy_evaluation.ope import evaluate_policy
from ..simulation.scenarios import (budget_frontier, stress_test, value_of_information, winners_curse, monte_carlo)
from .pipeline import Analysis, build_problem, customer_segments, default_opt_config, naive_arr_matrix
from .registry import audit

CFG_FIELDS = ["budget", "agent_capacity", "channel_capacity", "per_intervention_cap", "max_contacts", "costs", "value_per_event",
              "available", "min_arr", "max_se", "z", "objective", "target_events", "coverage_min"]


def make_config(n_decision: int, over: dict | None) -> OptConfig:
    over = {k: v for k, v in (over or {}).items() if k in CFG_FIELDS and v is not None}
    if "costs" in over:
        base = default_opt_config(n_decision).costs
        base.update(over["costs"]); over["costs"] = base
    return default_opt_config(n_decision, **over)


def config_dict(c: OptConfig) -> dict:
    return {k: getattr(c, k) for k in CFG_FIELDS}


def policy_version(c: OptConfig) -> str:
    return "policy-" + hashlib.sha256(json.dumps(config_dict(c), sort_keys=True, default=str).encode()).hexdigest()[:8]


def exclusion_reasons(p: Problem, c: OptConfig, sel_i: np.ndarray) -> dict:
    n = len(p.ids)
    selected = np.zeros(n, bool); selected[sel_i] = True
    avail = np.array([a in c.available for a in p.arms])[None, :]
    elig = p.elig & avail
    adj = p.arr - c.z * p.se
    net = c.value_per_event * adj - np.array([c.costs[a] for a in p.arms])[None, :]
    no_elig = ~elig.any(axis=1)
    unc = ~no_elig & (c.max_se is not None) & ((elig & (p.se <= (c.max_se if c.max_se is not None else np.inf))).sum(axis=1) == 0)
    belowmin = ~no_elig & ~unc & ((elig & (adj >= c.min_arr)).sum(axis=1) == 0)
    nonet = ~no_elig & ~unc & ~belowmin & ((elig & (net > 0)).sum(axis=1) == 0)
    rest = ~selected & ~(no_elig | unc | belowmin | nonet)
    out = {"selected": int(selected.sum()), "no eligible/available action": int((no_elig & ~selected).sum()),
           "excluded by uncertainty cap (SE)": int((unc & ~selected).sum()),
           "below minimum estimated benefit": int((belowmin & ~selected).sum()),
           "no action with positive net benefit": int((nonet & ~selected).sum()),
           "positive net benefit but displaced by binding constraints": int(rest.sum())}
    return out


def run_optimization(a: Analysis, over: dict | None, cohorts=None, compare=True, seed=42, draws=50) -> dict:
    p, d = build_problem(a, cohorts)
    c = make_config(len(d), over)
    with timed("optimization run"):
        if compare:
            naive = naive_arr_matrix(a, d)
            comp = compare_strategies(p, c, seed=seed, random_draws=draws, conservative_z=1.28, naive_arr=naive)
            strategies, allocs = comp["strategies"], comp["allocations"]
            main = next(s for s in strategies if s["key"] == "optimised")
            si, sa = allocs["optimised"]
            if c.z > 0:  # user asked for a conservative primary run: report it as the main result
                s2, si, sa = optimise(p, c); main = s2
        else:
            main, si, sa = optimise(p, c)
            strategies, allocs = [main], {"optimised": (si, sa)}
    main["strategy_label"] = "Optimised (MILP)" if c.z == 0 else f"Conservative MILP (z={c.z})"
    out = {"config": config_dict(c), "policy_version": policy_version(c), "model_version": a.model_version, "dataset": a.dataset,
           "n_decision_customers": int(len(d)), "summary": main, "strategies": strategies,
           "exclusion_reasons": exclusion_reasons(p, c, si), "seed": seed,
           "decision_cohorts": sorted(d["cohort_month"].unique().tolist()),
           "assumption_provenance": {"costs": "analyst assumption", "value_per_event": "analyst assumption", "arr": "estimated (DR-learner)",
                                     "capacities": "analyst assumption", "eligibility": "rules applied to observed fields"},
           "limitations": ["Estimated ARR carries sampling uncertainty (SE) but not confounding or misspecification uncertainty.",
                           "Benefit is an expected value, not a guaranteed number of prevented events.",
                           "Costs are subtracted once, inside the net-benefit objective."]}
    return out, p, d, si, sa, allocs


def persist_run(db: Session, ds: Dataset, a: Analysis, res: dict, p: Problem, d: pd.DataFrame, si, sa, cfg: OptConfig) -> OptimizationRun:
    run_id = "run-" + uuid.uuid4().hex[:10]
    manifest = {"dataset": a.dataset, "dataset_fingerprint": a.fingerprint, "model_version": a.model_version, "seed": res["seed"],
                "config": res["config"], "policy_version": res["policy_version"], "feature_set": a.risk_bundle.features,
                "outcome": a.risk_bundle.target, "treatments": TREATMENTS, "decision_cohorts": res["decision_cohorts"]}
    run = OptimizationRun(run_id=run_id, dataset_id=ds.id, config=res["config"], summary=res["summary"], comparison=res["strategies"],
                          status=res["summary"]["solver_status"], policy_version=res["policy_version"], model_version=a.model_version, manifest=manifest)
    db.add(run); db.flush()
    cost = np.array([cfg.costs[x] for x in TREATMENTS])
    for i, k in zip(si, sa):
        arr, se = float(p.arr[i, k]), float(p.se[i, k])
        db.add(OptimizationAllocation(run_pk=run.id, customer_id=str(p.ids[i]), action=TREATMENTS[k], baseline_risk=float(p.risk[i]), arr=arr,
                                      arr_se=se, cost=float(cost[k]), net_benefit=arr * cfg.value_per_event - float(cost[k])))
    # recommendation records with a compact explanation (full SHAP explanation is computed on demand)
    max_expl = 5000
    for n_, (i, k) in enumerate(zip(si, sa)):
        if n_ >= max_expl:
            break
        cid = str(p.ids[i])
        ex = explain_customer(a, cid, cfg, TREATMENTS[k], {"binding_constraints": res["summary"]["binding_constraints"], "solver_status": res["summary"]["solver_status"],
                                                               "run_id": run_id}, with_shap=False)
        arr, se = float(p.arr[i, k]), float(p.se[i, k])
        needs = ex["human_review_required"]
        db.add(DecisionRecommendation(
            decision_id=f"{run_id}-{cid[-7:]}", run_pk=run.id, dataset_id=ds.id, customer_id=cid, recommended_action=TREATMENTS[k],
            eligibility_result={x: bool(p.elig[i, j]) for j, x in enumerate(TREATMENTS)}, estimated_incremental_benefit=arr * cfg.value_per_event,
            estimated_intervention_cost=float(cost[k]), expected_net_benefit=arr * cfg.value_per_event - float(cost[k]), effect_uncertainty=se,
            policy_reason=ex["selection_reason"], explanation=_slim(ex), model_version=a.model_version, policy_version=res["policy_version"],
            data_version=a.fingerprint, human_review_required=needs, human_review_status="pending_review" if needs else "not_required"))
    audit(db, "optimization_run", "optimization_run", run_id, {"policy_version": res["policy_version"], "n_selected": int(len(si)), "config": res["config"]})
    db.commit()
    return run


def _slim(ex: dict) -> dict:
    keep = {k: ex[k] for k in ("baseline_risk", "candidate_interventions", "selection_reason", "alternatives_not_selected", "constraints",
                               "what_would_change_the_decision", "human_review_reasons", "response_segment", "disclaimer")}
    keep["explanations_note"] = "SHAP drivers computed on demand via /customers/{id}/explanation"
    return keep


def evaluate_strategies(a: Analysis, over: dict | None, cohorts=None, seed=42) -> dict:
    p, d = build_problem(a, cohorts)
    c = make_config(len(d), over)
    if (d.split == "train").any():
        raise ValueError("policy evaluation requires held-out (non-training) cohorts")
    comp = compare_strategies(p, c, seed=seed, random_draws=30, naive_arr=naive_arr_matrix(a, d))
    rows = []
    pos = d.index.values
    for s in comp["strategies"]:
        si, sa = comp["allocations"].get(s["key"], (np.array([], int), np.array([], int)))
        action_idx = np.zeros(len(d), int)
        action_idx[si] = sa + 1  # ARMS index: 0=none, 1..3=treatments
        truth = s.get("synthetic_truth", {}).get("avoided_events") if a.has_truth else None
        ev = evaluate_policy(a.frame, a.nuisance, pos, action_idx, truth_arr_value=truth)
        rows.append({"strategy": s["strategy"], "n_selected": s["n_selected"], "budget_used": s["budget_used"],
                     "model_estimated_events_prevented": s["expected_avoided_events"],
                     "dr_estimated_events_prevented": ev["estimated_events_prevented"], "dr_ci": ev["events_prevented_ci"],
                     "dr_events_prevented_per_1000": ev["events_prevented_per_1000_customers"],
                     "synthetic_truth_events_prevented": ev.get("synthetic_truth_events_prevented"),
                     "dr_error_vs_truth": ev.get("estimation_error_vs_truth"), "clipped_share": ev["clipped_share"],
                     "overlap_with_logged_policy": ev["share_policy_matches_logged_arm"]})
    ref = ev
    return {"evaluation_cohorts": sorted(d["cohort_month"].unique().tolist()), "n": int(len(d)), "policies": rows,
            "method": ref["method"], "propensity_source": ref["propensity_source"], "assumptions": ref["assumptions"],
            "evidence_type": ("OBSERVATIONAL off-policy evaluation" if "estimated" in ref["propensity_source"] else "Off-policy evaluation with logged randomisation probabilities")
                              + (" + synthetic ground-truth benchmark" if a.has_truth else ""),
            "separation": {"model_training": "cohorts 0-7", "model_validation": "cohorts 8-9", "policy_optimisation_and_evaluation": "cohorts 10-11 (predictions only used for optimisation; logged outcomes only for evaluation)"},
            "limitations": ["DR estimates rest on the stated assumptions; they are not an experiment.",
                            "Strategies selecting arms rarely logged for similar customers have low effective sample sizes (see clipped share).",
                            "The same 30/50 random-draw baseline is an expectation, not a single draw."]}


def fairness_for(a: Analysis, over: dict | None, si, sa, p, d, group_col: str) -> dict:
    sel = np.zeros(len(d), bool); sel[si] = True
    best = np.where(p.elig, p.arr, -np.inf).max(axis=1)
    best = np.where(np.isfinite(best), best, 0.0)
    return fairness_audit(d.reset_index(drop=True), p.risk, sel, best, group_col)
