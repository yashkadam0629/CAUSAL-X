"""Optimisation, scenarios, policies, experiment design, governance, monitoring, overview, reports, assistant."""
from __future__ import annotations

import re
import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.config import settings
from ..data.schema import ARM_LABELS, TREATMENTS
from ..database.models import (AuditLog, DecisionRecommendation, DecisionReview, MonitoringSnapshot, OptimizationAllocation, OptimizationRun, Policy,
                               PolicyEvaluationRow, ScenarioRun, get_db)
from ..experiment_design.sample_size import design as design_calc, protocol as make_protocol
from ..optimization.engine import OptConfig
from ..services import decisions as dec
from ..services import registry
from ..services.pipeline import build_problem, default_opt_config
from ..simulation.scenarios import (SCENARIO_CATALOGUE, apply_overrides, budget_frontier, run_scenario, stress_test, value_of_information, winners_curse)
from .deps import dataset_or_404, json_safe as J, trained_or_409

router = APIRouter()


# ------------------------------------------------------------------ optimisation
class OptReq(BaseModel):
    dataset_id: int
    budget: float | None = Field(None, gt=0)
    agent_capacity: float | None = Field(None, ge=0)
    channel_capacity: int | None = Field(None, ge=0)
    per_intervention_cap: dict[str, int | None] | None = None
    max_contacts: int | None = Field(None, ge=0)
    costs: dict[str, float] | None = None
    value_per_event: float | None = Field(None, ge=0)
    available: list[str] | None = None
    min_arr: float | None = Field(None, ge=0, le=1)
    max_se: float | None = Field(None, gt=0)
    z: float | None = Field(None, ge=0, le=4)
    objective: str | None = Field(None, pattern="^(net_benefit|avoided_events|min_cost)$")
    target_events: float | None = Field(None, gt=0)
    coverage_min: dict[str, float] | None = None
    cohorts: list[int] | None = None
    compare: bool = True
    save: bool = True
    seed: int = 42

    def overrides(self) -> dict:
        d = self.model_dump(exclude={"dataset_id", "cohorts", "compare", "save", "seed"}, exclude_none=True)
        bad = [a for a in d.get("available", []) if a not in TREATMENTS]
        if bad:
            raise HTTPException(422, f"unknown interventions {bad}")
        return d


@router.post("/optimization/run")
def optimization_run(req: OptReq, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, req.dataset_id)
    a = trained_or_409(ds)
    over = req.overrides()
    res, p, d, si, sa, allocs = dec.run_optimization(a, over, req.cohorts, req.compare, req.seed)
    cfg = dec.make_config(len(d), over)
    st = res["summary"]["solver_status"]
    if st in ("infeasible", "unbounded", "solver_error"):
        res["advice"] = ["Relax the budget, agent/channel capacity, coverage minima or the target.", "Loosen the SE cap / minimum benefit filters."]
    if req.save and st in ("optimal", "time_limit_or_iteration_limit"):
        run = dec.persist_run(db, ds, a, res, p, d, si, sa, cfg)
        res["run_id"] = run.run_id
        res["fairness_audit"] = {g: dec.fairness_for(a, over, si, sa, p, d, g) for g in ("age_band", "region_group")}
    else:
        res["run_id"] = None
        res["saved"] = False
    res["allocation_preview"] = _alloc_preview(p, cfg, si, sa)
    return J(res)


def _alloc_preview(p, cfg, si, sa, n=30):
    order = np.argsort([-(p.arr[i, k] * cfg.value_per_event - cfg.costs[TREATMENTS[k]]) for i, k in zip(si, sa)])[:n] if len(si) else []
    return [{"customer_id": str(p.ids[si[o]]), "action": TREATMENTS[sa[o]], "risk": float(p.risk[si[o]]), "arr": float(p.arr[si[o], sa[o]]), "se": float(p.se[si[o], sa[o]]),
             "cost": cfg.costs[TREATMENTS[sa[o]]], "net_benefit": float(p.arr[si[o], sa[o]] * cfg.value_per_event - cfg.costs[TREATMENTS[sa[o]]])} for o in order]


@router.get("/optimization/runs")
def optimization_runs(dataset_id: int | None = None, db: Session = Depends(get_db)):
    q = select(OptimizationRun).order_by(OptimizationRun.id.desc())
    if dataset_id:
        q = q.where(OptimizationRun.dataset_id == dataset_id)
    return J([{"run_id": r.run_id, "status": r.status, "policy_version": r.policy_version, "model_version": r.model_version, "created_at": r.created_at,
               "n_selected": r.summary.get("n_selected"), "net_benefit": r.summary.get("net_benefit"), "budget_used": r.summary.get("budget_used")} for r in db.scalars(q)])


def _run_or_404(db, run_id):
    r = db.scalar(select(OptimizationRun).where(OptimizationRun.run_id == run_id))
    if not r:
        raise HTTPException(404, "run not found")
    return r


@router.get("/optimization/runs/{run_id}")
def optimization_run_detail(run_id: str, db: Session = Depends(get_db)):
    r = _run_or_404(db, run_id)
    return J({"run_id": r.run_id, "status": r.status, "config": r.config, "summary": r.summary, "strategies": r.comparison, "manifest": r.manifest,
              "policy_version": r.policy_version, "model_version": r.model_version, "created_at": r.created_at})


@router.get("/optimization/runs/{run_id}/allocation")
def optimization_allocation(run_id: str, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=500), action: str | None = None, db: Session = Depends(get_db)):
    r = _run_or_404(db, run_id)
    q = select(OptimizationAllocation).where(OptimizationAllocation.run_pk == r.id)
    if action:
        q = q.where(OptimizationAllocation.action == action)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(OptimizationAllocation.net_benefit.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return J({"total": total, "page": page, "items": [{"customer_id": x.customer_id, "action": x.action, "baseline_risk": x.baseline_risk, "arr": x.arr, "arr_se": x.arr_se,
                                                        "cost": x.cost, "net_benefit": x.net_benefit} for x in rows]})


# ------------------------------------------------------------------ scenarios
class ScenarioReq(BaseModel):
    dataset_id: int
    scenarios: list[str] | None = None  # keys of SCENARIO_CATALOGUE
    custom: dict | None = None  # {name, overrides}
    base: dict | None = None
    mc_draws: int = Field(200, ge=20, le=2000)
    save_name: str | None = None
    seed: int = 42
    include_frontier: bool = False


@router.get("/scenarios/catalogue")
def scenario_catalogue():
    return J(SCENARIO_CATALOGUE)


@router.post("/scenarios/run")
def scenarios_run(req: ScenarioReq, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, req.dataset_id)
    a = trained_or_409(ds)
    p, d = build_problem(a)
    c = dec.make_config(len(d), req.base)
    keys = req.scenarios or ["A_base"]
    unknown = [k for k in keys if k not in SCENARIO_CATALOGUE]
    if unknown:
        raise HTTPException(422, f"unknown scenarios {unknown}")
    results = [run_scenario(p, c, SCENARIO_CATALOGUE[k]["overrides"], SCENARIO_CATALOGUE[k]["label"], req.seed, req.mc_draws) for k in keys]
    if req.custom:
        ov = req.custom.get("overrides", {})
        results.append(run_scenario(p, c, ov, req.custom.get("name", "Custom scenario"), req.seed, req.mc_draws))
    out = {"base_config": dec.config_dict(c), "results": results, "model_version": a.model_version, "seed": req.seed,
           "uncertainty": f"Monte Carlo over independent Normal(ARR, SE^2) errors ({req.mc_draws} draws, seed {req.seed}); fixed allocation per scenario",
           "label": "SIMULATED quantities under stated assumptions (not observed outcomes)"}
    if req.include_frontier:
        out["budget_frontier"] = budget_frontier(p, c)
        out["capacity_frontier"] = budget_frontier(p, c, vary="agent_capacity")
    if req.save_name:
        sid = "scn-" + uuid.uuid4().hex[:10]
        db.add(ScenarioRun(scenario_id=sid, dataset_id=ds.id, name=req.save_name, config=J(req.model_dump()), result=J(out)))
        registry.audit(db, "scenario_saved", "scenario", sid, {"name": req.save_name}); db.commit()
        out["scenario_id"] = sid
    return J(out)


@router.get("/scenarios")
def scenarios_list(dataset_id: int | None = None, db: Session = Depends(get_db)):
    q = select(ScenarioRun).order_by(ScenarioRun.id.desc())
    if dataset_id:
        q = q.where(ScenarioRun.dataset_id == dataset_id)
    return J([{"scenario_id": s.scenario_id, "name": s.name, "created_at": s.created_at,
               "n_results": len(s.result.get("results", []))} for s in db.scalars(q)])


@router.get("/scenarios/{scenario_id}")
def scenario_get(scenario_id: str, db: Session = Depends(get_db)):
    s = db.scalar(select(ScenarioRun).where(ScenarioRun.scenario_id == scenario_id))
    if not s:
        raise HTTPException(404, "scenario not found")
    return J({"scenario_id": s.scenario_id, "name": s.name, "config": s.config, "result": s.result, "created_at": s.created_at})


class AnalysisReq(BaseModel):
    dataset_id: int
    base: dict | None = None
    seed: int = 42


@router.post("/scenarios/advanced")
def scenarios_advanced(req: AnalysisReq, db: Session = Depends(get_db)):
    """Budget/capacity frontier, stress test, optimiser's-curse check and value of information for the current configuration."""
    ds = dataset_or_404(db, req.dataset_id)
    a = trained_or_409(ds)
    p, d = build_problem(a)
    c = dec.make_config(len(d), req.base)
    from ..optimization.engine import compare_strategies, optimise
    comp = compare_strategies(p, c, random_draws=10, conservative_z=1.28)
    s, si, sa = optimise(p, c)
    extra = {"highest_risk_first": comp["allocations"]["highest_risk_first_intensive"], "conservative": comp["allocations"]["conservative"]}
    return J({"budget_frontier": budget_frontier(p, c), "capacity_frontier": budget_frontier(p, c, vary="agent_capacity"),
              "stress_test": stress_test(p, c, si, sa, extra), "winners_curse": winners_curse(p, c, trials=6, seed=req.seed),
              "value_of_information": value_of_information(p, c, seed=req.seed),
              "conservative_objective": "max sum x*(value*(ARR - z*SE) - cost): each estimate is replaced by its lower confidence bound (a box uncertainty set). "
                                        "It is conservative per item (stricter than the aggregate lower bound under independent errors) and guards against the winner's curse; "
                                        "z=1.28 corresponds to a one-sided ~90% bound per estimate."})


# ------------------------------------------------------------------ policies
class PolicyReq(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    definition: dict


@router.post("/policies", status_code=201)
def policy_create(req: PolicyReq, db: Session = Depends(get_db)):
    ver = 1 + (db.scalar(select(func.count()).select_from(Policy).where(Policy.name == req.name)) or 0)
    pid = "pol-" + uuid.uuid4().hex[:8]
    db.add(Policy(policy_id=pid, name=req.name, definition=req.definition, version=f"v{ver}"))
    registry.audit(db, "policy_created", "policy", pid, {"name": req.name, "version": f"v{ver}"}); db.commit()
    return {"policy_id": pid, "version": f"v{ver}"}


@router.get("/policies")
def policy_list(db: Session = Depends(get_db)):
    return J([{"policy_id": p.policy_id, "name": p.name, "version": p.version, "definition": p.definition, "created_at": p.created_at} for p in db.scalars(select(Policy).order_by(Policy.id.desc()))])


class PolicyEvalReq(BaseModel):
    dataset_id: int
    base: dict | None = None
    seed: int = 42
    policy_id: str | None = None


@router.post("/policies/evaluate")
def policy_evaluate(req: PolicyEvalReq, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, req.dataset_id)
    a = trained_or_409(ds)
    base = req.base
    pol = None
    if req.policy_id:
        pol = db.scalar(select(Policy).where(Policy.policy_id == req.policy_id))
        if not pol:
            raise HTTPException(404, "policy not found")
        base = pol.definition
    out = dec.evaluate_strategies(a, base, None, req.seed)
    db.add(PolicyEvaluationRow(policy_pk=pol.id if pol else None, dataset_id=ds.id, result=J(out))); db.commit()
    return J(out)


# ------------------------------------------------------------------ experiment design
class SampleSizeReq(BaseModel):
    outcome_type: str = Field("binary", pattern="^(binary|continuous)$")
    baseline: float = Field(0.25, gt=0)
    mde_absolute: float
    alpha: float = Field(0.05, gt=0, lt=1)
    power: float = Field(0.8, gt=0, lt=1)
    allocation_ratio: float = Field(1.0, gt=0)
    comparisons_vs_control: int = Field(1, ge=1, le=10)
    attrition: float = Field(0.0, ge=0, lt=0.9)
    compliance: float = Field(1.0, gt=0.05, le=1)
    icc: float = Field(0.0, ge=0, le=1)
    cluster_size: int = Field(1, ge=1)
    sd: float | None = None


@router.post("/experiments/sample-size")
def sample_size(req: SampleSizeReq):
    try:
        return J(design_calc(req.outcome_type, req.baseline, req.mde_absolute, req.alpha, req.power, req.allocation_ratio, req.comparisons_vs_control,
                             req.attrition, req.compliance, req.icc, req.cluster_size, req.sd))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


class DesignReq(SampleSizeReq):
    objective: str = "Estimate the causal effect of the intervention on 30-day delinquency"
    target_population: str = "Active-loan customers eligible for the intervention"
    eligibility: str = "Rules documented in the data dictionary"
    treatment: str = "standard_reminder"
    primary_outcome: str = "delinquency_30d"
    horizon_days: int = 30
    duration_days: int | None = 60
    secondary_outcomes: list[str] = []
    stratification: list[str] = ["loan_type", "risk_band"]
    name: str | None = None
    dataset_id: int | None = None


@router.post("/experiments/design")
def experiment_design(req: DesignReq, db: Session = Depends(get_db)):
    try:
        calc = design_calc(req.outcome_type, req.baseline, req.mde_absolute, req.alpha, req.power, req.allocation_ratio, req.comparisons_vs_control,
                           req.attrition, req.compliance, req.icc, req.cluster_size, req.sd)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    cfg = req.model_dump()
    cfg["allocation_ratio"] = req.allocation_ratio
    proto = make_protocol(cfg, calc)
    risks = []
    if req.icc > 0 or req.cluster_size > 1:
        risks.append("Clustering/interference assumed via design effect; confirm the cluster definition (household, agent, branch).")
    if req.compliance < 1:
        risks.append(f"Noncompliance assumed ({req.compliance:.0%}): analysis must be ITT; the MDE refers to assignment.")
    if req.comparisons_vs_control > 1:
        risks.append("Multiple comparisons: Bonferroni applied; pre-register the primary comparison.")
    if req.attrition > 0:
        risks.append("Incomplete outcome observation: define a pre-specified handling rule; do not impute favourable outcomes.")
    proto["additional_risks"] = risks
    if req.name:
        from ..database.models import Experiment
        eid = "exp-" + uuid.uuid4().hex[:10]
        db.add(Experiment(experiment_id=eid, dataset_id=req.dataset_id, kind="design", name=req.name, manifest=J(proto)))
        registry.audit(db, "experiment_design_saved", "experiment", eid, {"name": req.name}); db.commit()
        proto["experiment_id"] = eid
    return J(proto)


# ------------------------------------------------------------------ governance
@router.get("/audit/decisions")
def audit_decisions(dataset_id: int | None = None, status: str | None = None, action: str | None = None, customer_id: str | None = None, run_id: str | None = None,
                    page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=200), db: Session = Depends(get_db)):
    q = select(DecisionRecommendation)
    if dataset_id:
        q = q.where(DecisionRecommendation.dataset_id == dataset_id)
    if status:
        q = q.where(DecisionRecommendation.human_review_status == status)
    if action:
        q = q.where(DecisionRecommendation.recommended_action == action)
    if customer_id:
        q = q.where(DecisionRecommendation.customer_id.contains(customer_id))
    if run_id:
        r = _run_or_404(db, run_id)
        q = q.where(DecisionRecommendation.run_pk == r.id)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(DecisionRecommendation.id.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    items = []
    for r in rows:
        revs = db.scalars(select(DecisionReview).where(DecisionReview.decision_pk == r.id).order_by(DecisionReview.id)).all()
        items.append({"decision_id": r.decision_id, "customer_id": r.customer_id, "recommended_action": r.recommended_action, "net_benefit": r.expected_net_benefit,
                      "benefit": r.estimated_incremental_benefit, "cost": r.estimated_intervention_cost, "uncertainty": r.effect_uncertainty, "reason": r.policy_reason,
                      "model_version": r.model_version, "policy_version": r.policy_version, "data_version": r.data_version[:12], "timestamp": r.recommendation_timestamp,
                      "human_review_required": r.human_review_required, "status": r.human_review_status, "explanation": r.explanation,
                      "reviews": [{"reviewer": v.reviewer, "outcome": v.outcome, "original_action": v.original_action, "final_action": v.final_action, "reason": v.reason, "reviewed_at": v.reviewed_at} for v in revs]})
    return J({"total": total, "page": page, "items": items})


class ReviewReq(BaseModel):
    reviewer: str = Field(..., min_length=2, max_length=120)
    outcome: str = Field(..., pattern="^(approve|override|reject)$")
    final_action: str | None = None
    reason: str = Field(..., min_length=3, max_length=2000)


@router.post("/audit/decisions/{decision_id}/review")
def review_decision(decision_id: str, req: ReviewReq, db: Session = Depends(get_db)):
    r = db.scalar(select(DecisionRecommendation).where(DecisionRecommendation.decision_id == decision_id))
    if not r:
        raise HTTPException(404, "decision not found")
    final = r.recommended_action
    if req.outcome == "override":
        if req.final_action not in (*TREATMENTS, "none"):
            raise HTTPException(422, "override requires final_action in " + str([*TREATMENTS, "none"]))
        if req.final_action != "none" and not r.eligibility_result.get(req.final_action, False):
            raise HTTPException(422, f"override rejected: customer is not eligible for {req.final_action} under the configured eligibility rules")
        final = req.final_action
    elif req.outcome == "reject":
        final = "none"
    db.add(DecisionReview(decision_pk=r.id, reviewer=req.reviewer, outcome=req.outcome, original_action=r.recommended_action, final_action=final, reason=req.reason))
    r.human_review_status = {"approve": "approved", "override": "overridden", "reject": "rejected"}[req.outcome]
    registry.audit(db, "decision_reviewed", "decision", decision_id, {"reviewer": req.reviewer, "outcome": req.outcome, "original": r.recommended_action, "final": final, "reason": req.reason})
    db.commit()
    return J({"decision_id": decision_id, "status": r.human_review_status, "original_action": r.recommended_action, "final_action": final})


@router.get("/audit/log")
def audit_log(event: str | None = None, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    q = select(AuditLog)
    if event:
        q = q.where(AuditLog.event == event)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.scalars(q.order_by(AuditLog.id.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return J({"total": total, "items": [{"id": x.id, "event": x.event, "entity": x.entity, "entity_id": x.entity_id, "detail": x.detail, "created_at": x.created_at} for x in rows]})


@router.get("/audit/decisions/export", response_class=PlainTextResponse)
def audit_export(dataset_id: int | None = None, db: Session = Depends(get_db)):
    import csv, io
    q = select(DecisionRecommendation).order_by(DecisionRecommendation.id)
    if dataset_id:
        q = q.where(DecisionRecommendation.dataset_id == dataset_id)
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["decision_id", "customer_id", "action", "benefit", "cost", "net_benefit", "uncertainty_se", "model_version", "policy_version", "data_version", "status", "timestamp", "reason"])
    for r in db.scalars(q):
        w.writerow([r.decision_id, r.customer_id, r.recommended_action, r.estimated_incremental_benefit, r.estimated_intervention_cost, r.expected_net_benefit,
                    r.effect_uncertainty, r.model_version, r.policy_version, r.data_version, r.human_review_status, r.recommendation_timestamp, r.policy_reason])
    return PlainTextResponse(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=decisions.csv"})


# ------------------------------------------------------------------ monitoring
@router.get("/monitoring/drift")
def monitoring_drift(dataset_id: int, db: Session = Depends(get_db)):
    a = trained_or_409(dataset_or_404(db, dataset_id))
    return J(a.drift)


@router.get("/monitoring/summary")
def monitoring_summary(dataset_id: int, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    a = trained_or_409(ds)
    last = a.drift["monthly"][-1]
    snap = {"model_version": a.model_version, "warnings": a.drift["warnings"], "prediction_status": a.drift["prediction_status"],
            "n_significant_feature_drifts": sum(f["status"] == "significant" for f in a.drift["feature_drift"]),
            "latest_cohort": last, "effect_stability": a.effect_stability,
            "effect_stability_note": "Reminder ATE (AIPW, logistic nuisance) by period. Changes can reflect shifts in who is treated, delivery or true effectiveness; they are not proof of any one cause.",
            "recommended_investigation": [
                "Check upstream data feeds for the drifting features.", "Review calibration by cohort before relying on absolute risk values.",
                "Re-estimate effects on recent cohorts; retraining requires an approved change (no automatic redeploy)."]}
    db.add(MonitoringSnapshot(dataset_id=ds.id, snapshot=J(snap))); db.commit()
    return J(snap)


# ------------------------------------------------------------------ overview (executive command centre)
@router.get("/overview")
def overview(dataset_id: int, run_id: str | None = None, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    a = trained_or_409(ds)
    if run_id:
        r = _run_or_404(db, run_id)
        res = {"config": r.config, "summary": r.summary, "strategies": r.comparison, "policy_version": r.policy_version, "run_id": r.run_id}
    else:
        memo = a.__dict__.setdefault("_memo", {})  # default-configuration overview is deterministic per model version
        if "overview_default" not in memo:
            o, p, d, si, sa, allocs = dec.run_optimization(a, None, None, True, 42, 30)
            memo["overview_default"] = {"config": o["config"], "summary": o["summary"], "strategies": o["strategies"], "policy_version": o["policy_version"],
                                        "run_id": None, "exclusion_reasons": o["exclusion_reasons"]}
        res = memo["overview_default"]
    p, d = build_problem(a)
    ev = a.risk_bundle.metrics["test"]
    hist = np.histogram(a.risk[d.index], bins=20, range=(0, 1))
    best = np.where(p.elig, p.arr, -np.inf).max(axis=1)
    bh = np.histogram(best[np.isfinite(best)], bins=25)
    cohorts = a.drift["monthly"]
    from ..database.models import DataQualityReport
    q = db.scalar(select(func.max(DataQualityReport.score)).where(DataQualityReport.dataset_id == ds.id))
    summ = res["summary"]
    unc = {"mean_se_selected": (summ["events_se_independent"] / max(summ["n_selected"], 1) ** 0.5) if summ["n_selected"] else None}
    return J({
        "dataset": ds.name, "scenario": ds.scenario, "model_version": a.model_version, "policy_version": res["policy_version"], "run_id": res["run_id"],
        "kpis": {"total_customers": int(len(a.frame)), "decision_customers": int(len(d)), "eligible_customers": int(p.elig.any(axis=1).sum()),
                 "baseline_risk": float(a.risk[d.index].mean()), "observed_adverse_rate_decision_cohorts": float(d["delinquency_30d"].mean()),
                 "budget": res["config"]["budget"], "budget_used": summ["budget_used"], "budget_utilisation": summ["budget_used"] / res["config"]["budget"],
                 "n_recommended": summ["n_selected"], "expected_avoided_events": summ["expected_avoided_events"], "avoided_events_se": summ["events_se_independent"],
                 "gross_benefit": summ["gross_benefit"], "net_benefit": summ["net_benefit"], "data_quality_score": q, "risk_model_test_auc": ev["roc_auc"]},
        "risk_histogram": {"edges": hist[1].tolist(), "counts": hist[0].tolist()},
        "benefit_histogram": {"edges": bh[1].tolist(), "counts": bh[0].tolist()},
        "strategies": res["strategies"], "mix": summ["mix"], "monthly": cohorts, "warnings": a.drift["warnings"], "config": res["config"],
        "uncertainty_summary": unc, "exclusion_reasons": res.get("exclusion_reasons"),
        "labels": {"baseline_risk": "predicted", "expected_avoided_events": "estimated (causal model)", "net_benefit": "estimated x analyst assumptions", "strategies": "simulated policy comparison on held-out cohorts"},
    })


# ------------------------------------------------------------------ analyst assistant (rule-based, shows config before execution)
class AssistReq(BaseModel):
    dataset_id: int
    question: str = Field(..., min_length=5, max_length=500)


@router.post("/assistant/parse")
def assistant_parse(req: AssistReq, db: Session = Depends(get_db)):
    """Translate a question into a *proposed* analysis configuration using keyword rules (no LLM, no results are invented)."""
    q = req.question.lower()
    notes = []
    treat = next((t for t, kws in {"counselling": ["counsel"], "repayment_plan": ["plan", "restructur", "repayment-date", "repayment date"],
                                   "standard_reminder": ["remind", "sms", "nudge"]}.items() if any(k in q for k in kws)), None)
    if not treat:
        treat = "standard_reminder"; notes.append("No intervention recognised: defaulted to standard_reminder.")
    outcome = "delinquency_30d"
    if "60" in q:
        outcome = "delinquency_60d"
    elif "default" in q:
        outcome = "default_event"
    elif "missed" in q:
        outcome = "missed_payment"
    est = next((e for e, kws in {"ipw": ["weight", "ipw"], "regression_adjustment": ["regression"], "difference_in_means": ["raw", "naive", "unadjusted"]}.items() if any(k in q for k in kws)), "aipw")
    sub = "high_baseline_risk" if re.search(r"high.{0,12}risk", q) else ("low_baseline_risk" if re.search(r"low.{0,12}risk", q) else None)
    if re.search(r"\bcompare\b.*\bagainst\b|\bvs\b|versus", q) and "no intervention" not in q and "no additional" not in q:
        notes.append("Comparison assumed to be no additional intervention.")
    cfg = {"dataset_id": req.dataset_id, "treatment": treat, "comparison": "none", "outcome": outcome, "estimator": est, "subgroup": sub,
           "restrict_to_eligible": True}
    return {"proposed_configuration": cfg, "notes": notes, "executed": False,
            "message": "This is a proposed configuration only. Review it, then run it in the Causal Lab; nothing has been estimated and no result is implied."}
