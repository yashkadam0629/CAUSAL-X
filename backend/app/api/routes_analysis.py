"""Predictive models, causal analysis, uplift."""
from __future__ import annotations

import hashlib
import json
import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..causal.estimators import ASSUMPTIONS, ESTIMATORS, CausalSpec, compare_estimators, estimate, sensitivity
from ..data.schema import ARMS, CATEGORICAL, FEATURE_GROUPS, NUMERIC, OUTCOMES, TREATMENTS
from ..database.models import (CausalDiagnosticRow, CausalEstimateRow, Dataset, Experiment, ModelMetric, ModelRegistry, UpliftResultRow, get_db)
from ..services import registry
from ..services.pipeline import customer_segments, run_analysis, get_trained
from ..uplift.hte import SEGMENTS
from .deps import dataset_or_404, frame_for, json_safe as J, trained_or_409
from .routes_data import portfolio_table
from ..services.pipeline import default_opt_config

router = APIRouter()


# ------------------------------------------------------------------ predictive models
class TrainReq(BaseModel):
    dataset_id: int
    seed: int = 42
    force: bool = True


def _train_job(name: str, seed: int, force: bool, progress=lambda f, m: None):
    a = run_analysis(name, seed, progress=progress, force=force)
    from ..database.models import SessionLocal
    with SessionLocal() as db:
        ds = db.scalar(select(Dataset).where(Dataset.name == name))
        _register_models(db, ds, a)
    return {"model_version": a.model_version, "risk_model": a.risk_bundle.model_name}


def _register_models(db: Session, ds: Dataset, a):
    base = f"{a.model_version}"
    if db.scalar(select(ModelRegistry).where(ModelRegistry.model_id == f"{base}-risk")):
        return
    rb = a.risk_bundle
    m = ModelRegistry(model_id=f"{base}-risk", dataset_id=ds.id, kind="risk", name=rb.model_name, target=rb.target, config={**rb.config, "comparison": [
        {k: c[k] for k in ("model", "calibrated", "fit_seconds")} for c in rb.comparison], "dataset_fingerprint": a.fingerprint}, seed=a.seed,
        artifact_path=str(__import__("app.core.config", fromlist=["settings"]).settings.artifacts_dir / "models" / f"{ds.name}__seed{a.seed}" / "analysis.joblib"))
    db.add(m); db.flush()
    db.add(ModelMetric(model_pk=m.id, split="validation", metrics=J(rb.metrics["validation"])))
    db.add(ModelMetric(model_pk=m.id, split="test", metrics=J(rb.metrics["test"])))
    for arm, cm in a.cate.items():
        cr = ModelRegistry(model_id=f"{base}-cate-{arm}", dataset_id=ds.id, kind="cate", name=f"DR-learner {arm}", target=cm.outcome,
                           config={"method": cm.method, "meta": cm.meta, "features": cm.features, "training_cohorts": "0-7"}, seed=a.seed, artifact_path=m.artifact_path)
        db.add(cr); db.flush()
        ev = a.uplift_eval.get(arm, {})
        db.add(ModelMetric(model_pk=cr.id, split="test", metrics=J({k: v for k, v in ev.items() if k != "rankers"} | {"qini": {k: v["qini"] for k, v in ev.get("rankers", {}).items()}})))
        db.add(UpliftResultRow(dataset_id=ds.id, model_pk=cr.id, arm=arm, result=J(ev)))
    registry.audit(db, "models_trained", "model", base, {"dataset": ds.name, "seed": a.seed})
    db.commit()


@router.post("/models/train", status_code=202)
def train(req: TrainReq, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, req.dataset_id)
    if ds.source != "synthetic":
        raise HTTPException(422, "Model training on uploaded single-table data is not supported; see /datasets/{id}/validate capabilities.")
    return {"job_id": registry.submit_job("train_models", _train_job, ds.name, req.seed, req.force), "status": "queued"}


@router.get("/models")
def models(dataset_id: int | None = None, db: Session = Depends(get_db)):
    q = select(ModelRegistry).order_by(ModelRegistry.id.desc())
    if dataset_id:
        q = q.where(ModelRegistry.dataset_id == dataset_id)
    return J([{"model_id": m.model_id, "kind": m.kind, "name": m.name, "target": m.target, "dataset_id": m.dataset_id, "seed": m.seed, "created_at": m.created_at} for m in db.scalars(q)])


def _model_or_404(db, model_id):
    m = db.scalar(select(ModelRegistry).where(ModelRegistry.model_id == model_id))
    if not m:
        raise HTTPException(404, "model not found")
    return m


@router.get("/models/{model_id}")
def model(model_id: str, db: Session = Depends(get_db)):
    m = _model_or_404(db, model_id)
    return J({"model_id": m.model_id, "kind": m.kind, "name": m.name, "target": m.target, "dataset_id": m.dataset_id, "seed": m.seed, "config": m.config,
              "artifact_path": m.artifact_path, "created_at": m.created_at})


@router.get("/models/{model_id}/metrics")
def model_metrics(model_id: str, db: Session = Depends(get_db)):
    m = _model_or_404(db, model_id)
    out = {"model_id": m.model_id, "kind": m.kind,
           "metric_family": "predictive (discrimination, calibration, cost)" if m.kind == "risk" else "uplift / treatment-effect ranking (NOT predictive accuracy)",
           "metrics": {x.split: x.metrics for x in m.metrics}}
    if m.kind == "risk":
        ds = db.get(Dataset, m.dataset_id)
        a = get_trained(ds.name, m.seed)
        if a:
            out["comparison"] = a.risk_bundle.comparison
            out["importance"] = a.importance
    return J(out)


@router.get("/models/overview/{dataset_id}")
def models_overview(dataset_id: int, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    a = trained_or_409(ds)
    rb = a.risk_bundle
    return J({"model_version": a.model_version, "risk": {"chosen": rb.model_name, "config": rb.config, "metrics": rb.metrics, "comparison": rb.comparison, "importance": a.importance},
              "cate": {arm: {"method": m.method, **m.meta, "features": len(m.features)} for arm, m in a.cate.items()},
              "feature_groups": FEATURE_GROUPS, "split": rb.config["split"], "timings": a.timings,
              "metric_note": "Predictive metrics (risk) and uplift/treatment-effect metrics (CATE) answer different questions and are never combined."})


# ------------------------------------------------------------------ causal
class CausalReq(BaseModel):
    dataset_id: int
    treatment: str = "standard_reminder"
    comparison: str = "none"
    outcome: str = "delinquency_30d"
    covariates: list[str] | None = None
    cohorts: list[int] | None = None
    estimator: str = "aipw"
    bootstrap: int = Field(60, ge=0, le=300)
    seed: int = 42
    trim_low: float | None = Field(None, ge=0, lt=0.5)
    ps_model: str = "logistic"
    outcome_learner: str = "gbm"
    subgroup: str | None = Field(None, description="'high_baseline_risk' | 'low_baseline_risk' (needs trained models)")


def _spec(req: CausalReq, ds: Dataset, frame: pd.DataFrame) -> tuple[CausalSpec, pd.DataFrame, list[str]]:
    warns = []
    if req.estimator not in ESTIMATORS:
        raise HTTPException(422, f"estimator must be one of {ESTIMATORS}")
    arms_present = set(frame["arm"].unique()) if "arm" in frame else set()
    for arm in (req.treatment, req.comparison):
        if arm not in arms_present:
            raise HTTPException(422, f"arm '{arm}' not present in dataset (no treatment information => causal analysis not possible)")
    if req.treatment == req.comparison:
        raise HTTPException(422, "treatment and comparison must differ")
    if req.outcome not in frame.columns:
        raise HTTPException(422, f"outcome '{req.outcome}' not in dataset")
    if req.covariates:
        bad = [c for c in req.covariates if c not in frame.columns]
        if bad:
            raise HTTPException(422, f"unknown covariates {bad}")
        post = [c for c in req.covariates if c in OUTCOMES or c in ("arm", "intervention_cost", "delivery_status", "repayment_amount")]
        if post:
            raise HTTPException(422, f"post-treatment/outcome variables cannot be covariates: {post}")
    if req.subgroup:
        a = get_trained(ds.name, req.seed) if ds.source == "synthetic" else None
        if a is None:
            warns.append("Subgroup ignored: baseline risk requires trained models.")
        else:
            thr = np.quantile(a.risk, 2 / 3 if req.subgroup == "high_baseline_risk" else 1 / 3)
            frame = frame[(a.risk >= thr) if req.subgroup == "high_baseline_risk" else (a.risk <= thr)]
            warns.append(f"Population restricted to {req.subgroup.replace('_', ' ')} (model-predicted risk {'>=' if 'high' in req.subgroup else '<='} {thr:.2f}); "
                         "the estimand is now a subgroup effect.")
    spec = CausalSpec(treatment=req.treatment, comparison=req.comparison, outcome=req.outcome, covariates=req.covariates or [], cohorts=req.cohorts,
                      estimator=req.estimator, bootstrap=req.bootstrap, seed=req.seed, trim=(req.trim_low, 1 - req.trim_low) if req.trim_low else None,
                      ps_model=req.ps_model, outcome_learner=req.outcome_learner)
    return spec, frame, warns


def _truth_for(ds: Dataset, spec: CausalSpec, frame):
    if ds.source != "synthetic" or spec.comparison != "none" or spec.treatment not in TREATMENTS or spec.outcome != "delinquency_30d":
        return None
    from ..causal.estimators import prepare
    from ..data.frames import get_truth
    d, _, _ = prepare(frame, spec)
    return {"label": "SYNTHETIC GROUND TRUTH (benchmark only; never a model input)",
            "true_ate_itt": float(get_truth(ds.name).set_index("customer_id").loc[d.customer_id, f"itt_effect_{spec.treatment}"].mean())}


@router.post("/causal/estimate")
def causal_estimate(req: CausalReq, save: bool = Query(True), db: Session = Depends(get_db)):
    ds = dataset_or_404(db, req.dataset_id)
    frame = frame_for(ds)
    spec, fr, warns = _spec(req, ds, frame)
    res = estimate(fr, spec)
    res["extra_warnings"] = warns
    tr = _truth_for(ds, spec, fr)
    if tr and res.get("status") == "ok":
        res["synthetic_benchmark"] = tr | {"estimation_error": res["ate"] - tr["true_ate_itt"], "covered_by_ci": bool(res["ci_low"] <= tr["true_ate_itt"] <= res["ci_high"])}
    if res.get("status") == "ok" and ds.scenario:
        res["dataset_scenario"] = ds.scenario
    res["interpretability"] = _interpretability(res, ds)
    if save and res.get("status") == "ok":
        row = CausalEstimateRow(dataset_id=ds.id, treatment=spec.treatment, comparison=spec.comparison, outcome=spec.outcome, estimator=spec.estimator,
                                ate=res["ate"], ci_low=res["ci_low"], ci_high=res["ci_high"], n=res["n"], result=J(res))
        db.add(row); db.flush()
        db.add(CausalDiagnosticRow(estimate_pk=row.id, diagnostics=J(res["diagnostics"])))
        db.commit()
        res["estimate_id"] = row.id
    return J(res)


def _interpretability(res: dict, ds: Dataset) -> dict:
    """Plain-language reasons an estimate may not be causally interpretable."""
    why = []
    if res.get("status") != "ok":
        return {"status": "insufficient_evidence", "reasons": [res.get("reason", "not estimable")]}
    est = res["estimator"]
    logged = "randomised" in res.get("design_note", "")
    if est == "difference_in_means" and not logged:
        why.append("Unadjusted comparison without randomisation: treated and untreated customers differ systematically, so the gap mixes treatment effect with selection.")
    if not logged:
        why.append("Assignment was not randomised in the data. A causal reading requires that the covariates capture everything that drove both assignment and outcome (untestable).")
    why += res.get("diagnostics", {}).get("warnings", [])
    why.append("A narrow confidence interval only measures sampling noise; it says nothing about bias from unmeasured confounding or model misspecification.")
    status = "design_supported" if logged and not res["diagnostics"]["warnings"] else "conditional_on_assumptions"
    if est == "difference_in_means" and not logged:
        status = "association_only"
    return {"status": status, "reasons": why, "what_would_strengthen_it": ["A randomised holdout (see Experiment Designer)", "Negative-control / placebo checks (Sensitivity)", "Richer pre-treatment covariates"]}


@router.post("/causal/compare")
def causal_compare(req: CausalReq, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, req.dataset_id)
    spec, fr, warns = _spec(req, ds, frame_for(ds))
    rows = compare_estimators(fr, spec)
    tr = _truth_for(ds, spec, fr)
    return J({"estimators": rows, "synthetic_benchmark": tr, "warnings": warns,
              "note": "Agreement between estimators does not establish validity; they share the same identification assumptions."})


@router.post("/causal/diagnostics")
def causal_diagnostics(req: CausalReq, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, req.dataset_id)
    spec, fr, warns = _spec(req, ds, frame_for(ds))
    spec.estimator, spec.bootstrap = ("ipw" if req.estimator == "difference_in_means" else req.estimator), 0
    res = estimate(fr, spec)
    if res.get("status") != "ok":
        return J(res)
    return J({"population": res["population"], "diagnostics": res["diagnostics"], "warnings": res["warnings"] + warns, "spec": res["spec"]})


@router.post("/causal/sensitivity-analysis")
def causal_sensitivity(req: CausalReq, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, req.dataset_id)
    spec, fr, warns = _spec(req, ds, frame_for(ds))
    s = sensitivity(fr, spec)
    s["warnings"] = warns
    return J(s)


class ExperimentSave(BaseModel):
    dataset_id: int | None = None
    name: str = Field(..., min_length=2, max_length=200)
    kind: str = "causal"
    config: dict
    results: dict | None = None


@router.get("/causal/experiments")
def list_experiments(kind: str | None = None, db: Session = Depends(get_db)):
    q = select(Experiment).order_by(Experiment.id.desc())
    if kind:
        q = q.where(Experiment.kind == kind)
    return J([{"experiment_id": e.experiment_id, "name": e.name, "kind": e.kind, "dataset_id": e.dataset_id, "created_at": e.created_at, "manifest": e.manifest} for e in db.scalars(q)])


@router.post("/causal/experiments", status_code=201)
def save_experiment(req: ExperimentSave, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, req.dataset_id) if req.dataset_id else None
    v = ds.versions[-1] if ds and ds.versions else None
    eid = "exp-" + uuid.uuid4().hex[:10]
    manifest = {"config": req.config, "results": req.results, "dataset": ds.name if ds else None, "dataset_fingerprint": v.fingerprint if v else None,
                "assumptions": ASSUMPTIONS, "saved_at": __import__("datetime").datetime.utcnow().isoformat() + "Z"}
    manifest["manifest_hash"] = hashlib.sha256(json.dumps(J(manifest), sort_keys=True, default=str).encode()).hexdigest()
    db.add(Experiment(experiment_id=eid, dataset_id=ds.id if ds else None, kind=req.kind, name=req.name, manifest=J(manifest)))
    registry.audit(db, "experiment_saved", "experiment", eid, {"name": req.name, "kind": req.kind})
    db.commit()
    return {"experiment_id": eid, "manifest_hash": manifest["manifest_hash"]}


# ------------------------------------------------------------------ uplift
@router.post("/uplift/train", status_code=202)
def uplift_train(req: TrainReq, db: Session = Depends(get_db)):
    """Uplift (CATE) models are trained together with the risk models in one reproducible pipeline run."""
    return train(req, db)


@router.get("/uplift/metrics")
def uplift_metrics(dataset_id: int, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    a = trained_or_409(ds)
    return J({"evaluation": a.uplift_eval, "model_version": a.model_version,
              "explanation": "Uplift metrics rank customers by estimated treatment benefit using IPW-weighted held-out outcomes; classification accuracy is not used. "
                             "A Qini value above that of the risk-only ranker means targeting by estimated benefit adds value beyond targeting by risk."})


@router.get("/uplift/segments")
def uplift_segments(dataset_id: int, high: float = 0.08, moderate: float = 0.03, z: float = 1.28, scatter_n: int = 1500, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    a = trained_or_409(ds)
    d = a.frame[a.frame.split == "test"]
    seg = customer_segments(a, d, high, moderate, z)
    sc = a.scores.loc[d.index]
    arr = np.column_stack([sc[f"arr_{t}"].values for t in TREATMENTS])
    elig = np.column_stack([d[f"elig_{t}"].values for t in TREATMENTS])
    j = np.where(elig, arr, -np.inf).argmax(axis=1); r = np.arange(len(d))
    best = np.where(elig.any(axis=1), arr[r, j], np.nan)
    se = np.column_stack([sc[f"se_{t}"].values for t in TREATMENTS])[r, j]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(d), size=min(scatter_n, len(d)), replace=False)
    risk = a.risk[d.index]
    scatter = [{"customer_id": d.customer_id.iloc[i], "risk": float(risk[i]), "arr": None if np.isnan(best[i]) else float(best[i]), "se": float(se[i]),
                "action": TREATMENTS[j[i]], "segment": seg.iloc[i]} for i in pick if not np.isnan(best[i])]
    risk_med = float(np.nanmedian(risk))
    arr_med = float(np.nanmedian(best))
    quad = {"high_risk_high_benefit": int(((risk >= risk_med) & (best >= max(high, arr_med))).sum()), "high_risk_low_benefit": int(((risk >= risk_med) & (best < moderate)).sum()),
            "moderate_risk_high_benefit": int(((risk < risk_med) & (best >= high)).sum()), "low_risk_low_benefit": int(((risk < risk_med) & (best < moderate)).sum())}
    hist = np.histogram(best[~np.isnan(best)], bins=30)
    corr = float(np.corrcoef(risk[~np.isnan(best)], best[~np.isnan(best)])[0, 1])
    return J({"thresholds": {"high": high, "moderate": moderate, "z": z, "how_chosen": "analyst-configurable heuristics: benefit cutoffs on estimated ARR and an 80% interval (z=1.28) for the lower bound"},
              "counts": seg.value_counts().reindex(SEGMENTS, fill_value=0).to_dict(), "n": int(len(d)), "scatter": scatter,
              "benefit_histogram": {"edges": hist[1].tolist(), "counts": hist[0].tolist()}, "quadrants": quad,
              "risk_benefit_correlation": corr, "risk_median": risk_med,
              "note": "Segments are heuristic groupings of model estimates, not established categories. Individual values are estimates, not facts about a person."})
