"""Health, configuration, datasets, validation, customers/portfolio, jobs."""
from __future__ import annotations

import io
import json
import re
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.config import settings
from ..data import frames
from ..data.schema import (ARM_LABELS, AUDIT_ONLY, COVERAGE_GROUP, DEFAULT_COSTS, DEFAULT_VALUE_PER_EVENT, FEATURE_GROUPS, FEATURES, PROVENANCE, TREATMENTS)
from ..data.synthetic import SCENARIOS, SIZE_PRESETS
from ..data.validation import inspect_upload, validate_tables, validate_upload
from ..database.models import DataQualityReport, Dataset, DecisionRecommendation, Job, get_db
from ..services import registry
from ..services.pipeline import customer_segments, default_opt_config, get_trained, run_analysis
from .deps import dataset_or_404, frame_for, json_safe as J, trained_or_409

router = APIRouter()
STAGING = settings.upload_dir / "_staging"


# ------------------------------------------------------------------ health & config
@router.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(select(1))
    return {"status": "ok", "app": settings.app_name, "version": settings.version, "database": "sqlite" if settings.is_sqlite else "postgresql",
            "time": datetime.now(timezone.utc).isoformat()}


@router.get("/config")
def config():
    return J({"default_seed": settings.default_seed, "default_n_customers": settings.default_n_customers, "size_presets": SIZE_PRESETS,
              "scenarios": list(SCENARIOS), "max_upload_mb": settings.max_upload_mb, "treatments": TREATMENTS, "arm_labels": ARM_LABELS,
              "default_costs": DEFAULT_COSTS, "default_value_per_event": DEFAULT_VALUE_PER_EVENT, "provenance_legend": PROVENANCE,
              "database": {"engine": "sqlite" if settings.is_sqlite else "postgresql",
                           "note": "SQLite is a lightweight dev default; it does not enforce column types/concurrency like PostgreSQL."},
              "security_model": "Single local user, no authentication. Do not expose this service to untrusted networks."})


@router.get("/metadata")
def metadata():
    return J({"feature_groups": FEATURE_GROUPS, "features": FEATURES, "audit_only_attributes": AUDIT_ONLY, "coverage_group": COVERAGE_GROUP,
              "sign_convention": "tau = E[Y(treat)-Y(comparison)] on an adverse outcome (negative = beneficial); ARR = -tau (positive = beneficial).",
              "estimand": "Intention-to-treat effect of assignment",
              "definitions": {
                  "baseline_risk": "Predicted probability of the adverse outcome under historical practice (prediction).",
                  "arr": "Estimated absolute risk reduction vs no additional intervention (estimated causal effect, uncertain).",
                  "gross_benefit": "ARR x value per avoided event (value is an analyst assumption).",
                  "net_benefit": "Gross benefit - intervention cost (cost counted once).",
                  "budget_utilisation": "Intervention spend / available budget."}})


# ------------------------------------------------------------------ jobs
@router.get("/jobs/{job_id}")
def job(job_id: str, db: Session = Depends(get_db)):
    j = db.scalar(select(Job).where(Job.job_id == job_id))
    if not j:
        raise HTTPException(404, "job not found")
    return J({"job_id": j.job_id, "kind": j.kind, "status": j.status, "progress": j.progress, "message": j.message, "result": j.result})


# ------------------------------------------------------------------ datasets
class GenerateReq(BaseModel):
    name: str = Field(..., pattern=r"^[A-Za-z0-9_\-]{1,60}$")
    size: str | None = Field(None, description="small | medium | large preset")
    n_customers: int | None = Field(None, ge=1000, le=500_000)
    scenario: str = "observational"
    seed: int = 42
    train: bool = True


def _gen_job(req: dict, progress=lambda f, m: None):
    progress(0.05, "Generating synthetic portfolio")
    man = registry.generate_dataset(req["name"], req["n_customers"], req["scenario"], req["seed"])
    if req["train"]:
        from .routes_analysis import _train_job  # trains and registers models (analysis seed = configured default)
        _train_job(req["name"], settings.default_seed, False, progress=lambda f, m: progress(0.1 + 0.9 * f, m))
    return {"name": req["name"], "fingerprint": man["fingerprint"]}


@router.post("/datasets/generate", status_code=202)
def generate(req: GenerateReq, db: Session = Depends(get_db)):
    if req.scenario not in SCENARIOS:
        raise HTTPException(422, f"scenario must be one of {SCENARIOS}")
    n = req.n_customers or SIZE_PRESETS.get(req.size or "", settings.default_n_customers)
    if db.scalar(select(Dataset).where(Dataset.name == req.name)):
        raise HTTPException(409, f"dataset '{req.name}' already exists (delete it first or choose another name)")
    job_id = registry.submit_job("generate_dataset", _gen_job, {"name": req.name, "n_customers": n, "scenario": req.scenario, "seed": req.seed, "train": req.train})
    return {"job_id": job_id, "status": "queued", "n_customers": n}


def _ds_row(ds: Dataset):
    v = ds.versions[-1] if ds.versions else None
    a = get_trained(ds.name) if ds.source == "synthetic" else None
    return {"id": ds.id, "name": ds.name, "source": ds.source, "scenario": ds.scenario, "n_customers": ds.n_customers,
            "created_at": ds.created_at, "fingerprint": v.fingerprint if v else None, "versions": len(ds.versions),
            "models_trained": a is not None, "model_version": a.model_version if a else None}


@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    registry.sync_datasets(db)
    return J([_ds_row(d) for d in db.scalars(select(Dataset).order_by(Dataset.id))])


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    row = _ds_row(ds)
    row["version_history"] = [{"version": v.version, "fingerprint": v.fingerprint, "created_at": v.created_at, "row_counts": v.row_counts, "config": v.config} for v in ds.versions]
    if ds.source == "synthetic":
        row["manifest"] = frames.manifest(ds.name)
    return J(row)


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, confirm: bool = Query(False), db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    if not confirm:
        raise HTTPException(400, "Deletion is permanent. Re-send with ?confirm=true to delete the dataset, its models and results.")
    registry.delete_dataset(db, ds)
    return {"deleted": ds.name}


@router.post("/datasets/{dataset_id}/validate")
def validate(dataset_id: int, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    if ds.source == "synthetic":
        rep = validate_tables(frames.load_tables(ds.name))
    else:
        man = json.loads((settings.upload_dir / ds.name / "manifest.json").read_text())
        rep = man["validation"]
    row = DataQualityReport(dataset_id=ds.id, score=rep["quality_score"], report=J(rep))
    db.add(row); registry.audit(db, "dataset_validated", "dataset", ds.id, {"score": rep["quality_score"]}); db.commit()
    return J(rep)


@router.get("/datasets/{dataset_id}/profile")
def profile(dataset_id: int, db: Session = Depends(get_db)):
    from ..data.validation import profile as prof
    ds = dataset_or_404(db, dataset_id)
    df = frame_for(ds)
    cols = [c for c in df.columns if not c.startswith(("elig_",))]
    return J({"n_rows": len(df), "n_columns": len(cols), "columns": prof(df[cols])})


@router.get("/datasets/{dataset_id}/quality-report")
def quality_report(dataset_id: int, download: bool = False, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    r = db.scalar(select(DataQualityReport).where(DataQualityReport.dataset_id == ds.id).order_by(DataQualityReport.id.desc()))
    if r is None:
        raise HTTPException(404, "no quality report yet; POST /datasets/{id}/validate")
    from fastapi.responses import JSONResponse
    resp = JSONResponse(J({"dataset": ds.name, "generated_at": r.created_at, "score": r.score, **r.report}))
    if download:
        resp.headers["Content-Disposition"] = f'attachment; filename="{ds.name}_quality_report.json"'
    return resp


# ------------------------------------------------------------------ upload (two-step: stage+inspect, then commit with mapping)
@router.post("/datasets/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "Only .csv files are accepted")
    raw = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception as e:
        raise HTTPException(422, f"Could not parse CSV: {type(e).__name__}") from e
    if df.empty or len(df.columns) < 2:
        raise HTTPException(422, "CSV is empty or has fewer than 2 columns")
    STAGING.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:16]
    df.to_parquet(STAGING / f"{token}.parquet", index=False)
    return J({"upload_token": token, "filename": re.sub(r"[^\w.\-]", "_", file.filename), **inspect_upload(df)})


class CommitReq(BaseModel):
    upload_token: str = Field(..., pattern=r"^[0-9a-f]{16}$")
    name: str = Field(..., pattern=r"^[A-Za-z0-9_\-]{1,60}$")
    mapping: dict[str, str | None]
    covariates: list[str] | None = None
    control_label: str | None = None


@router.post("/datasets/upload/commit")
def commit_upload(req: CommitReq, db: Session = Depends(get_db)):
    p = STAGING / f"{req.upload_token}.parquet"
    if not p.exists():
        raise HTTPException(404, "upload token not found or expired")
    if db.scalar(select(Dataset).where(Dataset.name == req.name)):
        raise HTTPException(409, "dataset name already exists")
    df = pd.read_parquet(p)
    try:
        rep = validate_upload(df, req.mapping, req.covariates, req.control_label)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    if any(f["severity"] == "error" for f in rep["findings"] if f["code"] in ("no_customer_id",)):
        raise HTTPException(422, "A customer_id column must be mapped")
    m = {k: v for k, v in req.mapping.items() if v}
    d = df.drop_duplicates(m["customer_id"]) if "customer_id" in m else df.copy()
    out = pd.DataFrame({"customer_id": d[m["customer_id"]].astype(str) if "customer_id" in m else np.arange(len(d)).astype(str)})
    if "decision_date" in m:
        dt = pd.to_datetime(d[m["decision_date"]], errors="coerce"); keep = dt.notna().values
        d, out, dt = d[keep], out[keep], dt[keep]
        out["decision_date"] = dt.values
        out["cohort_month"] = ((dt.dt.year - dt.dt.year.min()) * 12 + dt.dt.month - dt.dt.month.min()).values
    if "outcome" in m:
        y = pd.to_numeric(d[m["outcome"]], errors="coerce"); keep = y.notna().values
        d, out, y = d[keep], out[keep], y[keep]
        out[m["outcome"]] = y.values
    if "treatment" in m:
        out["arm"] = d[m["treatment"]].astype(str).values
        if req.control_label:
            out["arm"] = out["arm"].replace({req.control_label: "none"})
    covs = req.covariates or [c for c in d.columns if c not in m.values()]
    for c in covs:
        if c in d:
            out[c] = d[c].values
    out = out.reset_index(drop=True)
    d_dir = settings.upload_dir / req.name
    d_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d_dir / "frame.parquet", index=False)
    fp = __import__("hashlib").sha256(pd.util.hash_pandas_object(out.astype(str), index=False).values.tobytes()).hexdigest()
    man = {"name": req.name, "source": "upload", "fingerprint": fp, "rows": {"customers": int(len(out))}, "config": {"mapping": req.mapping, "covariates": covs,
           "control_label": req.control_label}, "capabilities": rep["capabilities"], "validation": J(rep), "outcome_column": m.get("outcome")}
    (d_dir / "manifest.json").write_text(json.dumps(J(man), default=str), encoding="utf-8")
    p.unlink(missing_ok=True)
    registry.sync_datasets(db)
    ds = db.scalar(select(Dataset).where(Dataset.name == req.name))
    db.add(DataQualityReport(dataset_id=ds.id, score=rep["quality_score"], report=J(rep)))
    registry.audit(db, "dataset_uploaded", "dataset", ds.id, {"rows_in": rep["rows_in"], "rows": len(out), "exclusions": rep["exclusions"]})
    db.commit()
    return J({"dataset_id": ds.id, "validation": rep})


# ------------------------------------------------------------------ customers / portfolio
SORTABLE = {"risk", "best_arr", "best_net", "outstanding_balance", "debt_to_income_ratio", "customer_id", "uncertainty"}


def portfolio_table(a, cfg) -> pd.DataFrame:
    memo = a.__dict__.setdefault("_memo", {})
    key = ("ptable", a.model_version, cfg.budget, cfg.value_per_event, tuple(sorted(cfg.costs.items())))
    if key not in memo:
        memo[key] = _portfolio_table(a, cfg)
    return memo[key]


def _portfolio_table(a, cfg) -> pd.DataFrame:
    fr, sc = a.frame, a.scores
    arr = np.column_stack([sc[f"arr_{t}"].values for t in TREATMENTS])
    se = np.column_stack([sc[f"se_{t}"].values for t in TREATMENTS])
    elig = np.column_stack([fr[f"elig_{t}"].values for t in TREATMENTS])
    cost = np.array([cfg.costs[t] for t in TREATMENTS])
    net = np.where(elig, cfg.value_per_event * arr - cost[None, :], -np.inf)
    j = net.argmax(axis=1); r = np.arange(len(fr))
    best_net = np.where(np.isfinite(net[r, j]), net[r, j], np.nan)
    unc = np.where(arr[r, j] - 1.28 * se[r, j] > 0, np.where(se[r, j] < 0.02, "low", "moderate"), "high")
    t = pd.DataFrame({
        "customer_id": fr.customer_id, "cohort_month": fr.cohort_month, "split": fr.split, "loan_type": fr.loan_type, "customer_segment": fr.customer_segment,
        "region_group": fr.region_group, "outstanding_balance": fr.outstanding_balance, "debt_to_income_ratio": fr.debt_to_income_ratio,
        "historical_delinquency_count": fr.historical_delinquency_count, "risk": a.risk, "best_action": np.where(np.isfinite(net[r, j]) & (net[r, j] > 0), np.array(TREATMENTS)[j], "none"),
        "best_arr": np.where(elig.any(axis=1), arr[r, j], np.nan), "best_cost": cost[j], "best_net": best_net, "uncertainty": unc,
        "eligible_any": elig.any(axis=1), "eligible_actions": [",".join(t for t, e in zip(TREATMENTS, row) if e) for row in elig],
        "data_completeness": 1 - fr[["monthly_income", "credit_history_length"]].isna().mean(axis=1),
        "observed_outcome": fr["delinquency_30d"], "decision_date": fr.decision_date.dt.date.astype(str),
    })
    t["risk_band"] = pd.cut(t["risk"], [0, 0.15, 0.30, 0.50, 1.0], labels=["low", "medium", "high", "very_high"]).astype(str)
    t["response_segment"] = customer_segments(a, fr).values
    return t


@router.get("/customers")
def customers(dataset_id: int, page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=200), q: str | None = None,
              sort: str = "risk", desc: bool = True, loan_type: str | None = None, risk_band: str | None = None, customer_segment: str | None = None,
              response_segment: str | None = None, eligible_action: str | None = None, min_arr: float | None = None, uncertainty: str | None = None,
              region_group: str | None = None, min_delinq: int | None = None, split: str | None = None, cohort_from: int | None = None,
              cohort_to: int | None = None, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    a = trained_or_409(ds)
    cfg = default_opt_config(int((a.frame.split == "test").sum()))
    t = portfolio_table(a, cfg)
    if q:
        t = t[t.customer_id.str.contains(q, case=False, regex=False)]
    for col, val in (("loan_type", loan_type), ("risk_band", risk_band), ("customer_segment", customer_segment), ("response_segment", response_segment),
                     ("uncertainty", uncertainty), ("region_group", region_group), ("split", split)):
        if val:
            t = t[t[col].isin(val.split(","))]
    if eligible_action:
        t = t[t.eligible_actions.str.contains(eligible_action)]
    if min_arr is not None:
        t = t[t.best_arr >= min_arr]
    if min_delinq is not None:
        t = t[t.historical_delinquency_count >= min_delinq]
    if cohort_from is not None:
        t = t[t.cohort_month >= cohort_from]
    if cohort_to is not None:
        t = t[t.cohort_month <= cohort_to]
    if sort not in SORTABLE:
        raise HTTPException(422, f"sort must be one of {sorted(SORTABLE)}")
    t = t.sort_values(sort, ascending=not desc, na_position="last")
    total = len(t)
    sl = t.iloc[(page - 1) * page_size: page * page_size]
    return J({"total": total, "page": page, "page_size": page_size, "items": sl.to_dict("records"),
              "labels": {"risk": "predicted (model)", "best_arr": "estimated (causal model)", "best_net": "estimated x assumption"}})


@router.get("/customers/{customer_id}")
def customer(customer_id: str, dataset_id: int, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    fr = frame_for(ds)
    r = fr[fr.customer_id == customer_id]
    if r.empty:
        raise HTTPException(404, "customer not found")
    row = r.iloc[0]
    drop = {c for c in row.index if c.startswith("elig_")}
    return J({k: (None if pd.isna(v) else v) for k, v in row.items() if k not in drop and k not in ("delinquency_60d", "default_event", "missed_payment")} |
             {"note": "Outcome columns shown are post-decision observations and are never model inputs."})


@router.get("/customers/{customer_id}/repayment-history")
def repayment_history(customer_id: str, dataset_id: int, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    if ds.source != "synthetic":
        raise HTTPException(422, "repayment history is available only for standard-schema datasets")
    t = frames.load_tables(ds.name)
    loan = t["loans"][t["loans"].customer_id == customer_id]
    if loan.empty:
        raise HTTPException(404, "customer not found")
    rep = t["repayments"][t["repayments"].loan_id == loan.iloc[0].loan_id].sort_values("scheduled_due_date")
    return J({"loan": loan.iloc[0].to_dict(), "repayments": rep.to_dict("records")})


@router.get("/customers/{customer_id}/recommendations")
def customer_recs(customer_id: str, dataset_id: int, db: Session = Depends(get_db)):
    rows = db.scalars(select(DecisionRecommendation).where(DecisionRecommendation.customer_id == customer_id,
                                                           DecisionRecommendation.dataset_id == dataset_id).order_by(DecisionRecommendation.id.desc())).all()
    return J([{"decision_id": r.decision_id, "action": r.recommended_action, "net_benefit": r.expected_net_benefit, "policy_version": r.policy_version,
               "model_version": r.model_version, "status": r.human_review_status, "timestamp": r.recommendation_timestamp, "reason": r.policy_reason} for r in rows])


@router.get("/customers/{customer_id}/explanation")
def customer_explanation(customer_id: str, dataset_id: int, db: Session = Depends(get_db)):
    from ..explainability.explain import explain_customer
    from ..services.decisions import make_config
    ds = dataset_or_404(db, dataset_id)
    a = trained_or_409(ds)
    try:
        return J(explain_customer(a, customer_id, make_config(int((a.frame.split == "test").sum()), None)))
    except KeyError:
        raise HTTPException(404, "customer not found")
