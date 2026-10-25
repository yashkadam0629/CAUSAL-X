"""CAUSAL-X FastAPI application."""
from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .api import routes_analysis, routes_data, routes_decision
from .api.deps import dataset_or_404, json_safe as J, trained_or_409
from .core.config import log, settings
from .database.models import SessionLocal, get_db, init_db
from .reporting import report as report_mod
from .services import decisions as dec
from .services.pipeline import build_problem
from .services.registry import sync_datasets


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with SessionLocal() as db:
        sync_datasets(db)
    log.info("CAUSAL-X ready (db=%s)", "sqlite" if settings.is_sqlite else "postgresql")
    yield


app = FastAPI(title="CAUSAL-X API", version=settings.version, lifespan=lifespan,
              description="Financial decision intelligence: prediction, causal effects, uplift, constrained optimisation. "
                          "Single-local-user security model (no authentication).")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error. See server logs for details.", "type": type(exc).__name__})


@app.exception_handler(ValueError)
async def value_error(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def not_found(request: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=404, content={"detail": f"not found: {exc}"})


API = "/api/v1"
app.include_router(routes_data.router, prefix=API)
app.include_router(routes_analysis.router, prefix=API)
app.include_router(routes_decision.router, prefix=API)


# ---------------------------------------------------------------- reports
@app.post(f"{API}/reports/generate", tags=["reports"])
def generate_report(dataset_id: int, run_id: str | None = None, db: Session = Depends(get_db)):
    ds = dataset_or_404(db, dataset_id)
    a = trained_or_409(ds)
    if run_id:
        r = routes_decision._run_or_404(db, run_id)
        run = {"config": r.config, "summary": r.summary, "strategies": r.comparison, "policy_version": r.policy_version,
               "decision_cohorts": r.manifest.get("decision_cohorts", [])}
    else:
        run, *_ = dec.run_optimization(a, None, None, True, 42, 30)
    p, d = build_problem(a)
    cfg = dec.make_config(len(d), run["config"])
    out = report_mod.generate(a, run, p, cfg, mc_draws=100)
    from .services.registry import audit
    audit(db, "report_generated", "report", out["markdown"], out); db.commit()
    return J(out)


@app.get(f"{API}/reports", tags=["reports"])
def list_reports():
    d = settings.artifacts_dir / "reports"
    return J([{"name": p.name, "size": p.stat().st_size, "modified": p.stat().st_mtime} for p in sorted(d.glob("causalx_report_*"), key=lambda p: -p.stat().st_mtime)])


@app.get(f"{API}/reports/{{name}}", tags=["reports"])
def download_report(name: str):
    if not re.fullmatch(r"causalx_report_[\w\-]+\.(md|html|pdf)", name):
        raise HTTPException(400, "invalid report name")
    p = (settings.artifacts_dir / "reports" / name).resolve()
    if (settings.artifacts_dir / "reports").resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "report not found")
    return FileResponse(p, filename=name)
