"""Dataset registry (DB metadata <-> parquet storage), generation, uploads, jobs."""
from __future__ import annotations

import json
import re
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import log, settings
from ..data import frames
from ..data.synthetic import GenConfig, generate, save
from ..data.validation import validate_tables
from ..database.models import (AuditLog, DataQualityReport, Dataset, DatasetVersion, Job, SessionLocal)

_pool = ThreadPoolExecutor(max_workers=2)
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,60}$")


def audit(db: Session, event: str, entity: str, entity_id: str, detail: dict | None = None):
    db.add(AuditLog(event=event, entity=entity, entity_id=str(entity_id), detail=detail or {}))


def register_dir(db: Session, name: str) -> Dataset:
    man = frames.manifest(name)
    ds = db.scalar(select(Dataset).where(Dataset.name == name))
    if ds is None:
        ds = Dataset(name=name, source=man.get("source", "synthetic"), scenario=man.get("config", {}).get("scenario"),
                     n_customers=man["rows"].get("customers", 0), capabilities={})
        db.add(ds); db.flush()
    have = {v.fingerprint for v in ds.versions}
    if man["fingerprint"] not in have:
        db.add(DatasetVersion(dataset_id=ds.id, version=len(have) + 1, fingerprint=man["fingerprint"], config=man.get("config", {}),
                              row_counts=man["rows"], storage_path=str(frames.dataset_dir(name))))
    db.commit()
    return ds


def sync_datasets(db: Session):
    for p in settings.synthetic_dir.glob("*/manifest.json"):
        try:
            register_dir(db, p.parent.name)
        except Exception as e:
            log.warning("could not register %s: %s", p.parent.name, e)
    for p in settings.upload_dir.glob("*/manifest.json"):
        try:
            man = json.loads(p.read_text())
            if not db.scalar(select(Dataset).where(Dataset.name == man["name"])):
                ds = Dataset(name=man["name"], source="upload", scenario=None, n_customers=man["rows"]["customers"], capabilities=man.get("capabilities", {}))
                db.add(ds); db.flush()
                db.add(DatasetVersion(dataset_id=ds.id, version=1, fingerprint=man["fingerprint"], config=man.get("config", {}), row_counts=man["rows"], storage_path=str(p.parent)))
                db.commit()
        except Exception as e:
            log.warning("could not register upload %s: %s", p.parent.name, e)


def generate_dataset(name: str, n_customers: int, scenario: str, seed: int, **kw) -> dict:
    if not NAME_RE.match(name):
        raise ValueError("name must be 1-60 chars of letters, digits, '_' or '-'")
    cfg = GenConfig(n_customers=n_customers, seed=seed, scenario=scenario, **kw)
    tables = generate(cfg)
    man = save(tables, cfg, settings.synthetic_dir, name)
    frames.clear_cache()
    with SessionLocal() as db:
        ds = register_dir(db, name)
        q = validate_tables(tables)
        db.add(DataQualityReport(dataset_id=ds.id, score=q["quality_score"], report=q))
        audit(db, "dataset_generated", "dataset", ds.id, {"name": name, "fingerprint": man["fingerprint"], "seed": seed, "scenario": scenario})
        db.commit()
    return man


def delete_dataset(db: Session, ds: Dataset):
    base = settings.upload_dir if ds.source == "upload" else settings.synthetic_dir
    path = (base / ds.name).resolve()
    if base.resolve() in path.parents and path.exists():
        shutil.rmtree(path)
    mp = settings.artifacts_dir / "models"
    for p in mp.glob(f"{ds.name}__seed*"):
        shutil.rmtree(p, ignore_errors=True)
    audit(db, "dataset_deleted", "dataset", ds.id, {"name": ds.name})
    db.delete(ds); db.commit()
    frames.clear_cache()


# ---------------------------------------------------------------- jobs
def submit_job(kind: str, fn, *args, **kwargs) -> str:
    job_id = "job-" + uuid.uuid4().hex[:10]
    with SessionLocal() as db:
        db.add(Job(job_id=job_id, kind=kind, status="queued")); db.commit()

    def update(**vals):
        with SessionLocal() as db:
            j = db.scalar(select(Job).where(Job.job_id == job_id))
            for k, v in vals.items():
                setattr(j, k, v)
            db.commit()

    def runner():
        update(status="running", progress=0.01, message="started")
        try:
            res = fn(*args, progress=lambda f, m: update(progress=float(f), message=m), **kwargs)
            update(status="succeeded", progress=1.0, message="done", result=res if isinstance(res, dict) else None, finished_at=datetime.now(timezone.utc))
        except Exception as e:
            log.error("job %s failed: %s\n%s", job_id, e, traceback.format_exc())
            update(status="failed", message=f"{type(e).__name__}: {e}", finished_at=datetime.now(timezone.utc))

    _pool.submit(runner)
    return job_id
