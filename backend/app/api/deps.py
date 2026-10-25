from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import settings
from ..data.frames import get_frame
from ..database.models import Dataset, get_db
from ..services.pipeline import Analysis, get_trained


def json_safe(o):
    """Convert numpy / pandas / non-finite values so responses are valid JSON."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    return o


def dataset_or_404(db: Session, dataset_id: int) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(404, f"dataset {dataset_id} not found")
    return ds


def frame_for(ds: Dataset) -> pd.DataFrame:
    if ds.source == "upload":
        p = settings.upload_dir / ds.name / "frame.parquet"
        if not p.exists():
            raise HTTPException(404, "uploaded dataset frame missing")
        return pd.read_parquet(p)
    return get_frame(ds.name)


def trained_or_409(ds: Dataset, seed: int | None = None) -> Analysis:
    if ds.source == "upload":
        raise HTTPException(422, "This analysis requires the standard CAUSAL-X schema (eligibility, costs, pre-treatment features); uploaded single-table datasets support validation, profiling and causal estimation only.")
    a = get_trained(ds.name, seed)
    if a is None:
        raise HTTPException(409, f"No trained models for dataset '{ds.name}'. POST /api/v1/models/train first (runs as a background job).")
    return a
