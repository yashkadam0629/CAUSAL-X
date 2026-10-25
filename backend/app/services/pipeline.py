"""Analysis pipeline: fits the risk model, cross-fitted CATE models, uplift evaluation, causal summary and monitoring for a dataset."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from ..causal.estimators import CausalSpec, estimate
from ..core.config import log, settings, timed
from ..data.frames import get_frame, get_truth, manifest
from ..data.schema import (ARMS, DEFAULT_COSTS, DEFAULT_VALUE_PER_EVENT, FEATURES, PRIMARY_OUTCOME, TREATMENTS)
from ..monitoring.drift import drift_report, effect_stability
from ..optimization.engine import OptConfig, Problem
from ..policy_evaluation.ope import LoggingNuisance
from ..predictive.risk import RiskBundle, global_importance, train_risk
from ..uplift.hte import CateModel, evaluate_uplift, fit_cate, score_all, segment

_LOCK = threading.Lock()
_CACHE: dict[str, "Analysis"] = {}
MODEL_VERSION_PREFIX = "causalx"


@dataclass
class Analysis:
    dataset: str
    seed: int
    model_version: str
    fingerprint: str
    scenario: str
    has_truth: bool
    frame: pd.DataFrame
    risk_bundle: RiskBundle
    risk: np.ndarray
    cate: dict
    scores: pd.DataFrame
    uplift_eval: dict
    causal_summary: dict
    importance: list
    nuisance: LoggingNuisance
    drift: dict
    effect_stability: list
    timings: dict = field(default_factory=dict)


def _version(dataset: str, fingerprint: str, seed: int) -> str:
    return f"{MODEL_VERSION_PREFIX}-{hashlib.sha256(f'{dataset}{fingerprint}{seed}{FEATURES}'.encode()).hexdigest()[:8]}"


def _pair_propensity(frame: pd.DataFrame, arm: str) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from ..data.schema import CATEGORICAL, NUMERIC
    from ..predictive.risk import make_preprocessor
    s = frame[(frame.split == "train") & frame.arm.isin(["none", arm]) & frame[f"elig_{arm}"]]
    prep = make_preprocessor().fit(s[NUMERIC + CATEGORICAL])
    lr = LogisticRegression(max_iter=500).fit(prep.transform(s[NUMERIC + CATEGORICAL]), (s.arm == arm).astype(int))
    return lr.predict_proba(prep.transform(frame[NUMERIC + CATEGORICAL]))[:, 1]


def run_analysis(dataset: str, seed: int | None = None, progress=lambda f, m: None, force=False) -> Analysis:
    seed = settings.default_seed if seed is None else seed
    key = f"{dataset}:{seed}"
    with _LOCK:
        if key in _CACHE and not force:
            return _CACHE[key]
    cache_file = settings.artifacts_dir / "models" / f"{dataset}__seed{seed}" / "analysis.joblib"
    if cache_file.exists() and not force:
        try:
            a = joblib.load(cache_file)
            a.frame = get_frame(dataset)
            _CACHE[key] = a
            return a
        except Exception as e:  # stale / incompatible artifact: recompute
            log.warning("could not load cached analysis (%s); recomputing", e)
    timings = {}
    man = manifest(dataset)
    frame = get_frame(dataset).copy()
    truth = get_truth(dataset) if man.get("source") == "synthetic" else None
    progress(0.05, "Training predictive risk models")
    with timed("risk models"):
        bundle = train_risk(frame, PRIMARY_OUTCOME, seed)
    risk = bundle.predict(frame)
    importance = global_importance(bundle, frame, seed=seed)
    progress(0.25, "Fitting heterogeneous-effect (DR-learner) models")
    cate: dict[str, CateModel] = {}
    for k, arm in enumerate(TREATMENTS):
        with timed(f"cate {arm}"):
            cate[arm] = fit_cate(frame, arm, PRIMARY_OUTCOME, seed=seed)
        progress(0.25 + 0.15 * (k + 1), f"CATE model {arm} fitted")
    scores = score_all(frame, cate)
    progress(0.7, "Evaluating uplift on held-out cohorts")
    ev = {}
    for arm in TREATMENTS:
        e_hat = _pair_propensity(frame, arm)
        ev[arm] = evaluate_uplift(frame, scores, arm, risk, truth, e_hat, seed=seed)
    progress(0.8, "Estimating average effects with all estimators")
    summary = {}
    for arm in TREATMENTS:
        summary[arm] = {}
        for est in ("difference_in_means", "regression_adjustment", "ipw", "aipw"):
            r = estimate(frame, CausalSpec(treatment=arm, estimator=est, bootstrap=30, seed=seed))
            slim = {k: r.get(k) for k in ("status", "ate", "se", "ci_low", "ci_high", "att", "n", "se_method", "reason")}
            slim["warnings"] = r.get("warnings", [])[:3]
            summary[arm][est] = slim
        if truth is not None:
            from ..causal.estimators import prepare
            d, _, _ = prepare(frame, CausalSpec(treatment=arm))
            summary[arm]["synthetic_truth_ate"] = float(truth.set_index("customer_id").loc[d.customer_id, f"itt_effect_{arm}"].mean())
    progress(0.92, "Monitoring and effect-stability checks")
    nuis = LoggingNuisance(frame, PRIMARY_OUTCOME, seed)
    drift = drift_report(frame, risk)
    stab = effect_stability(frame, "standard_reminder", lambda ch: {k: v for k, v in estimate(
        frame, CausalSpec(treatment="standard_reminder", estimator="aipw", cohorts=ch, seed=seed, outcome_learner="logistic", cross_fit_folds=3)).items()
        if k in ("status", "ate", "ci_low", "ci_high", "n")})
    a = Analysis(dataset, seed, _version(dataset, man["fingerprint"], seed), man["fingerprint"], man["config"].get("scenario", "uploaded"),
                 truth is not None, frame, bundle, risk, cate, scores, ev, summary, importance, nuis, drift, stab, timings)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    stash = a.frame
    a.frame = None  # frame is rebuilt from parquet on load; avoids duplicating data in the artifact
    joblib.dump(a, cache_file)
    a.frame = stash
    with _LOCK:
        _CACHE[key] = a
    progress(1.0, "Analysis complete")
    return a


def default_opt_config(n_decision: int, **kw) -> OptConfig:
    """Defaults scale with the decision-cohort size; every value is an editable analyst assumption."""
    base = dict(budget=float(round(18 * n_decision, -2)), agent_capacity=float(max(10, round(0.012 * n_decision))),
                channel_capacity=int(n_decision), value_per_event=DEFAULT_VALUE_PER_EVENT,
                costs={a: DEFAULT_COSTS[a] for a in TREATMENTS})
    base.update(kw)
    return OptConfig(**base)


def decision_mask(a: Analysis, cohorts=None) -> np.ndarray:
    if cohorts is None:
        return (a.frame.split == "test").values
    return a.frame.cohort_month.isin(cohorts).values


def build_problem(a: Analysis, cohorts=None, use_truth=True) -> tuple[Problem, pd.DataFrame]:
    m = decision_mask(a, cohorts)
    d = a.frame[m]
    sc = a.scores[m]
    arr = np.column_stack([sc[f"arr_{x}"].values for x in TREATMENTS])
    se = np.column_stack([sc[f"se_{x}"].values for x in TREATMENTS])
    elig = np.column_stack([d[f"elig_{x}"].values for x in TREATMENTS])
    true_arr = None
    if a.has_truth and use_truth:
        tr = get_truth(a.dataset).set_index("customer_id").loc[d.customer_id]
        true_arr = np.column_stack([-tr[f"itt_effect_{x}"].values for x in TREATMENTS])
    p = Problem(d.customer_id.values, a.risk[m], arr, se, elig, d["region_group"].values, true_arr, list(TREATMENTS))
    return p, d


def naive_arr_matrix(a: Analysis, d: pd.DataFrame) -> np.ndarray:
    """Simpler (non-causal) baseline: raw treated-vs-untreated outcome gap within baseline-risk deciles, from training cohorts."""
    tr = a.frame[a.frame.split == "train"].copy()
    tr["risk"] = a.risk[tr.index]
    edges = np.unique(np.quantile(tr["risk"], np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    tr["dec"] = np.digitize(tr["risk"], edges[1:-1])
    dd = d.copy(); dd["risk"] = a.risk[d.index]; dd["dec"] = np.digitize(dd["risk"], edges[1:-1])
    out = np.zeros((len(d), len(TREATMENTS)))
    for j, arm in enumerate(TREATMENTS):
        for k in range(10):
            g = tr[tr.dec == k]
            t, c = g[g.arm == arm][PRIMARY_OUTCOME], g[g.arm == "none"][PRIMARY_OUTCOME]
            gap = (c.mean() - t.mean()) if len(t) >= 20 and len(c) >= 20 else 0.0
            out[(dd.dec == k).values, j] = gap
    return out


def customer_segments(a: Analysis, d: pd.DataFrame, high=0.08, moderate=0.03, z=1.28) -> pd.Series:
    sc = a.scores.loc[d.index]
    arr = np.column_stack([sc[f"arr_{x}"].values for x in TREATMENTS])
    se = np.column_stack([sc[f"se_{x}"].values for x in TREATMENTS])
    elig = np.column_stack([d[f"elig_{x}"].values for x in TREATMENTS])
    arr_m = np.where(elig, arr, -np.inf)
    j = arr_m.argmax(axis=1)
    r = np.arange(len(d))
    best_arr = np.where(elig.any(axis=1), arr[r, j], 0.0)
    return pd.Series(segment(best_arr, se[r, j], elig.any(axis=1), high, moderate, z), index=d.index)


def get_trained(dataset: str, seed: int | None = None) -> "Analysis | None":
    """Return a previously trained analysis (memory or disk) without triggering computation."""
    if seed is None:  # most recently trained analysis for this dataset (any seed)
        found = sorted((settings.artifacts_dir / "models").glob(f"{dataset}__seed*/analysis.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
        seed = int(found[0].parent.name.rsplit("seed", 1)[1]) if found else settings.default_seed
    key = f"{dataset}:{seed}"
    if key in _CACHE:
        return _CACHE[key]
    cache_file = settings.artifacts_dir / "models" / f"{dataset}__seed{seed}" / "analysis.joblib"
    if cache_file.exists():
        try:
            a = joblib.load(cache_file)
            a.frame = get_frame(dataset)
            _CACHE[key] = a
            return a
        except Exception as e:
            log.warning("cached analysis unreadable: %s", e)
    return None
