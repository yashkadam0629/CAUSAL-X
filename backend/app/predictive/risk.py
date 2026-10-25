"""Layer A: predictive credit-risk models (chronological splits, calibration, cost-aware thresholds)."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..core.config import settings
from ..data.schema import CATEGORICAL, DEFAULT_COSTS, DEFAULT_VALUE_PER_EVENT, FEATURES, NUMERIC, PRIMARY_OUTCOME

warnings.filterwarnings("ignore", category=UserWarning)


def make_preprocessor(numeric=None, categorical=None) -> ColumnTransformer:
    """Fit on training data only (callers must never fit on validation/test rows)."""
    numeric = NUMERIC if numeric is None else numeric
    categorical = CATEGORICAL if categorical is None else categorical
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    return ColumnTransformer(transformers)


def make_model(name: str, seed: int, numeric=None, categorical=None) -> Pipeline:
    clf = {
        "logistic_regression": LogisticRegression(max_iter=500, C=1.0),
        "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=30, n_jobs=-1, random_state=seed),
        "gradient_boosting": LGBMClassifier(n_estimators=200, learning_rate=0.04, num_leaves=15, min_child_samples=40,
                                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1),
    }[name]
    return Pipeline([("prep", make_preprocessor(numeric, categorical)), ("clf", clf)])


class Calibrated:
    """Isotonic calibration of a fitted pipeline using held-out validation predictions."""

    def __init__(self, base: Pipeline):
        self.base = base
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)

    def fit(self, X, y):
        self.iso.fit(self.base.predict_proba(X)[:, 1], y)
        return self

    def predict_proba(self, X):
        p = self.iso.predict(self.base.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])


def calibration_table(y, p, bins=10) -> list[dict]:
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "bin": q})
    g = d.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index()
    return g.round(4).to_dict("records")


def expected_cost_threshold(y, p, fn_cost: float, fp_cost: float) -> tuple[float, float]:
    """Threshold minimising expected cost fn_cost*FN + fp_cost*FP on the supplied (validation) data."""
    grid = np.linspace(0.02, 0.9, 89)
    costs = [fn_cost * ((p < t) & (y == 1)).sum() + fp_cost * ((p >= t) & (y == 0)).sum() for t in grid]
    j = int(np.argmin(costs))
    return float(grid[j]), float(costs[j])


def evaluate(y, p, threshold: float, fn_cost: float, fp_cost: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "event_rate": float(y.mean()), "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "expected_cost": float(fn_cost * fn + fp_cost * fp),
        "calibration": calibration_table(y, p),
    }


@dataclass
class RiskBundle:
    model_name: str
    model: object
    features: list
    target: str
    metrics: dict
    comparison: list
    config: dict
    shap_background: pd.DataFrame | None = None
    explainer_model: object = None
    extras: dict = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame[self.features])[:, 1]


def train_risk(frame: pd.DataFrame, target: str = PRIMARY_OUTCOME, seed: int = 42, features=None,
               fn_cost: float = DEFAULT_VALUE_PER_EVENT * 0.2, fp_cost: float = DEFAULT_COSTS["standard_reminder"]) -> RiskBundle:
    """Train on cohorts 0-7, select on 8-9 (Brier), report final metrics on untouched 10-11."""
    features = features or FEATURES
    num = [f for f in features if f in NUMERIC]
    cat = [f for f in features if f in CATEGORICAL]
    tr, va, te = (frame[frame.split == s] for s in ("train", "validation", "test"))
    assert tr["cohort_month"].max() < va["cohort_month"].min() <= va["cohort_month"].max() < te["cohort_month"].min(), "temporal split broken"
    comparison, fitted = [], {}
    for name in ("logistic_regression", "random_forest", "gradient_boosting"):
        t0 = time.perf_counter()
        m = make_model(name, seed, num, cat).fit(tr[features], tr[target])
        fitted[name] = m
        pv = m.predict_proba(va[features])[:, 1]
        thr, _ = expected_cost_threshold(va[target].values, pv, fn_cost, fp_cost)
        comparison.append({"model": name, "calibrated": False, "fit_seconds": round(time.perf_counter() - t0, 2),
                           "validation": evaluate(va[target], pv, thr, fn_cost, fp_cost)})
    # isotonic calibration for the best raw model, fit on validation (calibration set), evaluated on test
    best_raw = min(comparison, key=lambda r: r["validation"]["brier"])["model"]
    # Calibration is fit on the first validation cohort and judged on the second, so the choice
    # between raw and calibrated is not made on the data used to fit the calibrator.
    v1, v2 = va[va.cohort_month == va.cohort_month.min()], va[va.cohort_month > va.cohort_month.min()]
    cal_probe = Calibrated(fitted[best_raw]).fit(v1[features], v1[target].values)
    brier_raw = brier_score_loss(v2[target], fitted[best_raw].predict_proba(v2[features])[:, 1])
    brier_cal = brier_score_loss(v2[target], cal_probe.predict_proba(v2[features])[:, 1])
    cal = Calibrated(fitted[best_raw]).fit(va[features], va[target].values)  # final calibrator uses both validation cohorts
    pv_c = cal.predict_proba(va[features])[:, 1]
    thr_c, _ = expected_cost_threshold(va[target].values, pv_c, fn_cost, fp_cost)
    comparison.append({"model": f"{best_raw}+isotonic", "calibrated": True, "fit_seconds": 0.0,
                       "validation": evaluate(va[target], pv_c, thr_c, fn_cost, fp_cost),
                       "holdout_brier_raw": float(brier_raw), "holdout_brier_calibrated": float(brier_cal),
                       "note": "calibrator fit on validation cohorts: its validation metrics are optimistic; "
                               "raw-vs-calibrated choice used cohort 8 -> cohort 9 split"})
    for r in comparison:
        mdl = cal if r["calibrated"] else fitted[r["model"]]
        r["test"] = evaluate(te[target], mdl.predict_proba(te[features])[:, 1], r["validation"]["threshold"], fn_cost, fp_cost)
    chosen_name = f"{best_raw}+isotonic" if brier_cal < brier_raw else best_raw
    chosen = cal if chosen_name.endswith("isotonic") else fitted[best_raw]
    chosen_row = next(r for r in comparison if r["model"] == chosen_name)
    cfg = {"target": target, "seed": seed, "features": features, "split": {"train_cohorts": "0-7", "validation_cohorts": "8-9", "test_cohorts": "10-11"},
           "selection_rule": "lowest validation Brier score (not accuracy)", "threshold_rule": "minimises expected cost on validation",
           "fn_cost": fn_cost, "fp_cost": fp_cost, "cost_provenance": "analyst assumption"}
    bundle = RiskBundle(chosen_name, chosen, features, target, {"validation": chosen_row["validation"], "test": chosen_row["test"]},
                        comparison, cfg)
    # SHAP on the gradient-boosting model (predictive, NOT causal, explanations)
    gb = fitted["gradient_boosting"]
    bundle.explainer_model = gb
    bundle.shap_background = tr[features].sample(min(500, len(tr)), random_state=seed)
    return bundle


def global_importance(bundle: RiskBundle, frame: pd.DataFrame, n=800, seed=42) -> list[dict]:
    import shap
    gb = bundle.explainer_model
    X = frame[bundle.features].sample(min(n, len(frame)), random_state=seed)
    Xt = gb.named_steps["prep"].transform(X)
    names = list(gb.named_steps["prep"].get_feature_names_out())
    sv = shap.TreeExplainer(gb.named_steps["clf"]).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = np.abs(sv).mean(axis=0)
    return [{"feature": n_.split("__", 1)[-1], "mean_abs_shap": float(v)} for n_, v in sorted(zip(names, imp), key=lambda t: -t[1])[:15]]


def local_explanation(bundle: RiskBundle, row: pd.DataFrame, top=6) -> list[dict]:
    import shap
    gb = bundle.explainer_model
    Xt = gb.named_steps["prep"].transform(row[bundle.features])
    names = list(gb.named_steps["prep"].get_feature_names_out())
    sv = shap.TreeExplainer(gb.named_steps["clf"]).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = np.asarray(sv).reshape(-1)
    order = np.argsort(-np.abs(sv))[:top]
    return [{"feature": names[j].split("__", 1)[-1], "shap_log_odds": float(sv[j])} for j in order]


def save_bundle(bundle: RiskBundle, path):
    joblib.dump(bundle, path)


def load_bundle(path) -> RiskBundle:
    return joblib.load(path)
