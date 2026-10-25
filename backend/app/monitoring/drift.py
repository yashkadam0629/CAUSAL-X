"""Layer G: monitoring. Feature / prediction / outcome / treatment drift, calibration drift, effect stability."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..data.schema import CATEGORICAL, NUMERIC, TREATMENTS


def psi(ref: pd.Series, cur: pd.Series, bins=10) -> float:
    """Population Stability Index with reference-quantile bins (numeric) or category shares."""
    ref, cur = ref.dropna(), cur.dropna()
    if len(ref) < 50 or len(cur) < 50:
        return float("nan")
    if ref.dtype == object or str(ref.dtype) == "category":
        cats = sorted(set(ref) | set(cur))
        p = np.array([(ref == c).mean() for c in cats]) + 1e-4
        q = np.array([(cur == c).mean() for c in cats]) + 1e-4
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref, edges)[0] / len(ref) + 1e-4
        q = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((q - p) * np.log(q / p)))


def level(v: float) -> str:
    if np.isnan(v):
        return "insufficient_data"
    return "significant" if v > 0.25 else ("moderate" if v > 0.10 else "stable")


def drift_report(frame: pd.DataFrame, risk: np.ndarray, ref_cohorts=range(0, 8), cur_cohorts=range(10, 12), outcome="delinquency_30d") -> dict:
    ref = frame[frame.cohort_month.isin(list(ref_cohorts))]
    cur = frame[frame.cohort_month.isin(list(cur_cohorts))]
    r_ref, r_cur = risk[ref.index], risk[cur.index]
    feats = []
    for f in NUMERIC + CATEGORICAL:
        v = psi(ref[f], cur[f])
        ks = stats.ks_2samp(ref[f].dropna(), cur[f].dropna()).pvalue if f in NUMERIC and len(cur) > 50 else None
        feats.append({"feature": f, "psi": v, "status": level(v), "ks_pvalue": float(ks) if ks is not None else None,
                      "missing_ref": float(ref[f].isna().mean()), "missing_cur": float(cur[f].isna().mean())})
    feats.sort(key=lambda r: -(r["psi"] if not np.isnan(r["psi"]) else -1))
    pred_psi = psi(pd.Series(r_ref), pd.Series(r_cur))
    monthly = []
    from sklearn.metrics import roc_auc_score
    for m, g in frame.groupby("cohort_month"):
        rr = risk[g.index]
        y = g[outcome].values
        monthly.append({
            "cohort_month": int(m), "n": int(len(g)), "outcome_rate": float(y.mean()), "mean_predicted_risk": float(rr.mean()),
            "calibration_gap": float(rr.mean() - y.mean()), "auc": float(roc_auc_score(y, rr)) if 0 < y.mean() < 1 else None,
            "treated_share": float((g.arm != "none").mean()), "delivery_rate": float((g.delivery_status == "delivered").sum() / max((g.arm != "none").sum(), 1)),
            "mean_cost_per_treated": float(g.loc[g.arm != "none", "intervention_cost"].mean()) if (g.arm != "none").any() else None,
            "economic_stress": float(g["economic_stress_index"].mean()),
            **{f"share_{a}": float((g.arm == a).mean()) for a in ["none"] + TREATMENTS},
        })
    warnings = []
    sig = [f["feature"] for f in feats if f["status"] == "significant"]
    mod = [f["feature"] for f in feats if f["status"] == "moderate"]
    if sig:
        warnings.append(f"Significant feature drift (PSI>0.25): {', '.join(sig)}. Investigate data pipeline and population change before trusting scores.")
    if mod:
        warnings.append(f"Moderate feature drift (PSI 0.10-0.25): {', '.join(mod)}.")
    if level(pred_psi) != "stable":
        warnings.append(f"Prediction drift: PSI of risk score = {pred_psi:.3f}.")
    cal_gap = np.mean([m["calibration_gap"] for m in monthly[-2:]])
    if abs(cal_gap) > 0.03:
        warnings.append(f"Calibration drift: recent mean predicted risk differs from observed by {cal_gap:+.3f}.")
    return {
        "reference_cohorts": [min(ref_cohorts), max(ref_cohorts)], "current_cohorts": [min(cur_cohorts), max(cur_cohorts)],
        "feature_drift": feats, "prediction_psi": pred_psi, "prediction_status": level(pred_psi), "monthly": monthly, "warnings": warnings,
        "notes": ["PSI thresholds (0.10 / 0.25) are conventions, not tests; combine with KS p-values and domain review.",
                  "Feature or prediction drift does not by itself show that intervention effectiveness changed (needs treatment & outcome data: see effect stability).",
                  "Concept drift (relationship between features and outcome) appears as calibration/AUC changes by cohort."],
    }


def effect_stability(frame: pd.DataFrame, arm: str, estimate_fn) -> list[dict]:
    """ATE of `arm` vs none in consecutive periods using the supplied estimator function (period -> result)."""
    out = []
    for label, cohorts in (("cohorts 0-3", [0, 1, 2, 3]), ("cohorts 4-7", [4, 5, 6, 7]), ("cohorts 8-11", [8, 9, 10, 11])):
        r = estimate_fn(cohorts)
        out.append({"period": label, **{k: r.get(k) for k in ("status", "ate", "ci_low", "ci_high", "n")}})
    return out
