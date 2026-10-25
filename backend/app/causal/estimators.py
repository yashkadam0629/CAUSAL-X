"""Layer B: causal effect estimators for a single treatment-vs-comparison contrast.

Notation: T=1 treatment arm, T=0 comparison arm; Y the outcome (for adverse outcomes lower is better).
Sign convention (used everywhere in CAUSAL-X):  tau = E[Y(1) - Y(0)]   (negative = treatment reduces the adverse outcome)
                                                  ARR = -tau            (absolute risk reduction; positive = beneficial)
The estimand is an intention-to-treat (ITT) effect of *assignment*, because delivery/receipt is imperfect.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

from ..data.schema import CATEGORICAL, NUMERIC
from ..predictive.risk import make_preprocessor

warnings.filterwarnings("ignore")
ESTIMATORS = ["difference_in_means", "regression_adjustment", "ipw", "aipw"]
ASSUMPTIONS = {
    "consistency_sutva": "A customer's outcome under the assigned action is the observed outcome; no interference between customers.",
    "conditional_exchangeability": "Given the chosen pre-treatment covariates, assignment is independent of the potential outcomes (untestable).",
    "positivity": "Every covariate profile in the population has a non-zero chance of receiving each compared arm (checkable via overlap).",
    "time_zero": "Covariates are measured at/before the decision date; the outcome window starts after it.",
    "correct_model": "Outcome and/or propensity models are adequately specified (doubly robust: at least one).",
}


@dataclass
class CausalSpec:
    treatment: str = "standard_reminder"
    comparison: str = "none"
    outcome: str = "delinquency_30d"
    covariates: list = field(default_factory=list)
    cohorts: list | None = None  # None = all cohorts
    estimator: str = "aipw"
    bootstrap: int = 60
    seed: int = 42
    trim: tuple | None = None  # explicit overlap restriction (lo, hi); None = no trimming
    ps_model: str = "logistic"
    outcome_learner: str = "gbm"
    clip_ps: float = 0.01
    restrict_eligible: bool = True
    cross_fit_folds: int = 5


def _kind(y: np.ndarray) -> str:
    return "binary" if set(np.unique(y)) <= {0, 1} else "continuous"


class Predictor:
    """Picklable callable wrapper (lambdas cannot be serialised with joblib)."""

    def __init__(self, model=None, proba=True, const=None):
        self.model, self.proba, self.const = model, proba, const

    def __call__(self, Z):
        if self.const is not None:
            return np.full(len(Z), self.const)
        return self.model.predict_proba(Z)[:, 1] if self.proba else self.model.predict(Z)


def fit_outcome_model(Z, y, kind, learner, seed):
    if learner == "gbm":
        kw = dict(n_estimators=120, learning_rate=0.05, num_leaves=8, min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=2)
        m = (LGBMClassifier if kind == "binary" else LGBMRegressor)(**kw)
    else:
        m = LogisticRegression(C=1.0, max_iter=400) if kind == "binary" else Ridge(alpha=1.0)
    if kind == "binary" and len(np.unique(y)) < 2:  # degenerate fold
        return Predictor(const=float(np.mean(y)))
    m.fit(Z, y)
    return Predictor(m, proba=(kind == "binary"))


def fit_propensity(Z, t, model, seed):
    if model == "gbm":
        m = LGBMClassifier(n_estimators=120, learning_rate=0.05, num_leaves=8, min_child_samples=50, subsample=0.8,
                           subsample_freq=1, random_state=seed, verbose=-1, n_jobs=2)
    else:
        m = LogisticRegression(C=1.0, max_iter=400)
    m.fit(Z, t)
    return Predictor(m)


def crossfit_nuisance(Z, t, y, kind, learner, ps_model, folds, seed, known_ps=None):
    n = len(y)
    mu0, mu1, e = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(Z):
        for arm, out in ((0, mu0), (1, mu1)):
            idx = tr[t[tr] == arm]
            out[te] = fit_outcome_model(Z[idx], y[idx], kind, learner, seed)(Z[te])
        e[te] = fit_propensity(Z[tr], t[tr], ps_model, seed)(Z[te])
    if known_ps is not None:
        e = known_ps
    return mu0, mu1, e


def ess(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def weights_for(t, e, estimand="ATE"):
    if estimand == "ATE":
        return np.where(t == 1, 1 / e, 1 / (1 - e))
    return np.where(t == 1, 1.0, e / (1 - e))  # ATT


def is_cat(d: pd.DataFrame, c: str) -> bool:
    return not pd.api.types.is_numeric_dtype(d[c])


def default_covariates(d: pd.DataFrame) -> list:
    std = [c for c in NUMERIC + CATEGORICAL if c in d.columns]
    return std if std else [c for c in d.columns if c not in ("customer_id", "arm", "outcome") and not c.startswith(("elig_", "y_"))]


def prepare(frame: pd.DataFrame, spec: CausalSpec):
    """Select the analysis population explicitly (documented in the result)."""
    d = frame[frame.arm.isin([spec.treatment, spec.comparison])]
    n_arms = len(d)
    if spec.restrict_eligible and f"elig_{spec.treatment}" in d and f"elig_{spec.comparison}" in d:
        d = d[d[f"elig_{spec.treatment}"] & d[f"elig_{spec.comparison}"]]
    n_elig = len(d)
    if spec.cohorts is not None and "cohort_month" in d:
        d = d[d.cohort_month.isin(spec.cohorts)]
    miss_y = int(d[spec.outcome].isna().sum())
    d = d.dropna(subset=[spec.outcome])
    cov = spec.covariates or default_covariates(d)
    pop = {"rows_in_two_arms": int(n_arms), "after_eligibility": int(n_elig), "after_cohort_filter": int(len(d) + miss_y),
           "dropped_missing_outcome": miss_y, "final": int(len(d)),
           "rule": f"customers assigned '{spec.treatment}' or '{spec.comparison}' and eligible for both" if spec.restrict_eligible
           else "all customers in either arm"}
    return d.reset_index(drop=True), cov, pop


def _design(d: pd.DataFrame, cov: list):
    num = [c for c in cov if not is_cat(d, c)]
    cat = [c for c in cov if is_cat(d, c)]
    prep = make_preprocessor(num, cat)
    Z = prep.fit_transform(d[num + cat])
    return np.asarray(Z, float), prep


def _wmean(y, w):
    return float(np.sum(w * y) / np.sum(w))


def smd_table(d: pd.DataFrame, cov: list, t: np.ndarray, w: np.ndarray | None = None) -> pd.DataFrame:
    """Standardised mean differences on raw covariates (one-hot for categoricals), unweighted and weighted."""
    X = pd.get_dummies(d[cov], columns=[c for c in cov if is_cat(d, c)], dtype=float)
    X = X.fillna(X.median())
    rows = []
    for c in X.columns:
        x = X[c].values
        def smd(wt):
            m1, m0 = _wmean(x[t == 1], wt[t == 1]), _wmean(x[t == 0], wt[t == 0])
            v1 = _wmean((x[t == 1] - m1) ** 2, wt[t == 1]); v0 = _wmean((x[t == 0] - m0) ** 2, wt[t == 0])
            sd = np.sqrt((v1 + v0) / 2)
            return float((m1 - m0) / sd) if sd > 1e-12 else 0.0
        rows.append({"covariate": c, "smd_unadjusted": smd(np.ones(len(x))),
                     "smd_weighted": smd(w) if w is not None else None})
    return pd.DataFrame(rows)


def diagnostics(d, cov, t, e, w, y) -> dict:
    n1, n0 = int((t == 1).sum()), int((t == 0).sum())
    smd = smd_table(d, cov, t, w)
    bins = np.linspace(0, 1, 21)
    h1, _ = np.histogram(e[t == 1], bins); h0, _ = np.histogram(e[t == 0], bins)
    miss = (d[cov].isna().groupby(t).mean().T.rename(columns={0: "comparison", 1: "treatment"}) if len(cov) else pd.DataFrame())
    outside = float(((e < 0.05) | (e > 0.95)).mean())
    warns = []
    if min(n1, n0) < 200:
        warns.append(f"Small arm (n_treated={n1}, n_comparison={n0}); estimates will be imprecise.")
    if outside > 0.10:
        warns.append(f"Poor overlap: {outside:.0%} of units have estimated propensity outside [0.05, 0.95].")
    if smd["smd_weighted"].abs().max() > 0.10:
        warns.append(f"Residual imbalance after weighting: max |SMD| = {smd['smd_weighted'].abs().max():.2f} (>0.10 convention).")
    ess_t, ess_c = ess(w[t == 1]), ess(w[t == 0])
    if min(ess_t / max(n1, 1), ess_c / max(n0, 1)) < 0.5:
        warns.append("Effective sample size is below 50% of the nominal size: weights are highly variable.")
    if w.max() / w.sum() > 0.01 and w.max() > 20:
        warns.append(f"Extreme weight detected (max={w.max():.1f}).")
    return {
        "n_treated": n1, "n_comparison": n0, "propensity_min": float(e.min()), "propensity_max": float(e.max()),
        "share_outside_0.05_0.95": outside,
        "overlap_histogram": {"bins": [round(b, 2) for b in bins[:-1]], "treated": h1.tolist(), "comparison": h0.tolist()},
        "ess_treated": ess_t, "ess_comparison": ess_c, "max_weight": float(w.max()),
        "weight_p99": float(np.percentile(w, 99)),
        "smd": smd.round(4).to_dict("records"), "max_abs_smd_unadjusted": float(smd["smd_unadjusted"].abs().max()),
        "max_abs_smd_weighted": float(smd["smd_weighted"].abs().max()),
        "missingness_by_group": miss.round(4).reset_index().rename(columns={"index": "covariate"}).to_dict("records") if len(miss) else [],
        "raw_outcome_means": {"treatment": float(y[t == 1].mean()), "comparison": float(y[t == 0].mean())},
        "warnings": warns,
    }


def _result(name, ate, se, att=None, att_se=None, extra=None):
    z = stats.norm.ppf(0.975)
    out = {"estimator": name, "ate": float(ate), "se": float(se), "ci_low": float(ate - z * se), "ci_high": float(ate + z * se),
           "p_value": float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None}
    if att is not None:
        out.update(att=float(att), att_se=float(att_se) if att_se is not None else None,
                   att_ci_low=float(att - z * att_se) if att_se else None, att_ci_high=float(att + z * att_se) if att_se else None)
    if extra:
        out.update(extra)
    return out


def estimate(frame: pd.DataFrame, spec: CausalSpec, known_ps_col: str | None = "assignment_probability") -> dict:
    """Run one estimator under the explicit specification and return estimate + diagnostics + metadata."""
    d, cov, pop = prepare(frame, spec)
    t = (d.arm == spec.treatment).astype(int).values
    y = d[spec.outcome].astype(float).values
    kind = _kind(y)
    if min(t.sum(), (1 - t).sum()) < 30:
        return {"status": "insufficient_evidence", "reason": "fewer than 30 units in an arm", "population": pop, "spec": _spec_dict(spec)}
    Z, _ = _design(d, cov)
    rng = np.random.default_rng(spec.seed)
    n = len(d)

    # known (logged) propensities: the design is randomised among eligible arms if all rows carry them
    logged = d[known_ps_col].notna().all() if known_ps_col in d else False
    ps_hat = fit_propensity(Z, t, spec.ps_model, spec.seed)(Z)
    if logged:  # P(assigned arm) recorded; convert to P(T=1 | T in {t,c})
        p_assigned = d[known_ps_col].values
        ps_known = None  # arm-specific marginal propensities need both arms' probabilities -> estimated within the pair instead
    e = np.clip(ps_hat, spec.clip_ps, 1 - spec.clip_ps)
    n_clipped = int((e != ps_hat).sum())
    keep = np.ones(n, bool)
    if spec.trim:
        keep = (ps_hat >= spec.trim[0]) & (ps_hat <= spec.trim[1])
        if keep.sum() < 60:
            return {"status": "insufficient_evidence", "reason": "trimming leaves too few units", "population": pop, "spec": _spec_dict(spec)}
        d, Z, t, y, e, ps_hat = d[keep].reset_index(drop=True), Z[keep], t[keep], y[keep], e[keep], ps_hat[keep]
        n = len(d)
    pop["trimmed_out"] = int((~keep).sum())
    pop["ps_clipped"] = n_clipped

    w_ate = weights_for(t, e, "ATE")
    diag = diagnostics(d, cov, t, e, w_ate, y)
    name = spec.estimator
    if name == "difference_in_means":
        m1, m0 = y[t == 1].mean(), y[t == 0].mean()
        se = np.sqrt(y[t == 1].var(ddof=1) / (t == 1).sum() + y[t == 0].var(ddof=1) / (t == 0).sum())
        res = _result(name, m1 - m0, se, att=m1 - m0, att_se=se)
    elif name == "regression_adjustment":
        def fit_pred(idx):
            f1 = fit_outcome_model(Z[idx][t[idx] == 1], y[idx][t[idx] == 1], kind, "logistic", spec.seed)
            f0 = fit_outcome_model(Z[idx][t[idx] == 0], y[idx][t[idx] == 0], kind, "logistic", spec.seed)
            m1, m0 = f1(Z[idx]), f0(Z[idx])
            tt = t[idx] == 1
            return (m1 - m0).mean(), (m1 - m0)[tt].mean()
        ate, att = fit_pred(np.arange(n))
        boots = np.array([fit_pred(rng.integers(0, n, n)) for _ in range(spec.bootstrap)])
        res = _result(name, ate, boots[:, 0].std(ddof=1), att, boots[:, 1].std(ddof=1), {"se_method": f"bootstrap (B={spec.bootstrap})"})
    elif name == "ipw":
        def ipw(idx):
            ee = fit_propensity(Z[idx], t[idx], spec.ps_model, spec.seed)(Z[idx])
            ee = np.clip(ee, spec.clip_ps, 1 - spec.clip_ps)
            tt, yy = t[idx], y[idx]
            wa = weights_for(tt, ee, "ATE"); wt = weights_for(tt, ee, "ATT")
            ate = _wmean(yy[tt == 1], wa[tt == 1]) - _wmean(yy[tt == 0], wa[tt == 0])
            att = yy[tt == 1].mean() - _wmean(yy[tt == 0], wt[tt == 0])
            return ate, att
        ate, att = ipw(np.arange(n))
        boots = np.array([ipw(rng.integers(0, n, n)) for _ in range(spec.bootstrap)])
        res = _result(name, ate, boots[:, 0].std(ddof=1), att, boots[:, 1].std(ddof=1),
                      {"se_method": f"bootstrap with propensity refit (B={spec.bootstrap})"})
    elif name == "aipw":
        mu0, mu1, ecf = crossfit_nuisance(Z, t, y, kind, spec.outcome_learner, spec.ps_model, spec.cross_fit_folds, spec.seed)
        ecf = np.clip(ecf, spec.clip_ps, 1 - spec.clip_ps)
        psi = mu1 - mu0 + t * (y - mu1) / ecf - (1 - t) * (y - mu0) / (1 - ecf)
        res = _result(name, psi.mean(), psi.std(ddof=1) / np.sqrt(n),
                      extra={"se_method": f"influence function, {spec.cross_fit_folds}-fold cross-fitting",
                             "att_note": "ATT not reported for AIPW in this build"})
        diag["max_weight_cross_fit"] = float(np.max(np.where(t == 1, 1 / ecf, 1 / (1 - ecf))))
    else:
        raise ValueError(f"unknown estimator {name}")

    base = float(y[t == 0].mean())
    res.update({
        "status": "ok", "spec": _spec_dict(spec), "population": pop, "diagnostics": diag, "n": int(n),
        "estimand": "ITT effect of assignment (ATE%s) on %s; tau=E[Y(treat)-Y(comparison)], negative=beneficial" % (
            " & ATT" if "att" in res else "", spec.outcome),
        "arr": -res["ate"], "arr_ci": [-res["ci_high"], -res["ci_low"]],
        "comparison_mean_outcome": base,
        "assumptions": ASSUMPTIONS,
        "design_note": ("Logged assignment probabilities are present for every unit (consistent with a randomised design). "
                        "Whether randomisation actually occurred must be confirmed from the experiment protocol, not inferred."
                        if logged else
                        "No logged assignment probabilities: treat as observational. Causal reading requires conditional exchangeability."),
    })
    res["stability"] = "see sensitivity analysis"
    res["warnings"] = list(diag["warnings"]) + (
        ["Unadjusted difference in means is an association; it is causal only if assignment was randomised."] if name == "difference_in_means" else [])
    res["warnings"].append("A narrow interval reflects sampling error only; it does not rule out unmeasured confounding or misspecification.")
    return res


def _spec_dict(spec: CausalSpec) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.__dict__.items()}


def compare_estimators(frame, spec: CausalSpec) -> list[dict]:
    out = []
    for name in ESTIMATORS:
        s = CausalSpec(**{**spec.__dict__, "estimator": name})
        r = estimate(frame, s)
        out.append({k: r.get(k) for k in ("estimator", "status", "ate", "ci_low", "ci_high", "att", "n", "se_method")} | {"estimator": name})
    return out


def e_value(rr: float, rr_ci_limit: float | None = None) -> dict:
    """VanderWeele & Ding (2017) E-value for a risk ratio (protective effects are inverted)."""
    def ev(r):
        r = 1 / r if r < 1 else r
        return float(r + np.sqrt(r * (r - 1)))
    out = {"e_value_point": ev(rr)}
    if rr_ci_limit is not None:
        crosses = (rr < 1 and rr_ci_limit >= 1) or (rr > 1 and rr_ci_limit <= 1)
        out["e_value_ci_limit"] = 1.0 if crosses else ev(rr_ci_limit)
    out["interpretation"] = ("An unmeasured confounder would need to be associated with both treatment and outcome by a risk ratio of at least "
                             "the E-value (beyond measured covariates) to fully explain away the point estimate.")
    return out


def sensitivity(frame: pd.DataFrame, spec: CausalSpec) -> dict:
    """Applicable sensitivity checks for AIPW/IPW-style estimates. Each entry documents what it tests."""
    from ..data.schema import FEATURE_GROUPS, NEGATIVE_CONTROL_OUTCOME
    base_cov = spec.covariates or default_covariates(frame)
    fast = dict(estimator="aipw", bootstrap=0, outcome_learner="logistic", cross_fit_folds=3)

    def run(**kw):
        s = CausalSpec(**{**spec.__dict__, **fast, **kw})
        r = estimate(frame, s)
        return r

    def row(label, r):
        if r.get("status") != "ok":
            return {"label": label, "status": r.get("status"), "reason": r.get("reason")}
        return {"label": label, "status": "ok", "ate": r["ate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": r["n"]}

    ref = run(covariates=base_cov)
    cov_sets = [row("full covariate set", ref)]
    for g, cols in FEATURE_GROUPS.items():
        cs = [c for c in base_cov if c not in cols]
        if cs and len(cs) < len(base_cov):
            cov_sets.append(row(f"without {g}", run(covariates=cs)))
    cov_sets.append(row("minimal (stress + DTI only)", run(covariates=["financial_stress_indicator", "debt_to_income_ratio"])))
    cov_sets.append(row("unadjusted (difference in means)", estimate(frame, CausalSpec(**{**spec.__dict__, "estimator": "difference_in_means", "covariates": base_cov}))))
    ps_models = [row(f"propensity: {m}", run(covariates=base_cov, ps_model=m)) for m in ("logistic", "gbm")]
    trims = [row("no trimming", ref)] + [row(f"overlap-restricted to e in [{lo}, {hi}]", run(covariates=base_cov, trim=(lo, hi)))
                                         for lo, hi in ((0.02, 0.98), (0.05, 0.95), (0.10, 0.90))]
    if NEGATIVE_CONTROL_OUTCOME in frame.columns:
        placebo_r = run(covariates=[c for c in base_cov], outcome=NEGATIVE_CONTROL_OUTCOME)
    else:
        placebo_r = {"status": "unavailable", "reason": "dataset has no pre-treatment placebo/negative-control outcome column"}
    placebo = row("placebo outcome: recent_late_flag (pre-treatment)", placebo_r)
    placebo["interpretation"] = (
        "A pre-treatment outcome cannot be affected by the treatment. A clearly non-zero adjusted estimate suggests residual confounding "
        "(informative but not conclusive; a zero estimate does not prove its absence)."
        if placebo.get("status") == "ok" else "unavailable")
    if placebo.get("status") == "ok":
        placebo["flags_confounding"] = bool(placebo["ci_low"] > 0 or placebo["ci_high"] < 0)
    alt_out = [row(f"outcome: {o}", run(covariates=base_cov, outcome=o)) for o in ("missed_payment", "delinquency_60d", "default_event")
               if o != spec.outcome and o in frame.columns]
    evalue = None
    if ref.get("status") == "ok" and ref["comparison_mean_outcome"] > 0:
        p0 = ref["comparison_mean_outcome"]
        rr = max(p0 + ref["ate"], 1e-6) / p0
        lim = max(p0 + (ref["ci_high"] if ref["ate"] < 0 else ref["ci_low"]), 1e-6) / p0
        evalue = e_value(rr, lim) | {"risk_ratio": rr, "applies_to": "binary outcome; approximate when p0 estimated from comparison arm"}
    ates = [r["ate"] for r in cov_sets[:-1] + ps_models + trims if r.get("status") == "ok"]
    same_sign = len({np.sign(a) for a in ates}) == 1 if ates else False
    rng_ = (max(ates) - min(ates)) if ates else None
    width = (ref["ci_high"] - ref["ci_low"]) if ref.get("status") == "ok" else None
    stability = "insufficient evidence"
    if ates and width:
        stability = "stable" if same_sign and rng_ < width else ("sign-consistent but spread exceeds sampling CI" if same_sign else "unstable (sign changes)")
    return {"reference_estimator": "AIPW (3-fold cross-fit, logistic nuisance)", "covariate_specifications": cov_sets,
            "propensity_models": ps_models, "overlap_restriction": trims, "placebo_outcome": placebo, "alternative_outcomes": alt_out,
            "e_value": evalue, "spread_of_ate_across_specs": rng_, "stability_indicator": stability,
            "notes": ["These checks probe robustness; they do not prove unconfoundedness.",
                      "Overlap-restricted estimates answer a different (trimmed-population) question."]}
