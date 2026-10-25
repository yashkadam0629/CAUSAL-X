"""Layer C: heterogeneous treatment effects (CATE) via the cross-fitted DR-learner, plus a T-learner for comparison.

Conventions: tau(x) = E[Y(a) - Y(none) | x] on an adverse outcome (negative is beneficial); benefit ARR(x) = -tau(x).
CATE estimates are model estimates of a *conditional average*, not facts about an individual customer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ..causal.estimators import crossfit_nuisance, fit_outcome_model
from ..data.schema import CATEGORICAL, FEATURES, NUMERIC, TREATMENTS
from ..predictive.risk import make_preprocessor


@dataclass
class CateModel:
    arm: str
    outcome: str
    features: list
    prep: object
    final_models: list
    t_mu0: object
    t_mu1: object
    n_train: int
    method: str = "DR-learner (cross-fitted nuisance, LightGBM final stage), bootstrap over final stage"
    meta: dict = field(default_factory=dict)

    def predict(self, d: pd.DataFrame):
        Z = np.asarray(self.prep.transform(d[self.features]), float)
        preds = np.vstack([m.predict(Z) for m in self.final_models])  # B+1 rows; row 0 is the full-sample fit
        tau = preds[0]
        se = preds[1:].std(axis=0, ddof=1) if len(preds) > 2 else np.zeros(len(d))
        tlearn = self.t_mu1(Z) - self.t_mu0(Z)
        return tau, se, tlearn


def _final(seed):
    return LGBMRegressor(n_estimators=120, learning_rate=0.03, num_leaves=6, min_child_samples=120, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, verbose=-1, n_jobs=2)


def fit_cate(frame: pd.DataFrame, arm: str, outcome: str = "delinquency_30d", features=None, seed: int = 42,
             boots: int = 20, clip: float = 0.02) -> CateModel:
    """Fit on training cohorts (0-7) only, among customers in {none, arm} eligible for `arm`."""
    features = features or FEATURES
    num = [f for f in features if f in NUMERIC]; cat = [f for f in features if f in CATEGORICAL]
    s = frame[(frame.split == "train") & frame.arm.isin(["none", arm]) & frame[f"elig_{arm}"]].reset_index(drop=True)
    t = (s.arm == arm).astype(int).values
    y = s[outcome].astype(float).values
    prep = make_preprocessor(num, cat)
    Z = np.asarray(prep.fit_transform(s[num + cat]), float)
    mu0, mu1, e = crossfit_nuisance(Z, t, y, "binary", "gbm", "logistic", 5, seed)
    e = np.clip(e, clip, 1 - clip)
    phi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)  # DR pseudo-outcome, E[phi|x] = tau(x)
    rng = np.random.default_rng(seed)
    models = [_final(seed).fit(Z, phi)]
    for b in range(boots):
        idx = rng.integers(0, len(Z), len(Z))
        models.append(_final(seed + b + 1).fit(Z[idx], phi[idx]))
    f0 = fit_outcome_model(Z[t == 0], y[t == 0], "binary", "gbm", seed)
    f1 = fit_outcome_model(Z[t == 1], y[t == 1], "binary", "gbm", seed)
    return CateModel(arm, outcome, features, prep, models, f0, f1, len(s),
                     meta={"n_treated": int(t.sum()), "n_comparison": int((1 - t).sum()), "ate_from_pseudo_outcomes": float(phi.mean()),
                           "bootstrap_B": boots, "uncertainty_note": "Bootstrap over the final stage only; excludes nuisance-model and confounding uncertainty."})


def score_all(frame: pd.DataFrame, models: dict[str, CateModel]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for a, m in models.items():
        tau, se, tl = m.predict(frame)
        out[f"arr_{a}"] = -tau
        out[f"se_{a}"] = se
        out[f"arr_tlearner_{a}"] = -tl
    return out


# ---------------------------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------------------------
SEGMENTS = ["high_benefit", "moderate_benefit", "low_benefit", "uncertain_effect", "possibly_counterproductive", "outside_supported_population"]


def segment(arr: np.ndarray, se: np.ndarray, eligible: np.ndarray, high=0.08, moderate=0.03, z=1.28) -> np.ndarray:
    """Interpretable, configurable segments (heuristic thresholds, not scientific categories)."""
    lcb, ucb = arr - z * se, arr + z * se
    seg = np.where(lcb > 0, np.where(arr >= high, "high_benefit", np.where(arr >= moderate, "moderate_benefit", "low_benefit")),
                   np.where(ucb < 0, "possibly_counterproductive", "uncertain_effect"))
    return np.where(eligible, seg, "outside_supported_population")


# ---------------------------------------------------------------------------------------------
# Uplift evaluation (weighted Qini / uplift curve on held-out cohorts)
# ---------------------------------------------------------------------------------------------
def ipw_benefit_scores(t, y, e):
    """Unbiased per-unit signal for the *benefit* ARR: E[score|x] = ARR(x) when e is the true propensity."""
    return (1 - t) / (1 - e) * y - t / e * y


def uplift_curve(score_rank: np.ndarray, signal: np.ndarray, points=50):
    order = np.argsort(-score_rank, kind="stable")
    cum = np.cumsum(signal[order]) / len(signal)  # incremental prevented events per eligible unit
    frac = np.arange(1, len(signal) + 1) / len(signal)
    idx = np.unique(np.linspace(0, len(signal) - 1, points).astype(int))
    return frac, cum, idx


def auuc(frac, cum):
    x = np.concatenate([[0], frac]); yv = np.concatenate([[0], cum])
    return float(np.trapezoid(yv, x))


def qini_coefficient(frac, cum):
    x = np.concatenate([[0], frac]); yv = np.concatenate([[0], cum])
    return float(np.trapezoid(yv - yv[-1] * x, x))


def evaluate_uplift(frame: pd.DataFrame, scores: pd.DataFrame, arm: str, risk: np.ndarray, truth: pd.DataFrame | None,
                    e_hat: np.ndarray, boots: int = 200, seed: int = 42, outcome: str = "delinquency_30d") -> dict:
    """Evaluate rankers on the untouched test cohorts. `e_hat` = P(T=arm | in pair, x) from a propensity model fit on training cohorts."""
    m = (frame.split == "test") & frame.arm.isin(["none", arm]) & frame[f"elig_{arm}"]
    d = frame[m]
    if len(d) < 300 or (d.arm == arm).sum() < 50 or (d.arm == "none").sum() < 50:
        return {"status": "insufficient_evidence", "reason": "held-out test cohorts too small for uplift evaluation", "n": int(len(d))}
    t = (d.arm == arm).astype(int).values
    y = d[outcome].astype(float).values
    e = np.clip(e_hat[m.values], 0.02, 0.98)
    sig = ipw_benefit_scores(t, y, e)
    rankers = {"dr_learner": scores.loc[m, f"arr_{arm}"].values, "t_learner": scores.loc[m, f"arr_tlearner_{arm}"].values,
               "baseline_risk_only": risk[m.values]}
    rng = np.random.default_rng(seed)
    n = len(d)
    res = {"status": "ok", "arm": arm, "n_test": int(n), "n_treated": int(t.sum()), "n_comparison": int((1 - t).sum()), "rankers": {},
           "method": "IPW-weighted uplift (Qini) curve on held-out cohorts 10-11; propensities fitted on training cohorts",
           "caveat": "For observational data, curve validity additionally requires conditional exchangeability and a good propensity model.",
           "total_benefit_per_unit": float(sig.mean())}
    boot_idx = [rng.integers(0, n, n) for _ in range(boots)]
    for name, rk in rankers.items():
        frac, cum, idx = uplift_curve(rk, sig)
        a, q = auuc(frac, cum), qini_coefficient(frac, cum)
        bq = []
        for bi in boot_idx:
            f_, c_, _ = uplift_curve(rk[bi], sig[bi]); bq.append(qini_coefficient(f_, c_))
        res["rankers"][name] = {"auuc": a, "qini": q, "qini_ci": [float(np.percentile(bq, 2.5)), float(np.percentile(bq, 97.5))],
                                "curve": [{"fraction": float(frac[i]), "cum_benefit": float(cum[i])} for i in idx]}
    # calibration of predicted benefit by decile
    pred = rankers["dr_learner"]
    dec = pd.qcut(pd.Series(pred).rank(method="first"), 5, labels=False).values
    cal = []
    for k in range(5):
        mk = dec == k
        cal.append({"quintile": k + 1, "mean_predicted_arr": float(pred[mk].mean()), "ipw_realised_arr": float(sig[mk].mean()),
                    "se": float(sig[mk].std(ddof=1) / np.sqrt(mk.sum())), "n": int(mk.sum())})
    res["calibration"] = cal
    if truth is not None:
        tr = truth.set_index("customer_id").loc[d.customer_id, f"itt_effect_{arm}"].values * -1
        res["synthetic_benchmark"] = {
            "label": "SYNTHETIC GROUND TRUTH (benchmark only, never a model input)",
            "corr_pred_vs_true_arr": float(np.corrcoef(pred, tr)[0, 1]), "rmse": float(np.sqrt(np.mean((pred - tr) ** 2))),
            "mean_true_arr": float(tr.mean()), "mean_predicted_arr": float(pred.mean()),
            "corr_risk_vs_true_arr": float(np.corrcoef(risk[m.values], tr)[0, 1]),
        }
    return res
