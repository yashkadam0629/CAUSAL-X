"""Off-policy evaluation of intervention policies on held-out logged data.

Doubly robust (augmented IPS) estimate of the expected adverse-outcome rate under a deterministic policy pi:

    V_hat(pi) = mean_i [ mu_{pi(i)}(x_i) + 1{A_i = pi(i)} / e_{pi(i)}(x_i) * (Y_i - mu_{pi(i)}(x_i)) ]

mu_a (outcome models) and e_a (assignment propensities) are fitted on training cohorts only and applied to the held-out
cohorts; the policies being evaluated were optimised using predicted effects for these customers but never their outcomes.
Validity requires (i) correct logging propensities (known in randomised data, *estimated* otherwise -> needs exchangeability)
and (ii) positivity for the arms the policy selects. Weights are clipped at e >= clip and the clipped share is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..causal.estimators import fit_outcome_model
from ..data.schema import ARMS, CATEGORICAL, NUMERIC
from ..predictive.risk import make_preprocessor


class LoggingNuisance:
    def __init__(self, frame: pd.DataFrame, outcome="delinquency_30d", seed=42):
        tr = frame[frame.split == "train"]
        self.feats = NUMERIC + CATEGORICAL
        self.prep = make_preprocessor().fit(tr[self.feats])
        Z = np.asarray(self.prep.transform(tr[self.feats]), float)
        y = tr[outcome].astype(float).values
        self.mu = {}
        for a in ARMS:
            m = (tr.arm == a).values
            self.mu[a] = fit_outcome_model(Z[m], y[m], "binary", "gbm", seed) if m.sum() > 50 else None
        self.prop = LogisticRegression(max_iter=600).fit(Z, tr.arm.map({a: k for k, a in enumerate(ARMS)}).values)
        self.outcome = outcome

    def transform(self, d: pd.DataFrame):
        Z = np.asarray(self.prep.transform(d[self.feats]), float)
        mu = np.column_stack([self.mu[a](Z) if self.mu[a] else np.full(len(d), np.nan) for a in ARMS])
        P = self.prop.predict_proba(Z)
        el = np.column_stack([d[f"elig_{a}"].values for a in ARMS])
        P = np.where(el, P, 0.0)
        P = P / P.sum(axis=1, keepdims=True)  # eligibility-aware multinomial propensities
        return mu, P


def evaluate_policy(frame: pd.DataFrame, nuis: LoggingNuisance, d_idx: np.ndarray, action_idx: np.ndarray, outcome="delinquency_30d",
                    clip=0.02, truth_arr_value: float | None = None, use_logged_propensity=True) -> dict:
    """`d_idx`: positions (in `frame`) of evaluated customers (must be held-out). `action_idx`: ARMS index chosen per customer (0 = none)."""
    d = frame.iloc[d_idx]
    if (d.split == "train").any():
        raise ValueError("policy evaluation must use held-out (non-training) cohorts")
    mu, P = nuis.transform(d)
    A = d.arm.map({a: k for k, a in enumerate(ARMS)}).values
    y = d[outcome].astype(float).values
    n = len(d)
    logged_known = d["assignment_probability"].notna().all()
    e_logged = np.where(logged_known, d["assignment_probability"].fillna(0).values, P[np.arange(n), A])
    pi = action_idx
    mu_pi = mu[np.arange(n), pi]
    e_pi = P[np.arange(n), pi]
    if logged_known and use_logged_propensity:  # known logging propensity for the logged arm; policy arm propensity needs per-arm probs.
        e_pi = np.where(pi == A, e_logged, P[np.arange(n), pi])
    e_c = np.clip(e_pi, clip, 1.0)
    match = (A == pi).astype(float)
    psi = mu_pi + match / e_c * (y - mu_pi)
    base = np.zeros(n, int)
    mu_0 = mu[:, 0]; e0 = np.clip(np.where(logged_known & (A == 0), e_logged, P[:, 0]), clip, 1.0)
    psi0 = mu_0 + (A == 0) / e0 * (y - mu_0)
    diff = psi0 - psi  # events prevented per customer (policy vs no intervention)
    se = diff.std(ddof=1) / np.sqrt(n)
    out = {
        "n_evaluated": int(n), "policy_rate_dr": float(psi.mean()), "no_intervention_rate_dr": float(psi0.mean()),
        "estimated_events_prevented": float(diff.sum()), "events_prevented_ci": [float((diff.mean() - 1.96 * se) * n), float((diff.mean() + 1.96 * se) * n)],
        "events_prevented_per_1000_customers": float(diff.mean() * 1000),
        "observed_rate_under_logged_policy": float(y.mean()), "clipped_share": float((e_pi < clip).mean()),
        "share_policy_matches_logged_arm": float(match.mean()),
        "method": "Doubly robust off-policy evaluation, nuisance fitted on training cohorts only",
        "propensity_source": "logged assignment probabilities for the logged arm (randomised design)" if logged_known else
                             "estimated multinomial propensities (observational: validity rests on conditional exchangeability)",
        "assumptions": ["Conditional exchangeability given covariates (guaranteed only by randomisation)",
                        "Positivity for the actions the policy selects", "Correct outcome or propensity model (doubly robust)",
                        "No interference between customers; policy applied as ITT assignment"],
    }
    if truth_arr_value is not None:
        out["synthetic_truth_events_prevented"] = float(truth_arr_value)
        out["estimation_error_vs_truth"] = float(out["estimated_events_prevented"] - truth_arr_value)
        out["synthetic_label"] = "SYNTHETIC GROUND TRUTH (benchmark only)"
    return out
