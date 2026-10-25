import numpy as np
import pandas as pd
import pytest

from app.causal.estimators import CausalSpec, e_value, ess, estimate, prepare, sensitivity, smd_table, weights_for


def true_ate(tables, spec, frame):
    d, _, _ = prepare(frame, spec)
    return tables["ground_truth"].set_index("customer_id").loc[d.customer_id, f"itt_effect_{spec.treatment}"].mean()


@pytest.mark.parametrize("est", ["difference_in_means", "regression_adjustment", "ipw", "aipw"])
def test_randomised_estimators_recover_truth(est, frame_rand, tables_rand):
    spec = CausalSpec(treatment="standard_reminder", estimator=est, bootstrap=30)
    r = estimate(frame_rand, spec)
    t = true_ate(tables_rand, spec, frame_rand)
    assert r["status"] == "ok" and r["ate"] < 0, "sign convention: reminders reduce adverse outcomes"
    assert abs(r["ate"] - t) < max(3.5 * r["se"], 0.02), (est, r["ate"], t, r["se"])
    assert r["ci_low"] < r["ate"] < r["ci_high"]


def test_confounded_naive_estimate_is_wrong_adjusted_is_close(frame_obs, tables_obs):
    spec = CausalSpec(treatment="standard_reminder", bootstrap=20)
    t = true_ate(tables_obs, spec, frame_obs)
    naive = estimate(frame_obs, CausalSpec(**{**spec.__dict__, "estimator": "difference_in_means"}))
    adj = estimate(frame_obs, CausalSpec(**{**spec.__dict__, "estimator": "aipw"}))
    assert naive["ate"] > 0 > t, "naive association has the opposite sign under confounding"
    assert abs(adj["ate"] - t) < 0.02 and adj["ate"] < 0
    assert any("association" in w for w in naive["warnings"])


def test_hidden_confounding_is_flagged_by_placebo(frame_hidden, frame_obs):
    s_h = sensitivity(frame_hidden, CausalSpec(treatment="standard_reminder"))
    s_o = sensitivity(frame_obs, CausalSpec(treatment="standard_reminder"))
    assert s_h["placebo_outcome"]["flags_confounding"] is True
    assert s_o["placebo_outcome"]["flags_confounding"] is False
    assert s_h["e_value"]["e_value_point"] > 1 and "stability_indicator" in s_h


def test_sensitivity_structure_and_alt_outcomes(frame_obs):
    s = sensitivity(frame_obs, CausalSpec(treatment="counselling"))
    assert len(s["covariate_specifications"]) >= 4 and len(s["overlap_restriction"]) == 4 and len(s["alternative_outcomes"]) == 3
    assert "prove" in " ".join(s["notes"]).lower()


def test_overlap_and_balance_diagnostics(frame_obs):
    r = estimate(frame_obs, CausalSpec(treatment="repayment_plan", estimator="ipw", bootstrap=10))
    d = r["diagnostics"]
    assert d["max_abs_smd_weighted"] < d["max_abs_smd_unadjusted"]
    assert 0 < d["ess_treated"] <= d["n_treated"]
    assert sum(d["overlap_histogram"]["treated"]) == d["n_treated"]


def test_poor_overlap_triggers_warning():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-4 * x))).astype(int)
    df = pd.DataFrame({"arm": np.where(t == 1, "A", "none"), "x1": x, "x2": rng.normal(size=n), "y": (rng.random(n) < 0.3).astype(int)})
    r = estimate(df, CausalSpec(treatment="A", comparison="none", outcome="y", covariates=["x1", "x2"], estimator="ipw", bootstrap=5, restrict_eligible=False))
    assert any("overlap" in w.lower() for w in r["warnings"])


def test_insufficient_evidence_instead_of_numbers():
    df = pd.DataFrame({"arm": ["A"] * 10 + ["none"] * 500, "x": np.arange(510.0), "y": 0})
    r = estimate(df, CausalSpec(treatment="A", comparison="none", outcome="y", covariates=["x"], restrict_eligible=False))
    assert r["status"] == "insufficient_evidence" and "ate" not in r


def test_extreme_weights_reported_not_hidden():
    t = np.array([1] * 50 + [0] * 50)
    e = np.full(100, 0.5)
    e[0] = 0.001
    w = weights_for(t, e, "ATE")
    assert w.max() > 500 and ess(w) < 100


def test_e_value_known_values():
    assert e_value(2.0)["e_value_point"] == pytest.approx(2 + np.sqrt(2), rel=1e-6)
    assert e_value(0.5)["e_value_point"] == pytest.approx(2 + np.sqrt(2), rel=1e-6)  # protective effects are inverted
    assert e_value(0.8, 1.1)["e_value_ci_limit"] == 1.0  # interval crosses the null


def test_smd_zero_when_balanced():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.choice(["u", "v"], 5000)})
    t = rng.integers(0, 2, 5000)
    assert smd_table(d, ["a", "b"], t)["smd_unadjusted"].abs().max() < 0.1
