import numpy as np
import pandas as pd
import pytest

from app.experiment_design.sample_size import design, n_two_proportions, protocol
from app.monitoring.drift import level, psi
from app.governance.fairness import MIN_GROUP_N, audit, wilson
from app.uplift.hte import SEGMENTS, auuc, ipw_benefit_scores, qini_coefficient, segment, uplift_curve


def test_segment_definitions_and_sign_convention():
    arr = np.array([0.12, 0.05, 0.01, 0.0, -0.10, 0.2])
    se = np.array([0.01, 0.01, 0.005, 0.1, 0.01, 0.01])
    elig = np.array([True, True, True, True, True, False])
    s = segment(arr, se, elig)
    assert list(s) == ["high_benefit", "moderate_benefit", "low_benefit", "uncertain_effect", "possibly_counterproductive", "outside_supported_population"]
    assert set(s) <= set(SEGMENTS)


def test_uplift_curve_perfect_ranker_beats_random_and_random_has_zero_qini():
    rng = np.random.default_rng(0)
    n = 20000
    true_arr = rng.uniform(-0.05, 0.25, n)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < np.where(t == 1, 0.4 - true_arr, 0.4)).astype(float)
    sig = ipw_benefit_scores(t, y, np.full(n, 0.5))
    f, c, _ = uplift_curve(true_arr, sig)
    fr, cr, _ = uplift_curve(rng.random(n), sig)
    assert qini_coefficient(f, c) > 0.01
    assert abs(qini_coefficient(fr, cr)) < 0.005
    assert auuc(f, c) > auuc(fr, cr)
    assert c[-1] == pytest.approx(sig.mean(), rel=1e-6)


def test_ipw_signal_is_unbiased_for_arr():
    rng = np.random.default_rng(1)
    n = 1_000_000
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < np.where(t == 1, 0.25, 0.30)).astype(float)
    assert ipw_benefit_scores(t, y, np.full(n, 0.5)).mean() == pytest.approx(0.05, abs=0.006)


def test_sample_size_matches_hand_calculation():
    n_c, n_t = n_two_proportions(0.20, 0.05, 0.05, 0.8)
    assert n_c == n_t and 1050 < n_c < 1150  # ~1091 per arm
    assert design("binary", 0.2, 0.05)["n_control"] == n_c


def test_adjustments_inflate_sample_size_as_documented():
    base = design("binary", 0.25, -0.03)
    assert design("binary", 0.25, -0.03, compliance=0.5)["n_control"] > 3.5 * base["n_control"]  # ~1/c^2
    assert design("binary", 0.25, -0.03, attrition=0.2)["n_control"] == pytest.approx(base["n_control"] / 0.8, rel=0.01)
    assert design("binary", 0.25, -0.03, n_arms_vs_control=3)["n_control"] > base["n_control"]
    assert design("binary", 0.25, -0.03, icc=0.05, cluster_size=20)["adjustments"]["design_effect"] == pytest.approx(1.95)
    cont = design("continuous", 0, 0.1, sd=1.0)
    assert 1500 < cont["n_control"] < 1650  # 2*(1.96+0.84)^2/0.01 = ~1568


def test_invalid_designs_rejected():
    with pytest.raises(ValueError):
        design("binary", 0.25, 0.0)
    with pytest.raises(ValueError):
        design("binary", 0.02, -0.05)
    with pytest.raises(ValueError):
        design("continuous", 0, 0.1)
    p = protocol({"objective": "x"}, design("binary", 0.25, -0.03))
    assert "not approved" in p["status"].lower()


def test_psi_detects_shift_and_stability():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=5000))
    assert level(psi(a, pd.Series(rng.normal(size=5000)))) == "stable"
    assert level(psi(a, pd.Series(rng.normal(1.5, 1, 5000)))) == "significant"
    assert level(psi(a.head(10), a.tail(10))) == "insufficient_data"


def test_fairness_suppresses_small_groups_and_reports_intervals():
    rng = np.random.default_rng(0)
    n = 1000
    d = pd.DataFrame({"g": ["a"] * 900 + ["b"] * 40 + ["c"] * 60, "delinquency_30d": rng.integers(0, 2, n)})
    out = audit(d, rng.random(n), rng.random(n) < 0.3, rng.random(n), "g")
    rows = {r["group"]: r for r in out["groups"]}
    assert rows["b"]["suppressed"] and rows["c"]["suppressed"] and not rows["a"]["suppressed"]
    lo, hi = rows["a"]["selection_rate_ci"]
    assert lo < rows["a"]["selection_rate"] < hi and "not proof" in " ".join(out["caveats"])
    assert MIN_GROUP_N == 100 and wilson(0, 0) == [None, None]
