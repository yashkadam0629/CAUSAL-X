import numpy as np
import pandas as pd
import pytest

from app.data.frames import leakage_free_features
from app.data.schema import ARMS, FEATURES, OUTCOMES
from app.data.synthetic import GenConfig, fingerprint, generate
from app.data.validation import inspect_upload, validate_tables, validate_upload


def test_generation_is_reproducible_and_seed_sensitive():
    a = generate(GenConfig(n_customers=1500, seed=3))
    b = generate(GenConfig(n_customers=1500, seed=3))
    c = generate(GenConfig(n_customers=1500, seed=4))
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(c)


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        generate(GenConfig(n_customers=1000, scenario="nope"))


def test_schema_referential_integrity_and_dates(tables_obs):
    rep = validate_tables(tables_obs)
    errors = [f for f in rep["findings"] if f["severity"] == "error"]
    assert errors == [], errors
    assert rep["quality_score"] == 100


def test_outcome_window_after_decision_and_history_before(tables_obs):
    c, o, r, l = (tables_obs[k] for k in ("customers", "outcomes", "repayments", "loans"))
    dec = pd.to_datetime(c.set_index("customer_id")["data_snapshot_date"])
    assert (pd.to_datetime(o["outcome_window_start"]) > o["customer_id"].map(dec)).all()
    due = pd.to_datetime(r["loan_id"].map(l.set_index("loan_id")["contractual_due_date"]))
    assert (pd.to_datetime(r["scheduled_due_date"]) < due).all()


def test_treatment_outcome_linkage_one_row_per_customer(frame_obs, tables_obs):
    assert frame_obs.customer_id.is_unique and len(frame_obs) == len(tables_obs["customers"])
    assert set(frame_obs.arm.unique()) <= set(ARMS)
    assert frame_obs[OUTCOMES].notna().all().all()


def test_assignment_respects_eligibility(frame_obs, frame_rand):
    for f in (frame_obs, frame_rand):
        for a in ARMS:
            assert f.loc[f.arm == a, f"elig_{a}"].all(), a


def test_randomised_has_logged_propensities_observational_does_not(frame_rand, frame_obs):
    assert frame_rand.assignment_probability.notna().all()
    assert frame_obs.assignment_probability.isna().all()


def test_confounding_makes_naive_comparison_misleading(frame_obs, tables_obs):
    d = frame_obs[frame_obs.arm.isin(["none", "standard_reminder"])]
    naive = d[d.arm == "standard_reminder"].delinquency_30d.mean() - d[d.arm == "none"].delinquency_30d.mean()
    truth = tables_obs["ground_truth"]["itt_effect_standard_reminder"].mean()
    assert truth < 0 < naive  # naive comparison has the wrong sign under confounded assignment


def test_ground_truth_is_separate_and_never_a_feature(tables_obs, frame_obs):
    truth_cols = set(tables_obs["ground_truth"].columns) - {"customer_id"}
    assert not truth_cols & set(frame_obs.columns), "ground truth leaked into the analysis frame"
    assert not truth_cols & set(FEATURES)
    assert leakage_free_features() == FEATURES


def test_upload_validation_reports_not_invents():
    df = pd.DataFrame({"id": [1, 2, 2, 3] * 100, "d": ["2025-01-01", "bad", "2025-02-01", "2025-03-01"] * 100, "y": [0, 1, 1, 0] * 100, "x": range(400)})
    rep = validate_upload(df, {"customer_id": "id", "decision_date": "d", "outcome": "y"})
    codes = {f["code"] for f in rep["findings"]}
    assert {"duplicates", "bad_dates", "no_treatment"} <= codes
    assert rep["capabilities"]["causal_effect_estimation"]["supported"] is False
    assert rep["rows_analysable"] < rep["rows_in"] and rep["exclusions"]
    assert inspect_upload(df)["suggested_mapping"]["customer_id"] == "id"


def test_upload_leakage_signal_detected():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    df = pd.DataFrame({"id": range(600), "y": y, "leaky": y + rng.normal(0, 0.05, 600), "recovered_amount": rng.random(600)})
    rep = validate_upload(df, {"customer_id": "id", "outcome": "y"})
    codes = {f["code"] for f in rep["findings"]}
    assert "leakage_signal" in codes and "leakage_name" in codes
