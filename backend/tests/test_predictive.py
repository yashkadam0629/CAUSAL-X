import numpy as np

from app.data.schema import FEATURES, OUTCOMES
from app.predictive.risk import calibration_table, evaluate, expected_cost_threshold, load_bundle, make_model, save_bundle, train_risk


def test_temporal_split_integrity(frame_rand):
    tr, va, te = (frame_rand[frame_rand.split == s] for s in ("train", "validation", "test"))
    assert tr.cohort_month.max() < va.cohort_month.min() and va.cohort_month.max() < te.cohort_month.min()


def test_no_outcome_or_treatment_features():
    assert not set(FEATURES) & (set(OUTCOMES) | {"arm", "intervention_cost", "recent_late_flag", "age_band", "region_group"})


def test_preprocessing_fit_on_training_only_and_pipeline_consistency(frame_rand):
    tr = frame_rand[frame_rand.split == "train"]
    m = make_model("logistic_regression", 1).fit(tr[FEATURES], tr["delinquency_30d"])
    scaler = m.named_steps["prep"].named_transformers_["num"].named_steps["sc"]
    num = m.named_steps["prep"].transformers_[0][2]
    # the scaler sees training rows only, so its means match the training-row means (median-imputed), not the full data
    imputed = tr[num].fillna(tr[num].median())
    assert np.allclose(scaler.mean_, imputed.mean().values, rtol=1e-6, atol=1e-6)
    p1, p2 = m.predict_proba(frame_rand[FEATURES].head(50)), m.predict_proba(frame_rand[FEATURES].head(50))
    assert np.allclose(p1, p2)


def test_metrics_and_calibration_output():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)
    m = evaluate(y, p, 0.5, 1000, 50)
    assert 0.78 < m["roc_auc"] < 0.88 and m["brier"] < 0.3 and sum(m["confusion_matrix"].values()) == 2000
    cal = calibration_table(y, p)
    assert len(cal) == 10 and all(abs(c["mean_predicted"] - c["observed_rate"]) < 0.12 for c in cal)
    thr, cost = expected_cost_threshold(y, p, 1000, 50)
    assert 0.02 <= thr <= 0.9 and cost > 0


def test_train_serialise_roundtrip(frame_rand, tmp_path):
    b = train_risk(frame_rand, seed=1)
    assert b.metrics["test"]["roc_auc"] > 0.6
    path = tmp_path / "bundle.joblib"
    save_bundle(b, path)
    b2 = load_bundle(path)
    sample = frame_rand.head(100)
    assert np.allclose(b.predict(sample), b2.predict(sample))
    assert b.config["selection_rule"].startswith("lowest validation Brier")
