"""API contract tests and the full end-to-end acceptance run (isolated temp database and data directory)."""
import io
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app

API = "/api/v1"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def wait(c, job_id, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"{API}/jobs/{job_id}").json()
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(1.5)
    raise TimeoutError(job_id)


@pytest.fixture(scope="module")
def ds(client):
    r = client.post(f"{API}/datasets/generate", json={"name": "acc_test", "n_customers": 14000, "scenario": "observational", "seed": 11, "train": True})
    assert r.status_code == 202
    j = wait(client, r.json()["job_id"])
    assert j["status"] == "succeeded", j
    return next(d for d in client.get(f"{API}/datasets").json() if d["name"] == "acc_test")


def test_health_config_metadata(client):
    assert client.get(f"{API}/health").json()["status"] == "ok"
    cfg = client.get(f"{API}/config").json()
    assert "randomized" in cfg["scenarios"] and "no authentication" in cfg["security_model"].lower()
    assert "tau" in client.get(f"{API}/metadata").json()["sign_convention"]


def test_missing_dataset_and_invalid_requests(client):
    assert client.get(f"{API}/datasets/9999").status_code == 404
    assert client.post(f"{API}/causal/estimate", json={"dataset_id": 9999}).status_code == 404
    assert client.post(f"{API}/datasets/generate", json={"name": "bad name!", "n_customers": 5000}).status_code == 422
    assert client.post(f"{API}/datasets/generate", json={"name": "x", "n_customers": 5000, "scenario": "nope"}).status_code == 422
    assert client.post(f"{API}/experiments/sample-size", json={"mde_absolute": 0.0}).status_code == 422
    assert client.post(f"{API}/experiments/sample-size", json={"mde_absolute": -0.03, "alpha": 3}).status_code == 422
    assert client.get(f"{API}/jobs/nope").status_code == 404
    assert client.get(f"{API}/reports/../../etc/passwd").status_code in (400, 404)
    assert client.get(f"{API}/reports/evil.txt").status_code == 400


def test_malformed_uploads(client):
    assert client.post(f"{API}/datasets/upload", files={"file": ("a.txt", b"x,y\n1,2", "text/plain")}).status_code == 415
    assert client.post(f"{API}/datasets/upload", files={"file": ("a.csv", b"", "text/csv")}).status_code == 422
    assert client.post(f"{API}/datasets/upload", files={"file": ("a.csv", b"onlyonecolumn\n1\n2", "text/csv")}).status_code == 422


def test_upload_flow_and_unsupported_analyses_are_honest(client):
    rows = ["id,date,y,x"] + [f"{i},2025-0{1 + i % 6},{i % 2},{i * 0.5}" for i in range(400)]
    r = client.post(f"{API}/datasets/upload", files={"file": ("d.csv", "\n".join(rows).encode(), "text/csv")})
    assert r.status_code == 200 and r.json()["suggested_mapping"]["customer_id"] == "id"
    c = client.post(f"{API}/datasets/upload/commit", json={"upload_token": r.json()["upload_token"], "name": "up1", "mapping": {"customer_id": "id", "decision_date": "date", "outcome": "y"}})
    assert c.status_code == 200
    caps = c.json()["validation"]["capabilities"]
    assert caps["causal_effect_estimation"]["supported"] is False
    d = next(x for x in client.get(f"{API}/datasets").json() if x["name"] == "up1")
    assert client.post(f"{API}/models/train", json={"dataset_id": d["id"]}).status_code == 422  # not supported, explained
    assert client.post(f"{API}/causal/estimate", json={"dataset_id": d["id"], "treatment": "A", "comparison": "none", "outcome": "y"}).status_code == 422
    assert client.delete(f"{API}/datasets/{d['id']}").status_code == 400  # destructive action needs confirmation
    assert client.delete(f"{API}/datasets/{d['id']}", params={"confirm": True}).status_code == 200


def test_untrained_dataset_returns_clear_409(client, ds):
    r = client.post(f"{API}/datasets/generate", json={"name": "untrained", "n_customers": 1500, "train": False})
    wait(client, r.json()["job_id"])
    d = next(x for x in client.get(f"{API}/datasets").json() if x["name"] == "untrained")
    resp = client.get(f"{API}/uplift/metrics", params={"dataset_id": d["id"]})
    assert resp.status_code == 409 and "models/train" in resp.json()["detail"]


def test_pagination_and_filters(client, ds):
    p1 = client.get(f"{API}/customers", params={"dataset_id": ds["id"], "page": 1, "page_size": 10}).json()
    p2 = client.get(f"{API}/customers", params={"dataset_id": ds["id"], "page": 2, "page_size": 10}).json()
    assert len(p1["items"]) == 10 and p1["total"] == 14000
    assert {i["customer_id"] for i in p1["items"]}.isdisjoint({i["customer_id"] for i in p2["items"]})
    f = client.get(f"{API}/customers", params={"dataset_id": ds["id"], "loan_type": "auto", "sort": "risk"}).json()
    assert f["total"] > 0 and all(i["loan_type"] == "auto" for i in f["items"])
    assert client.get(f"{API}/customers", params={"dataset_id": ds["id"], "q": "no-such-id"}).json()["items"] == []
    assert client.get(f"{API}/customers", params={"dataset_id": ds["id"], "sort": "bogus"}).status_code == 422
    assert client.get(f"{API}/customers", params={"dataset_id": ds["id"], "page_size": 1000}).status_code == 422


def test_causal_endpoints_validation(client, ds):
    base = {"dataset_id": ds["id"]}
    assert client.post(f"{API}/causal/estimate", json={**base, "treatment": "nonexistent"}).status_code == 422
    assert client.post(f"{API}/causal/estimate", json={**base, "treatment": "counselling", "comparison": "counselling"}).status_code == 422
    assert client.post(f"{API}/causal/estimate", json={**base, "covariates": ["delinquency_30d"]}).status_code == 422  # outcome as covariate
    assert client.post(f"{API}/causal/estimate", json={**base, "estimator": "magic"}).status_code == 422


def test_full_acceptance_workflow(client, ds):
    did = ds["id"]
    # 1-4 data generated and loaded; validate
    assert ds["models_trained"] and ds["fingerprint"]
    v = client.post(f"{API}/datasets/{did}/validate").json()
    assert v["quality_score"] == 100 and client.get(f"{API}/datasets/{did}/quality-report").json()["score"] == 100
    # 5 predictive models
    models = client.get(f"{API}/models", params={"dataset_id": did}).json()
    risk = next(m for m in models if m["kind"] == "risk")
    met = client.get(f"{API}/models/{risk['model_id']}/metrics").json()
    assert met["metrics"]["test"]["roc_auc"] > 0.6 and "predictive" in met["metric_family"]
    # 6-7 causal estimate, diagnostics, sensitivity
    est = client.post(f"{API}/causal/estimate", json={"dataset_id": did, "treatment": "standard_reminder", "estimator": "aipw"}).json()
    assert est["status"] == "ok" and est["ate"] < 0 and est["synthetic_benchmark"]["true_ate_itt"] < 0
    assert est["interpretability"]["status"] in ("conditional_on_assumptions", "design_supported")
    naive = client.post(f"{API}/causal/estimate", json={"dataset_id": did, "estimator": "difference_in_means"}).json()
    assert naive["interpretability"]["status"] == "association_only"
    diag = client.post(f"{API}/causal/diagnostics", json={"dataset_id": did}).json()
    assert "overlap_histogram" in diag["diagnostics"]
    sens = client.post(f"{API}/causal/sensitivity-analysis", json={"dataset_id": did}).json()
    assert "placebo_outcome" in sens and "e_value" in sens
    exp = client.post(f"{API}/causal/experiments", json={"dataset_id": did, "name": "reminder aipw", "config": {"a": 1}, "results": {"ate": est["ate"]}})
    assert exp.status_code == 201 and any(e["name"] == "reminder aipw" for e in client.get(f"{API}/causal/experiments").json())
    # 8 uplift
    um = client.get(f"{API}/uplift/metrics", params={"dataset_id": did}).json()
    assert um["evaluation"]["standard_reminder"]["status"] == "ok"
    us = client.get(f"{API}/uplift/segments", params={"dataset_id": did, "scatter_n": 50}).json()
    assert sum(us["counts"].values()) == us["n"]
    # 9-10 optimisation and comparison
    run = client.post(f"{API}/optimization/run", json={"dataset_id": did, "budget": 60000, "agent_capacity": 40, "save": True}).json()
    s = run["summary"]
    assert s["solver_status"] == "optimal" and s["budget_used"] <= 60000 + 1e-6 and s["agent_slots_used"] <= 40 + 1e-6
    keys = {x["key"] for x in run["strategies"]}
    assert {"no_intervention", "random", "highest_risk_first_intensive", "gross_benefit_first", "net_benefit_first", "optimised", "conservative"} <= keys
    rid = run["run_id"]
    assert rid and run["fairness_audit"]["age_band"]["caveats"]
    infeasible = client.post(f"{API}/optimization/run", json={"dataset_id": did, "objective": "min_cost", "target_events": 1e7, "save": False}).json()
    assert infeasible["summary"]["solver_status"] == "infeasible" and infeasible["run_id"] is None and infeasible["advice"]
    pe = client.post(f"{API}/policies/evaluate", json={"dataset_id": did, "base": {"budget": 60000, "agent_capacity": 40}}).json()
    assert "OBSERVATIONAL" in pe["evidence_type"] and len(pe["policies"]) >= 7
    pol = client.post(f"{API}/policies", json={"name": "base policy", "definition": {"budget": 60000}})
    assert pol.status_code == 201 and client.get(f"{API}/policies").json()
    # 11 scenarios
    sc = client.post(f"{API}/scenarios/run", json={"dataset_id": did, "scenarios": ["A_base", "B_budget_cut", "J_conservative"], "mc_draws": 50, "save_name": "acc set", "base": {"budget": 60000, "agent_capacity": 40}}).json()
    assert len(sc["results"]) == 3 and "SIMULATED" in sc["label"]
    assert client.get(f"{API}/scenarios/{sc['scenario_id']}").json()["name"] == "acc set"
    # 12-13 saved recommendations are retrievable and auditable; review with override rules
    dec = client.get(f"{API}/audit/decisions", params={"dataset_id": did, "run_id": rid, "page_size": 5}).json()
    assert dec["total"] == s["n_selected"] and dec["items"][0]["explanation"]["candidate_interventions"]
    d0 = dec["items"][0]
    assert d0["model_version"] and d0["policy_version"] and d0["data_version"]
    bad = client.post(f"{API}/audit/decisions/{d0['decision_id']}/review", json={"reviewer": "qa", "outcome": "override", "final_action": "bogus", "reason": "test"})
    assert bad.status_code == 422
    ok = client.post(f"{API}/audit/decisions/{d0['decision_id']}/review", json={"reviewer": "qa analyst", "outcome": "reject", "reason": "customer on hardship plan"})
    assert ok.status_code == 200 and ok.json()["final_action"] == "none"
    after = client.get(f"{API}/audit/decisions", params={"dataset_id": did, "customer_id": d0["customer_id"]}).json()["items"][0]
    assert after["status"] == "rejected" and after["recommended_action"] == d0["recommended_action"], "original recommendation must be preserved"
    assert client.get(f"{API}/audit/log", params={"event": "decision_reviewed"}).json()["total"] >= 1
    assert "decision_id" in client.get(f"{API}/audit/decisions/export", params={"dataset_id": did}).text
    # explanation object
    ex = client.get(f"{API}/customers/{d0['customer_id']}/explanation", params={"dataset_id": did}).json()
    assert ex["explanations"]["predictive"]["top_drivers"] and ex["explanations"]["causal"]["assumptions"]
    assert client.get(f"{API}/customers/{d0['customer_id']}/recommendations", params={"dataset_id": did}).json()
    assert client.get(f"{API}/customers/{d0['customer_id']}/repayment-history", params={"dataset_id": did}).json()["repayments"]
    # monitoring
    mon = client.get(f"{API}/monitoring/summary", params={"dataset_id": did}).json()
    assert mon["recommended_investigation"] and len(mon["effect_stability"]) == 3
    # 14-15 report generation and retrieval
    rep = client.post(f"{API}/reports/generate", params={"dataset_id": did, "run_id": rid}).json()
    assert rep["markdown"] and rep["scenarios_included"] == 10
    got = client.get(f"{API}/reports/{rep['markdown']}")
    assert got.status_code == 200 and "not an official audit" in got.text.lower()
    if rep["pdf"]:
        assert client.get(f"{API}/reports/{rep['pdf']}").content[:4] == b"%PDF"
    assert client.get(f"{API}/optimization/runs/{rid}").json()["summary"]["n_selected"] == s["n_selected"]
    assert client.get(f"{API}/optimization/runs/{rid}/allocation", params={"page_size": 5}).json()["total"] == s["n_selected"]
    assert client.get(f"{API}/overview", params={"dataset_id": did, "run_id": rid}).json()["kpis"]["n_recommended"] == s["n_selected"]


def test_experiment_design_endpoints(client):
    r = client.post(f"{API}/experiments/design", json={"mde_absolute": -0.03, "baseline": 0.25, "compliance": 0.8, "attrition": 0.1, "name": "Reminder trial"})
    assert r.status_code == 200
    j = r.json()
    assert "not approved" in j["status"].lower() and j["sample_size"]["n_total"] > 0 and j["experiment_id"]
    q = client.post(f"{API}/assistant/parse", json={"dataset_id": 1, "question": "Compare reminders against no intervention for eligible customers with a high baseline risk"}).json()
    assert q["executed"] is False and q["proposed_configuration"]["treatment"] == "standard_reminder" and q["proposed_configuration"]["subgroup"] == "high_baseline_risk"
