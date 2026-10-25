"""End-to-end demonstration without the API: data -> models -> causal -> optimisation -> policy evaluation.

Runs the three assignment scenarios and prints how naive and adjusted estimates compare with the synthetic ground truth.

    python scripts/run_demo_pipeline.py [--n 20000] [--seed 42]
Results are written to artifacts/experiment_results/demo_pipeline.json.
"""
import argparse
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
from app.causal.estimators import CausalSpec, estimate, prepare
from app.core.config import settings
from app.data import frames
from app.data.synthetic import SCENARIOS, GenConfig, generate, save
from app.services import decisions as dec
from app.services.pipeline import run_analysis

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seed", type=int, default=settings.default_seed)
a = ap.parse_args()

results = {}
for sc in SCENARIOS:
    t0 = time.time()
    name = f"pipeline_{sc}"
    cfg = GenConfig(n_customers=a.n, seed=a.seed, scenario=sc)
    tables = generate(cfg)
    save(tables, cfg, settings.synthetic_dir, name)
    frames.clear_cache()
    an = run_analysis(name, a.seed, force=True)
    fr, truth = an.frame, tables["ground_truth"].set_index("customer_id")
    print(f"\n=== scenario: {sc}  (n={a.n:,}, model {an.model_version}, risk model {an.risk_bundle.model_name}, test AUC {an.risk_bundle.metrics['test']['roc_auc']:.3f}) ===")
    print(f"{'treatment':20} {'truth(ITT)':>11} {'naive diff':>11} {'regression':>11} {'IPW':>9} {'AIPW [95% CI]':>26}")
    rows = {}
    for arm in ("standard_reminder", "counselling", "repayment_plan"):
        d, _, _ = prepare(fr, CausalSpec(treatment=arm))
        t = truth.loc[d.customer_id, f"itt_effect_{arm}"].mean()
        e = {k: estimate(fr, CausalSpec(treatment=arm, estimator=k, bootstrap=40, seed=a.seed)) for k in ("difference_in_means", "regression_adjustment", "ipw", "aipw")}
        ai = e["aipw"]
        print(f"{arm:20} {t:11.4f} {e['difference_in_means']['ate']:11.4f} {e['regression_adjustment']['ate']:11.4f} {e['ipw']['ate']:9.4f} {ai['ate']:9.4f} [{ai['ci_low']:.3f}, {ai['ci_high']:.3f}]")
        rows[arm] = {"truth": t, **{k: v["ate"] for k, v in e.items()}, "aipw_ci": [ai["ci_low"], ai["ci_high"]]}
    res, p, d, si, sa, allocs = dec.run_optimization(an, None, None, True, a.seed, 30)
    print("\nstrategy comparison (net benefit; model-estimated vs synthetic truth):")
    for s in res["strategies"]:
        tr = s.get("synthetic_truth", {}).get("net_benefit")
        print(f"  {s['strategy'][:56]:56} est {s['net_benefit']:>10,.0f}   truth {tr if tr is None else format(tr, ',.0f'):>10}   n={s['n_selected']:.0f}")
    ev = dec.evaluate_strategies(an, None, None, a.seed)
    print("off-policy (doubly robust) events prevented vs truth:")
    for p_ in ev["policies"]:
        if p_["strategy"] in ("optimised", "Highest-risk-first (most intensive eligible action)"):
            print(f"  {p_['strategy'][:56]:56} DR {p_['dr_estimated_events_prevented']:7.1f}  truth {p_['synthetic_truth_events_prevented']:7.1f}")
    results[sc] = {"effects": rows, "strategies": [{k: s[k] for k in ("strategy", "n_selected", "budget_used", "net_benefit")} | {"truth_net": s.get("synthetic_truth", {}).get("net_benefit")} for s in res["strategies"]],
                   "seconds": round(time.time() - t0, 1)}
out = settings.artifacts_dir / "experiment_results" / "demo_pipeline.json"
out.write_text(json.dumps(results, indent=2, default=float))
print(f"\nwritten {out}")
