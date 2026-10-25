"""Time the major analytical stages on documented dataset sizes.

    python scripts/benchmark.py --sizes 8000 40000 [200000]
Prints a Markdown table (paste into docs/testing_strategy.md). Uses temporary dataset names and removes nothing else.
"""
import argparse
import platform
import time

import _bootstrap  # noqa: F401
from app.causal.estimators import CausalSpec, estimate
from app.core.config import settings
from app.data import frames
from app.data.synthetic import GenConfig, generate, save
from app.optimization.engine import optimise
from app.services import decisions as dec
from app.services.pipeline import build_problem, run_analysis

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[8000, 40000])
a = ap.parse_args()
rows = []
for n in a.sizes:
    r = {"n": n}
    t = time.perf_counter(); cfg = GenConfig(n_customers=n, seed=42, scenario="observational"); tb = generate(cfg); r["generate"] = time.perf_counter() - t
    t = time.perf_counter(); save(tb, cfg, settings.synthetic_dir, f"bench_{n}"); r["write_parquet"] = time.perf_counter() - t
    frames.clear_cache()
    t = time.perf_counter(); an = run_analysis(f"bench_{n}", 42, force=True); r["train_all_models"] = time.perf_counter() - t
    fr = an.frame
    t = time.perf_counter(); estimate(fr, CausalSpec(treatment="standard_reminder", estimator="aipw")); r["aipw_estimate"] = time.perf_counter() - t
    p, d = build_problem(an)
    c = dec.make_config(len(d), None)
    t = time.perf_counter(); s, *_ = optimise(p, c); r["milp_solve"] = time.perf_counter() - t; r["milp_vars"] = s["n_variables"]; r["decision_customers"] = len(d)
    rows.append(r)
cols = ["n", "decision_customers", "milp_vars", "generate", "write_parquet", "train_all_models", "aipw_estimate", "milp_solve"]
print(f"Machine: {platform.processor() or platform.machine()}, Python {platform.python_version()} (seconds)")
print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
for r in rows:
    print("| " + " | ".join(f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
