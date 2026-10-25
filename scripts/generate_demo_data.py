"""Regenerate a reproducible synthetic portfolio.

    python scripts/generate_demo_data.py --name demo_obs --size medium --scenario observational --seed 42 [--train]
"""
import argparse

import _bootstrap  # noqa: F401
from app.core.config import settings
from app.data.synthetic import SCENARIOS, SIZE_PRESETS, GenConfig, generate, save
from app.data.validation import validate_tables

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--name", default="demo_obs")
ap.add_argument("--size", choices=list(SIZE_PRESETS), default=None, help="small=8k, medium=40k, large=200k customers")
ap.add_argument("--n", type=int, default=None, help="explicit customer count (overrides --size)")
ap.add_argument("--scenario", choices=SCENARIOS, default="observational")
ap.add_argument("--seed", type=int, default=settings.default_seed)
ap.add_argument("--train", action="store_true", help="also train all models (risk, uplift) and cache the analysis")
a = ap.parse_args()

n = a.n or SIZE_PRESETS.get(a.size or "", settings.default_n_customers)
cfg = GenConfig(n_customers=n, seed=a.seed, scenario=a.scenario)
tables = generate(cfg)
man = save(tables, cfg, settings.synthetic_dir, a.name)
rep = validate_tables(tables)
print(f"generated '{a.name}': {n:,} customers, scenario={a.scenario}, seed={a.seed}")
print(f"  fingerprint {man['fingerprint']}")
print(f"  stored in   {settings.synthetic_dir / a.name}")
print(f"  validation  score {rep['quality_score']:.0f}, errors: {sum(f['severity'] == 'error' for f in rep['findings'])}")
if a.train:
    from app.data import frames
    from app.services.pipeline import run_analysis
    frames.clear_cache()
    an = run_analysis(a.name, settings.default_seed, progress=lambda f, m: print(f"  [{f:4.0%}] {m}"), force=True)
    print(f"  trained model version {an.model_version} (risk model: {an.risk_bundle.model_name})")
print("Restart (or refresh) the API to register the dataset; it is picked up automatically on startup and on GET /datasets.")
