"""Validate a stored synthetic dataset (schema, referential integrity, temporal logic, leakage). Exit code 1 on errors.

    python scripts/validate_dataset.py demo_obs
"""
import json
import sys

import _bootstrap  # noqa: F401
from app.data.frames import load_tables
from app.data.validation import validate_tables

name = sys.argv[1] if len(sys.argv) > 1 else "demo_obs"
rep = validate_tables(load_tables(name))
for f in rep["findings"]:
    print(f"[{f['severity'].upper():7}] {f['message']}")
print(f"quality score: {rep['quality_score']:.0f}   rows: {json.dumps(rep['rows'])}")
sys.exit(1 if any(f["severity"] == "error" for f in rep["findings"]) else 0)
