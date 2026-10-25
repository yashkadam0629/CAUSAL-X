"""Fairness *auditing* (not used for operational decisions). Group attributes are audit-only or coverage-config fields."""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_GROUP_N = 100  # groups smaller than this are reported as suppressed (no misleading comparisons)


def wilson(k: int, n: int, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def audit(frame: pd.DataFrame, risk: np.ndarray, selected: np.ndarray, arr_best: np.ndarray, group_col: str, outcome="delinquency_30d") -> dict:
    """`frame` must be the decision cohort; `selected` boolean per row; `arr_best` best-action estimated ARR."""
    rows = []
    for g, d in frame.groupby(group_col):
        idx = frame.index.get_indexer(d.index)
        n = len(d)
        if n < MIN_GROUP_N:
            rows.append({"group": str(g), "n": int(n), "suppressed": True, "reason": f"n<{MIN_GROUP_N}"})
            continue
        sel = selected[idx]
        y = d[outcome].values
        rr = risk[idx]
        rows.append({
            "group": str(g), "n": int(n), "suppressed": False, "selection_rate": float(sel.mean()),
            "selection_rate_ci": wilson(int(sel.sum()), n), "observed_outcome_rate": float(y.mean()),
            "outcome_rate_ci": wilson(int(y.sum()), n), "mean_predicted_risk": float(rr.mean()),
            "calibration_gap": float(rr.mean() - y.mean()), "mean_estimated_benefit": float(arr_best[idx].mean()),
        })
    return {"group_attribute": group_col, "min_group_n": MIN_GROUP_N, "groups": rows,
            "caveats": ["Differences are descriptive and are not proof of discrimination or of fairness.",
                        "Fairness criteria (parity, calibration, error balance) can conflict; the right choice is a policy decision.",
                        "Attributes used here are excluded from model features and optimiser inputs."]}
