"""Dataset validation, profiling, leakage warnings and capability detection (never invents missing treatments/outcomes)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

LEAK_NAME = re.compile(r"(default|delinq|outcome|recover|charge.?off|after|post|future|collect|npa|arrears|dpd)", re.I)
ROLES = ["customer_id", "decision_date", "treatment", "outcome", "outcome_date", "cost"]


def profile(df: pd.DataFrame) -> list[dict]:
    rows = []
    for c in df.columns:
        s = df[c]
        r = {"column": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()), "missing_pct": float(s.isna().mean()),
             "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            r.update(min=float(s.min()) if s.notna().any() else None, max=float(s.max()) if s.notna().any() else None,
                     mean=float(s.mean()) if s.notna().any() else None)
        else:
            r["top_values"] = s.value_counts().head(5).to_dict()
        rows.append(r)
    return rows


def score_from(findings: list[dict]) -> float:
    pen = {"error": 15, "warning": 5, "info": 0}
    return float(max(0, 100 - sum(pen[f["severity"]] for f in findings)))


def f(sev, code, msg, **kw):
    return {"severity": sev, "code": code, "message": msg, **kw}


def inspect_upload(df: pd.DataFrame) -> dict:
    """Guess a column mapping from names so the user starts from a sensible suggestion."""
    guess = {}
    low = {c.lower(): c for c in df.columns}
    for role, names in {"customer_id": ["customer_id", "cust_id", "id", "customerid"], "decision_date": ["decision_date", "snapshot_date", "date", "assignment_date"],
                        "treatment": ["treatment", "intervention", "intervention_type", "arm", "action"],
                        "outcome": ["outcome", "default", "delinquency_30d", "y", "target", "missed_payment"],
                        "outcome_date": ["outcome_date", "outcome_window_end"], "cost": ["cost", "intervention_cost"]}.items():
        for n in names:
            if n in low:
                guess[role] = low[n]; break
    return {"n_rows": int(len(df)), "n_columns": int(len(df.columns)), "columns": profile(df), "suggested_mapping": guess,
            "preview": df.head(5).astype(str).to_dict("records")}


def validate_upload(df: pd.DataFrame, mapping: dict, covariates: list[str] | None = None, control_label: str | None = None) -> dict:
    findings, excl = [], []
    n0 = len(df)
    m = {k: v for k, v in mapping.items() if v}
    miss_cols = [v for v in m.values() if v not in df.columns]
    if miss_cols:
        raise ValueError(f"mapped columns not found in file: {miss_cols}")
    if "customer_id" not in m:
        findings.append(f("error", "no_customer_id", "No customer_id column mapped; duplicates and linkage cannot be checked."))
    d = df.copy()
    if "customer_id" in m:
        dup = int(d.duplicated(m["customer_id"]).sum())
        if dup:
            findings.append(f("warning", "duplicates", f"{dup} duplicate customer_id rows (first kept).", count=dup))
            excl.append({"reason": "duplicate customer_id", "rows": dup})
            d = d.drop_duplicates(m["customer_id"])
    if "decision_date" in m:
        dt = pd.to_datetime(d[m["decision_date"]], errors="coerce")
        bad = int(dt.isna().sum())
        if bad:
            findings.append(f("warning", "bad_dates", f"{bad} rows have unparseable decision dates (excluded).", count=bad))
            excl.append({"reason": "unparseable decision_date", "rows": bad})
        d = d.assign(_decision_date=dt)[dt.notna()]
    else:
        findings.append(f("warning", "no_time_zero", "No decision date mapped: temporal split, leakage and time-zero checks are impossible."))
    if "outcome" in m:
        y = pd.to_numeric(d[m["outcome"]], errors="coerce")
        bad = int(y.isna().sum())
        if bad:
            findings.append(f("warning", "bad_outcome", f"{bad} rows with missing/non-numeric outcome (excluded, not imputed).", count=bad))
            excl.append({"reason": "missing or non-numeric outcome", "rows": bad})
        d = d[y.notna()].assign(_y=y[y.notna()])
        if set(d["_y"].unique()) - {0, 1}:
            findings.append(f("info", "continuous_outcome", "Outcome is not binary: continuous-outcome estimators will be used."))
    else:
        findings.append(f("warning", "no_outcome", "No outcome column mapped: only profiling and treatment-assignment checks are possible."))
    levels = {}
    if "treatment" in m:
        t = d[m["treatment"]].astype(str)
        levels = t.value_counts().to_dict()
        if len(levels) < 2:
            findings.append(f("error", "single_treatment_level", "Treatment column has one level: no contrast can be estimated."))
        small = {k: v for k, v in levels.items() if v < 30}
        if small:
            findings.append(f("warning", "small_arms", f"Arms with fewer than 30 rows cannot be analysed: {small}"))
        if control_label and control_label not in levels:
            findings.append(f("error", "control_missing", f"Control label '{control_label}' not found in treatment column."))
    else:
        findings.append(f("warning", "no_treatment", "No treatment column: causal analyses are NOT possible (predictive risk modelling only). "
                                                      "Treatment assignments are never inferred."))
    if "outcome_date" in m and "decision_date" in m:
        od = pd.to_datetime(d[m["outcome_date"]], errors="coerce")
        viol = int((od <= d["_decision_date"]).sum())
        if viol:
            findings.append(f("error", "outcome_before_time_zero", f"{viol} rows have an outcome date on/before the decision date (outcome-window violation).", count=viol))
    covs = covariates or [c for c in df.columns if c not in m.values()]
    num_fail = []
    for c in covs:
        if c in d and not pd.api.types.is_numeric_dtype(d[c]):
            conv = pd.to_numeric(d[c], errors="coerce")
            if conv.notna().mean() > 0.9:
                num_fail.append((c, int(conv.isna().sum() - d[c].isna().sum())))
    for c, k in num_fail:
        if k:
            findings.append(f("warning", "numeric_conversion", f"Column {c}: {k} non-numeric values in a mostly numeric column."))
    high_miss = [c for c in covs if c in d and d[c].isna().mean() > 0.3]
    if high_miss:
        findings.append(f("warning", "high_missingness", f">30% missing in: {', '.join(high_miss)}"))
    # leakage warnings
    if "outcome" in m:
        for c in covs:
            if c in d and LEAK_NAME.search(c):
                findings.append(f("warning", "leakage_name", f"Column '{c}' has a name suggesting it may be recorded after time zero or derived from the outcome. Verify before using as a covariate."))
        from sklearn.metrics import roc_auc_score
        yb = d["_y"].values
        if set(np.unique(yb)) <= {0, 1} and 0 < yb.mean() < 1:
            for c in covs:
                if c in d and pd.api.types.is_numeric_dtype(d[c]) and d[c].notna().sum() > 100 and d[c].nunique() > 2:
                    x = d[c].fillna(d[c].median()).values
                    auc = roc_auc_score(yb, x)
                    if max(auc, 1 - auc) > 0.95:
                        findings.append(f("warning", "leakage_signal", f"Column '{c}' alone separates the outcome with AUC={max(auc, 1 - auc):.2f}: possible target leakage."))
    caps = capabilities(d, m, levels)
    return {"rows_in": int(n0), "rows_analysable": int(len(d)), "exclusions": excl, "findings": findings, "quality_score": score_from(findings),
            "capabilities": caps, "profile": profile(df), "treatment_levels": levels,
            "note": "Rows are never silently discarded: every exclusion is listed with its count and reason."}


def capabilities(d: pd.DataFrame, m: dict, levels: dict) -> dict:
    has_y, has_t = "outcome" in m, "treatment" in m
    arms_ok = sum(v >= 30 for v in levels.values()) >= 2
    y_ok = has_y and "_y" in d and d["_y"].nunique() >= 2
    return {
        "profiling": {"supported": True},
        "predictive_risk": {"supported": bool(y_ok and len(d) >= 200), "needs": "outcome with both classes and >=200 rows"},
        "causal_effect_estimation": {"supported": bool(has_t and y_ok and arms_ok),
                                     "needs": "treatment column with >=2 arms (>=30 rows each) and an outcome; assignment mechanism must be documented"},
        "uplift_and_optimisation": {"supported": False,
                                    "needs": "standard CAUSAL-X schema (eligibility rules, costs, pre-treatment features); available for generated datasets"},
        "policy_evaluation": {"supported": bool(has_t and y_ok and arms_ok),
                              "needs": "logged assignment probabilities (randomised) or credible estimation of them"},
    }


def validate_tables(t: dict[str, pd.DataFrame]) -> dict:
    """Full validation of the standard multi-table schema (used for generated datasets and on demand)."""
    c, l, r, i, o = (t[k] for k in ("customers", "loans", "repayments", "interventions", "outcomes"))
    findings = []
    for name, df, key in (("customers", c, "customer_id"), ("loans", l, "loan_id"), ("outcomes", o, "outcome_id"), ("interventions", i, "intervention_id")):
        dup = int(df[key].duplicated().sum())
        findings.append(f("error" if dup else "info", f"pk_{name}", f"{name}: {dup} duplicate {key}" if dup else f"{name}: primary key unique", count=dup))
    for name, df, col, ref in (("loans", l, "customer_id", c), ("interventions", i, "customer_id", c), ("outcomes", o, "customer_id", c),
                               ("interventions", i, "loan_id", l), ("outcomes", o, "loan_id", l), ("repayments", r, "loan_id", l)):
        orphans = int((~df[col].isin(ref[col])).sum())
        findings.append(f("error" if orphans else "info", f"fk_{name}_{col}", f"{name}.{col}: {orphans} orphan rows" if orphans else f"{name}.{col}: referential integrity OK", count=orphans))
    dd = pd.to_datetime(c.set_index("customer_id")["data_snapshot_date"])
    ow = o.assign(ws=pd.to_datetime(o["outcome_window_start"]), we=pd.to_datetime(o["outcome_window_end"]))
    viol = int((ow["ws"] <= ow["customer_id"].map(dd)).sum())
    findings.append(f("error" if viol else "info", "outcome_window", f"{viol} outcomes start on/before the decision date" if viol else "Outcome windows start after time zero", count=viol))
    rep_late = int((pd.to_datetime(r["scheduled_due_date"]) >= pd.to_datetime(r["loan_id"].map(l.set_index("loan_id")["contractual_due_date"]))).sum())
    findings.append(f("error" if rep_late else "info", "repayment_history_timing", f"{rep_late} history rows scheduled at/after the next due date" if rep_late else "Repayment history precedes time zero", count=rep_late))
    asg = pd.to_datetime(i["assignment_date"]); dsn = pd.to_datetime(i["customer_id"].map(dd))
    findings.append(f("error" if (asg != dsn).any() else "info", "assignment_time_zero", "Assignment date equals decision date" if not (asg != dsn).any() else "Assignment date differs from decision date"))
    miss = c.isna().mean()
    heavy = miss[miss > 0.0]
    findings.append(f("info", "missingness", "Missing values (handled by training-only median/mode imputation): " + (", ".join(f"{k} {v:.1%}" for k, v in heavy.items()) or "none")))
    findings.append(f("info", "range_checks", "Ranges OK" if (c["debt_to_income_ratio"] >= 0).all() and (l["interest_rate"] > 0).all() else "Range violations"))
    if "assignment_probability" in i:
        known = i["assignment_probability"].notna().mean()
        findings.append(f("info", "logged_propensities", f"Logged assignment probabilities present for {known:.0%} of rows (full coverage suggests a randomised design)."))
    from .schema import FEATURES, OUTCOMES
    leak = sorted(set(FEATURES) & set(OUTCOMES))
    findings.append(f("error" if leak else "info", "feature_leakage", f"Outcome columns among features: {leak}" if leak else "No outcome-derived columns in the feature set"))
    return {"findings": findings, "quality_score": score_from(findings), "rows": {k: int(len(v)) for k, v in t.items()},
            "capabilities": {"profiling": {"supported": True}, "predictive_risk": {"supported": True}, "causal_effect_estimation": {"supported": True},
                             "uplift_and_optimisation": {"supported": True}, "policy_evaluation": {"supported": True}}}
