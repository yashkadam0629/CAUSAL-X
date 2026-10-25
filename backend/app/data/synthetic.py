"""Synthetic loan-portfolio generator with an explicit data-generating process (DGP).

Latent variables (never model inputs):
  s  financial stress           f  forgetfulness           u  hidden income shock
Potential outcomes are coupled through one uniform draw per customer, Y(a) = 1[U < p_a(x)],
so the individual-level counterfactuals are internally consistent.

Scenarios
  randomized            assignment random among eligible arms (known propensities)
  observational         assignment depends on *observed* risk features (confounding is adjustable)
  observational_hidden  as above plus dependence on hidden u (unmeasured confounding)

The ground-truth table is stored separately and is only used for benchmark evaluation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ARMS, DEFAULT_COSTS, TREATMENTS

SCENARIOS = ("randomized", "observational", "observational_hidden")
START = date(2025, 1, 1)
N_COHORTS = 12
DELIVERY_PROB = {"standard_reminder": 0.92, "counselling": 0.70, "repayment_plan": 0.85}
RANDOM_ARM_WEIGHTS = {"none": 0.31, "standard_reminder": 0.23, "counselling": 0.23, "repayment_plan": 0.23}
SIZE_PRESETS = {"small": 8_000, "medium": 40_000, "large": 200_000}


@dataclass
class GenConfig:
    n_customers: int = 20_000
    seed: int = 42
    scenario: str = "observational"
    base_intercept: float = -1.55
    confounding_strength: float = 1.0  # scales dependence of assignment on observed risk
    hidden_confounding_strength: float = 0.9  # only used when scenario == observational_hidden
    effect_scale: float = 1.0  # multiplies all logit-scale treatment effects
    econ_shift: float = 0.0  # additive shift to the economic stress level


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def generate(cfg: GenConfig) -> dict[str, pd.DataFrame]:
    if cfg.scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_customers
    ids = np.arange(1, n + 1)
    cust_id = np.array([f"SYN-CUST-{i:07d}" for i in ids])
    loan_id = np.array([f"SYN-LOAN-{i:07d}" for i in ids])

    # ---------------- latent traits ----------------
    f = rng.normal(size=n)  # forgetfulness
    u = rng.normal(size=n)  # hidden income shock
    region = rng.choice(["R1", "R2", "R3", "R4", "R5"], size=n, p=[0.28, 0.24, 0.2, 0.16, 0.12])
    age_band = rng.choice(["18-29", "30-44", "45-59", "60+"], size=n, p=[0.22, 0.40, 0.27, 0.11])
    employment = rng.choice(["salaried", "self_employed", "gig_worker", "other"], size=n, p=[0.55, 0.2, 0.15, 0.1])

    # ---------------- customers ----------------
    emp_inc = pd.Series(employment).map({"salaried": 0.15, "self_employed": 0.0, "gig_worker": -0.25, "other": -0.1}).values
    monthly_income = np.exp(10.3 + 0.55 * rng.normal(size=n) + emp_inc)
    expense_ratio = np.clip(0.45 + 0.12 * rng.normal(size=n) - 0.04 * _z(np.log(monthly_income)), 0.2, 0.95)
    loan_type = rng.choice(["personal", "auto", "credit_line", "small_business"], size=n, p=[0.42, 0.25, 0.23, 0.10])

    principal = np.exp(np.where(loan_type == "auto", 12.6, np.where(loan_type == "small_business", 12.9,
                              np.where(loan_type == "credit_line", 11.2, 11.9))) + 0.4 * rng.normal(size=n))
    term = np.where(loan_type == "auto", rng.choice([36, 48, 60], size=n),
                    np.where(loan_type == "credit_line", rng.choice([12, 24], size=n), rng.choice([12, 24, 36, 48], size=n)))
    elapsed = (rng.uniform(0.05, 0.85, size=n) * term).astype(int)
    elapsed = np.minimum(elapsed, term - 3)
    remaining = term - elapsed
    existing_debt_monthly = monthly_income * np.clip(0.12 + 0.10 * rng.normal(size=n) + 0.25 * np.abs(rng.normal(size=n)) * 0.5, 0.0, 0.7)

    stress0 = 0.5 * rng.normal(size=n)
    base_rate = np.select([loan_type == "auto", loan_type == "credit_line", loan_type == "small_business"], [0.085, 0.19, 0.14], 0.125)
    rate = np.clip(base_rate + 0.01 * stress0 + 0.005 * rng.normal(size=n), 0.05, 0.30)
    r = rate / 12
    installment = principal * r / (1 - (1 + r) ** (-term))
    outstanding = principal * (1 - ((1 + r) ** elapsed - 1) / ((1 + r) ** term - 1))
    dti = (existing_debt_monthly + installment) / monthly_income

    # latent financial stress s
    s = 0.7 * _z(dti) + 0.4 * _z(expense_ratio) - 0.3 * _z(np.log(monthly_income)) + 0.55 * rng.normal(size=n)
    s = _z(s)
    stress_indicator = sigmoid(1.2 * s + 0.6 * rng.normal(size=n))  # noisy observed proxy in (0, 1)

    # ---------------- history (pre-time-zero) ----------------
    p_late = sigmoid(-2.1 + 0.9 * f + 0.7 * s)
    n_hist = 24
    late_count = rng.binomial(n_hist, p_late)
    on_time_rate = 1 - late_count / n_hist
    hist_delinq = rng.poisson(0.15 * np.exp(0.8 * s))
    credit_len = np.clip(rng.gamma(4.0, 20.0, size=n), 6, 400)
    tenure = np.clip(rng.gamma(3.0, 12.0, size=n), 1, 240)
    recent_late = rng.random(n) < sigmoid(-1.5 + 0.8 * f + 0.6 * s + 0.9 * u)  # negative-control outcome candidate
    prior_interv = rng.poisson(0.3 + 0.5 * sigmoid(s))

    # ---------------- cohorts and macro ----------------
    cohort = rng.integers(0, N_COHORTS, size=n)
    dec_date = np.array([START + timedelta(days=int(30.4 * c) + int(rng.integers(0, 28))) for c in cohort])
    regions = ["R1", "R2", "R3", "R4", "R5"]
    reg_off = dict(zip(regions, [0.0, 0.25, -0.2, 0.4, 0.1]))
    econ_rows = []
    level = np.zeros(len(regions))
    for m in range(N_COHORTS):
        # mild AR(1) with deterioration in the final quarters (documented synthetic drift)
        shock = rng.normal(scale=0.12, size=len(regions))
        level = 0.7 * level + shock + (0.18 if m >= 8 else 0.0)
        for j, rg in enumerate(regions):
            ec = 0.0 + level[j] + reg_off[rg] + cfg.econ_shift
            econ_rows.append({
                "indicator_date": START + timedelta(days=int(30.4 * m)),
                "regional_economic_group": rg,
                "unemployment_proxy": round(5.5 + 1.2 * ec + rng.normal(scale=0.05), 3),
                "inflation_proxy": round(4.5 + 0.6 * ec + rng.normal(scale=0.05), 3),
                "interest_rate_proxy": round(6.5 + 0.3 * m / 12 + rng.normal(scale=0.03), 3),
                "economic_stress_index": round(ec, 4),
                "cohort_month": m,
                "source": "synthetic (not official statistics)",
            })
    econ = pd.DataFrame(econ_rows)
    econ_key = econ.set_index(["cohort_month", "regional_economic_group"])["economic_stress_index"]
    econ_val = np.array([econ_key[(int(c), rg)] for c, rg in zip(cohort, region)])
    econ_z = econ_val / 0.3

    # ---------------- baseline risk p0 and treatment effects ----------------
    dti_z = _z(dti)
    lp0 = (cfg.base_intercept + 0.85 * s + 0.50 * f + 0.35 * econ_z + 0.60 * u - 0.25 * _z(np.log(tenure))
           + np.select([loan_type == "credit_line", loan_type == "small_business"], [0.3, 0.4], 0.0)
           + 0.10 * hist_delinq)
    p0 = sigmoid(lp0)
    es = cfg.effect_scale
    d_rem = es * (-1.1 * sigmoid(1.8 * f - 1.0 * s - 0.2) + 0.12)
    d_cou = es * (-1.2 * sigmoid(2.0 * (s - 0.3)) * (1 - 0.3 * sigmoid(econ_z)) + 0.05)
    d_plan = es * (-1.7 * sigmoid(2.5 * (dti_z - 0.2)) * sigmoid(2.0 * s) + 0.55 * sigmoid(-2.5 * s))
    p = {"none": p0, "standard_reminder": sigmoid(lp0 + d_rem), "counselling": sigmoid(lp0 + d_cou),
         "repayment_plan": sigmoid(lp0 + d_plan)}

    # ---------------- eligibility (rule based, observed fields only) ----------------
    contactable = rng.random(n) < 0.97
    elig = {
        "none": np.ones(n, bool),
        "standard_reminder": contactable,
        "counselling": contactable & ((hist_delinq >= 1) | (stress_indicator > 0.5) | (dti > 0.40)),
        "repayment_plan": (loan_type != "credit_line") & (remaining >= 6) & ((dti >= 0.35) | (hist_delinq >= 2)),
    }

    # ---------------- assignment ----------------
    if cfg.scenario == "randomized":
        w = np.column_stack([np.where(elig[a], RANDOM_ARM_WEIGHTS[a], 0.0) for a in ARMS])
        prop = w / w.sum(axis=1, keepdims=True)
    else:
        risk_obs = (0.9 * _z(stress_indicator) + 0.6 * dti_z + 0.35 * hist_delinq + 0.5 * (1 - on_time_rate) * 4)
        hid = cfg.hidden_confounding_strength * u if cfg.scenario == "observational_hidden" else 0.0
        c = cfg.confounding_strength
        logits = {
            "none": np.zeros(n),
            "standard_reminder": -0.2 + 0.55 * c * risk_obs + 0.5 * hid,
            "counselling": -1.2 + 0.75 * c * _z(stress_indicator) + 0.4 * c * hist_delinq + 0.6 * hid,
            "repayment_plan": -1.6 + 0.85 * c * dti_z + 0.5 * c * hist_delinq + 0.6 * hid,
        }
        L = np.column_stack([np.where(elig[a], logits[a], -np.inf) for a in ARMS])
        L = L - L.max(axis=1, keepdims=True)
        e = np.exp(L)
        prop = e / e.sum(axis=1, keepdims=True)
    cum = prop.cumsum(axis=1)
    draw = rng.random(n)[:, None]
    arm_idx = (draw > cum).sum(axis=1)
    arm_idx = np.minimum(arm_idx, len(ARMS) - 1)
    assigned = np.array(ARMS)[arm_idx]

    # ---------------- delivery (assignment != receipt) and outcomes ----------------
    delivered = np.zeros(n, bool)
    for a in TREATMENTS:
        m = assigned == a
        delivered[m] = rng.random(m.sum()) < DELIVERY_PROB[a]
    U = rng.random(n)
    y_pot = {a: (U < p[a]).astype(int) for a in ARMS}
    y30 = np.where(delivered, np.select([assigned == a for a in ARMS], [y_pot[a] for a in ARMS]), y_pot["none"])
    cured = rng.random(n) < 0.15 * (1 + p0)
    missed = ((y30 == 1) | cured).astype(int)
    d60 = ((y30 == 1) & (rng.random(n) < sigmoid(0.2 + 0.6 * s))).astype(int)
    dflt = ((d60 == 1) & (rng.random(n) < 0.35)).astype(int)
    repay_amt = np.where(y30 == 1, installment * rng.uniform(0, 0.4, size=n), installment)
    recovered = np.where(dflt == 1, outstanding * rng.uniform(0.1, 0.45, size=n), 0.0)

    # ---------------- assemble tables ----------------
    miss_inc = rng.random(n) < 0.03
    miss_cl = rng.random(n) < 0.02
    due_date = np.array([d + timedelta(days=int(rng.integers(5, 20))) for d in dec_date])
    customers = pd.DataFrame({
        "customer_id": cust_id, "age_band": age_band, "employment_type": employment,
        "income_band": pd.cut(monthly_income, [0, 20000, 40000, 80000, np.inf], labels=["low", "lower_mid", "upper_mid", "high"]).astype(str),
        "monthly_income": np.where(miss_inc, np.nan, monthly_income.round(2)),
        "estimated_monthly_expenses": (monthly_income * expense_ratio).round(2),
        "existing_debt": (existing_debt_monthly * 18).round(2),
        "debt_to_income_ratio": dti.round(4), "credit_history_length": np.where(miss_cl, np.nan, credit_len.round(1)),
        "historical_delinquency_count": hist_delinq, "historical_on_time_payment_rate": on_time_rate.round(4),
        "account_tenure_months": tenure.round(1), "region_group": region,
        "customer_segment": np.select([(dti > 0.5) | (hist_delinq >= 2), (dti > 0.35) | (stress_indicator > 0.6)], ["stressed", "watchlist"], "stable"),
        "financial_stress_indicator": stress_indicator.round(4), "recent_late_flag": recent_late.astype(int),
        "prior_intervention_count": prior_interv, "cohort_month": cohort, "data_snapshot_date": dec_date,
    })
    loans = pd.DataFrame({
        "loan_id": loan_id, "customer_id": cust_id, "loan_type": loan_type, "principal_amount": principal.round(2),
        "outstanding_balance": outstanding.round(2), "interest_rate": rate.round(4), "loan_term_months": term,
        "remaining_term_months": remaining, "installment_amount": installment.round(2),
        "origination_date": [d - timedelta(days=int(30.4 * e)) for d, e in zip(dec_date, elapsed)],
        "contractual_due_date": due_date, "repayment_frequency": "monthly", "loan_status": "active",
        "eligibility_status": np.where(elig["standard_reminder"], "contactable", "not_contactable"),
    })
    # interventions: one assignment record per customer-decision (including 'none'), distinct from delivery
    base_cost = pd.Series(assigned).map(DEFAULT_COSTS).values
    cost = np.where(assigned == "none", 0.0, base_cost * np.exp(0.1 * rng.normal(size=n))).round(2)
    obs_prop_known = cfg.scenario == "randomized"
    assigned_prob = prop[np.arange(n), arm_idx]
    interventions = pd.DataFrame({
        "intervention_id": [f"SYN-INT-{i:07d}" for i in ids], "customer_id": cust_id, "loan_id": loan_id,
        "intervention_type": assigned, "assignment_date": dec_date,
        "delivery_date": [d + timedelta(days=int(rng.integers(0, 4))) if (dl and a != "none") else pd.NaT
                          for d, dl, a in zip(dec_date, delivered, assigned)],
        "intervention_cost": cost, "assignment_status": "assigned",
        "delivery_status": np.where(assigned == "none", "not_applicable", np.where(delivered, "delivered", "not_delivered")),
        "eligibility_at_assignment": True,
        "assignment_probability": np.where(obs_prop_known, assigned_prob, np.nan),
        "agent_or_channel_group": np.select([assigned == "standard_reminder", assigned == "none"], ["sms_email_channel", "none"],
                                            np.char.add("agent_pool_", rng.choice(["A", "B", "C"], size=n))),
        "cohort_month": cohort,
    })
    win_end = np.array([d + timedelta(days=30) for d in dec_date])
    outcomes = pd.DataFrame({
        "outcome_id": [f"SYN-OUT-{i:07d}" for i in ids], "customer_id": cust_id, "loan_id": loan_id,
        "outcome_window_start": [d + timedelta(days=1) for d in dec_date], "outcome_window_end": win_end,
        "outcome_type": "delinquency_30d", "delinquency_30d": y30, "missed_payment": missed,
        "delinquency_60d": d60, "default_event": dflt, "repayment_amount": repay_amt.round(2),
        "recovered_amount": recovered.round(2), "observation_complete": True,
        "outcome_recorded_at": [w + timedelta(days=3) for w in win_end],
    })
    # six prior instalments per loan (strictly before time zero)
    k = np.arange(1, 7)
    rep_loan = np.repeat(loan_id, 6)
    sched = np.array([d - timedelta(days=int(30 * kk)) for d in dec_date for kk in k])
    late_flag = rng.random(n * 6) < np.repeat(p_late, 6)
    dpd = np.where(late_flag, rng.integers(1, 45, size=n * 6), 0)
    inst6 = np.repeat(installment, 6)
    repayments = pd.DataFrame({
        "repayment_id": [f"SYN-REP-{i:08d}" for i in range(n * 6)], "loan_id": rep_loan, "scheduled_due_date": sched,
        "scheduled_amount": inst6.round(2), "paid_amount": np.where(dpd > 60, 0, inst6).round(2),
        "actual_payment_date": [s_ + timedelta(days=int(dd)) for s_, dd in zip(sched, dpd)],
        "days_past_due": dpd, "repayment_status": np.where(dpd == 0, "on_time", np.where(dpd < 30, "late", "delinquent")),
        "recorded_at": [s_ + timedelta(days=int(dd) + 1) for s_, dd in zip(sched, dpd)],
    })
    truth = pd.DataFrame({
        "customer_id": cust_id, "stress_latent": s, "forgetful_latent": f, "hidden_u": u,
        **{f"p_{a}": p[a] for a in ARMS},
        **{f"itt_effect_{a}": DELIVERY_PROB[a] * (p[a] - p0) for a in TREATMENTS},
        **{f"true_propensity_{a}": prop[:, j] for j, a in enumerate(ARMS)},
        **{f"eligible_{a}": elig[a] for a in ARMS},
    })
    return {"customers": customers, "loans": loans, "repayments": repayments, "interventions": interventions,
            "outcomes": outcomes, "economic_indicators": econ, "ground_truth": truth}


def fingerprint(tables: dict[str, pd.DataFrame]) -> str:
    h = hashlib.sha256()
    for name in sorted(tables):
        df = tables[name]
        h.update(name.encode())
        h.update(pd.util.hash_pandas_object(df.astype(str), index=False).values.tobytes())
    return h.hexdigest()


def save(tables: dict[str, pd.DataFrame], cfg: GenConfig, out_dir: Path, name: str) -> dict:
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    for k, df in tables.items():
        df.to_parquet(out / f"{k}.parquet", index=False)
    manifest = {
        "name": name, "source": "synthetic", "config": asdict(cfg), "fingerprint": fingerprint(tables),
        "rows": {k: int(len(v)) for k, v in tables.items()},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notice": "Synthetic data for methodology testing; not evidence of real-world intervention effectiveness.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
