"""Layer F/G support: scenario simulation, budget frontier, Monte Carlo uncertainty, stress tests, value of information.

All outputs are *simulated* quantities derived from model estimates under stated assumptions; they are not observed outcomes.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import expit, logit

from ..optimization.engine import OptConfig, Problem, compare_strategies, optimise, policy_assignments, summarize

SCENARIO_CATALOGUE = {
    "A_base": {"label": "A. Base case", "overrides": {}},
    "B_budget_cut": {"label": "B. Budget reduction (-40%)", "overrides": {"budget_mult": 0.6}},
    "C_budget_expansion": {"label": "C. Budget expansion (+100%)", "overrides": {"budget_mult": 2.0}},
    "D_economic_deterioration": {"label": "D. Economic deterioration (+0.5 logit baseline risk)", "overrides": {"econ_shift": 0.5}},
    "E_cost_increase": {"label": "E. Intervention costs +50%", "overrides": {"cost_mult": 1.5}},
    "F_weaker_effects": {"label": "F. Treatment effects 50% weaker", "overrides": {"effect_mult": 0.5}},
    "G_limited_agents": {"label": "G. Agent capacity -60%", "overrides": {"agent_mult": 0.4}},
    "H_reliable_effects_only": {"label": "H. Exclude unreliable effects (SE cap)", "overrides": {"max_se_quantile": 0.5}},
    "I_no_repayment_plan": {"label": "I. Repayment-plan review unavailable", "overrides": {"unavailable": ["repayment_plan"]}},
    "J_conservative": {"label": "J. Conservative (lower confidence bound, z=1.28)", "overrides": {"z": 1.28}},
}


def apply_overrides(p: Problem, c: OptConfig, o: dict) -> tuple[Problem, OptConfig]:
    arr, se, risk = p.arr.copy(), p.se.copy(), p.risk.copy()
    true_arr = None if p.true_arr is None else p.true_arr.copy()
    if "econ_shift" in o:
        # baseline risk rises on the logit scale; effects are assumed to scale proportionally with baseline risk
        new = expit(logit(np.clip(risk, 1e-4, 1 - 1e-4)) + o["econ_shift"])
        ratio = (new / np.clip(risk, 1e-4, None))[:, None]
        arr, se, risk = arr * ratio, se * ratio, new
        if true_arr is not None:
            true_arr = true_arr * ratio
    if "effect_mult" in o:
        arr = arr * o["effect_mult"]
        if true_arr is not None:
            true_arr = true_arr * o["effect_mult"]
    p2 = replace(p, arr=arr, se=se, risk=risk, true_arr=true_arr)
    kw = dict(c.__dict__)
    kw["budget"] = c.budget * o.get("budget_mult", 1.0)
    kw["agent_capacity"] = c.agent_capacity * o.get("agent_mult", 1.0)
    kw["channel_capacity"] = c.channel_capacity * o.get("channel_mult", 1.0)
    kw["costs"] = {a: v * o.get("cost_mult", 1.0) for a, v in c.costs.items()}
    kw["value_per_event"] = o.get("value_per_event", c.value_per_event)
    if "unavailable" in o:
        kw["available"] = [a for a in c.available if a not in o["unavailable"]]
    if "z" in o:
        kw["z"] = o["z"]
    if "max_se_quantile" in o:
        el = p.elig & (np.array([a in c.available for a in p.arms])[None, :])
        kw["max_se"] = float(np.quantile(se[el], o["max_se_quantile"])) if el.any() else None
    return p2, OptConfig(**kw)


def run_scenario(p: Problem, c: OptConfig, overrides: dict, label: str, seed=42, mc_draws=200) -> dict:
    p2, c2 = apply_overrides(p, c, overrides)
    s, si, sa = optimise(p2, c2)
    s["strategy"] = label
    rf_i, rf_a = policy_assignments("highest_risk_first_intensive", p2, c2, np.random.default_rng(seed))
    rf = summarize(p2, c2, rf_i, rf_a, "Highest-risk-first (intensive)")
    mc = monte_carlo(p2, c2, si, sa, mc_draws, seed)
    return {"label": label, "overrides": overrides, "optimised": s, "highest_risk_first": rf, "monte_carlo": mc,
            "net_benefit_advantage_over_risk_first": s["net_benefit"] - rf["net_benefit"],
            "drivers": _drivers(overrides)}


def _drivers(o: dict) -> list[str]:
    d = []
    if "budget_mult" in o: d.append(f"Budget x{o['budget_mult']}")
    if "econ_shift" in o: d.append(f"Baseline risk +{o['econ_shift']} logit (effects assumed proportional to baseline risk)")
    if "cost_mult" in o: d.append(f"Costs x{o['cost_mult']}")
    if "effect_mult" in o: d.append(f"Effects x{o['effect_mult']}")
    if "agent_mult" in o: d.append(f"Agent capacity x{o['agent_mult']}")
    if "unavailable" in o: d.append("Unavailable: " + ", ".join(o["unavailable"]))
    if "z" in o: d.append(f"Lower-confidence-bound objective, z={o['z']}")
    if "max_se_quantile" in o: d.append("Exclude estimates with SE above the median")
    return d or ["Current modelled assumptions"]


def monte_carlo(p: Problem, c: OptConfig, si, sa, draws=200, seed=42) -> dict:
    """Propagates ARR sampling uncertainty (independent normal errors) through a *fixed* allocation.
    Excludes confounding/misspecification uncertainty, error correlation and cost/value uncertainty (held fixed)."""
    if len(si) == 0:
        return {"method": "none", "draws": 0, "note": "empty allocation"}
    rng = np.random.default_rng(seed)
    mu, sd = p.arr[si, sa], p.se[si, sa]
    cost = np.array([c.costs[a] for a in p.arms])[sa].sum()
    ev = (mu[None, :] + sd[None, :] * rng.standard_normal((draws, len(mu)))).sum(axis=1)
    net = ev * c.value_per_event - cost
    return {"method": f"Monte Carlo over independent Normal(ARR, SE^2) errors, {draws} draws, seed {seed}", "draws": draws,
            "events_mean": float(ev.mean()), "events_p05_p95": [float(np.percentile(ev, 5)), float(np.percentile(ev, 95))],
            "net_benefit_mean": float(net.mean()), "net_benefit_p05_p95": [float(np.percentile(net, 5)), float(np.percentile(net, 95))],
            "prob_net_positive": float((net > 0).mean()),
            "excludes": ["unmeasured confounding", "model misspecification", "correlated estimation errors", "cost and value uncertainty"]}


def budget_frontier(p: Problem, c: OptConfig, points=14, max_mult=2.5, vary="budget") -> dict:
    mults = np.linspace(0.0, max_mult, points + 1)[1:] if vary == "budget" else np.linspace(0.0, max_mult, points + 1)
    rows = []
    for m in mults:
        cc = OptConfig(**{**c.__dict__, "budget": c.budget * m}) if vary == "budget" else OptConfig(**{**c.__dict__, "agent_capacity": c.agent_capacity * m})
        s, _, _ = optimise(p, cc)
        rows.append({"multiplier": float(m), "budget": cc.budget, "agent_capacity": cc.agent_capacity, "net_benefit": s["net_benefit"],
                     "gross_benefit": s["gross_benefit"], "events": s["expected_avoided_events"], "spend": s["budget_used"], "n_selected": s["n_selected"]})
    for k in range(len(rows)):
        prev = rows[k - 1] if k else {"net_benefit": 0.0, "budget": 0.0, "agent_capacity": 0.0}
        dx = (rows[k]["budget"] - prev["budget"]) if vary == "budget" else (rows[k]["agent_capacity"] - prev["agent_capacity"])
        rows[k]["marginal_net_benefit_per_unit"] = (rows[k]["net_benefit"] - prev["net_benefit"]) / dx if dx else None
    sat = next((r for r in rows if r["spend"] < r["budget"] * 0.995), None) if vary == "budget" else None
    return {"vary": vary, "points": rows, "diminishing_returns_note": (
        f"Spending saturates near budget {sat['budget']:.0f}: beyond it no additional pair has positive estimated net benefit." if sat else
        "Budget remains binding across the examined range.")}


def stress_test(p: Problem, c: OptConfig, si, sa, extra_allocs: dict | None = None) -> dict:
    """Evaluate fixed allocations under adverse worlds (no re-optimisation)."""
    grid = {"effects_x1.0": {}, "effects_x0.75": {"effect_mult": 0.75}, "effects_x0.5": {"effect_mult": 0.5}, "costs_x1.5": {"cost_mult": 1.5},
            "costs_x2.0": {"cost_mult": 2.0}, "baseline_risk_+0.5logit": {"econ_shift": 0.5}, "effects_x0.5_and_costs_x1.5": {"effect_mult": 0.5, "cost_mult": 1.5}}
    allocs = {"optimised": (si, sa), **(extra_allocs or {})}
    out = []
    for name, o in grid.items():
        p2, c2 = apply_overrides(p, c, o)
        row = {"stress": name}
        for lab, (i_, a_) in allocs.items():
            s = summarize(p2, c2, i_, a_, lab)
            row[f"{lab}_net_benefit"] = s["net_benefit"]
        out.append(row)
    return {"rows": out, "note": "Fixed allocations re-valued under stressed assumptions; a re-optimised policy would adapt."}


def winners_curse(p: Problem, c: OptConfig, trials=8, seed=42, z_conservative=1.28) -> dict:
    """Optimiser's-curse check: treat the estimates as the truth distribution, generate noisy re-estimates, optimise on them, and
    compare the claimed benefit with the benefit under the 'true' draw. Shows why uncertainty-aware policies matter."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(trials):
        theta = p.arr + p.se * rng.standard_normal(p.arr.shape)  # one plausible truth
        noisy = theta + p.se * rng.standard_normal(p.arr.shape)  # what an analyst would estimate from a similar sample
        for lab, z in (("unconstrained optimiser", 0.0), ("conservative optimiser", z_conservative)):
            pn = replace(p, arr=noisy)
            cc = OptConfig(**{**c.__dict__, "z": z})
            r_s, si, sa = optimise(pn, cc)
            claimed = float(noisy[si, sa].sum()) * c.value_per_event - r_s["budget_used"]
            realised = float(theta[si, sa].sum()) * c.value_per_event - r_s["budget_used"]
            rows.append({"policy": lab, "claimed": claimed, "realised": realised})
    agg = {}
    for lab in ("unconstrained optimiser", "conservative optimiser"):
        r = [x for x in rows if x["policy"] == lab]
        agg[lab] = {"mean_claimed_net_benefit": float(np.mean([x["claimed"] for x in r])),
                    "mean_realised_net_benefit": float(np.mean([x["realised"] for x in r])),
                    "mean_optimism_gap": float(np.mean([x["claimed"] - x["realised"] for x in r]))}
    return {"trials": trials, "summary": agg, "method": "Simulated re-estimation noise ~ N(0, SE^2) around plausible truths ~ N(estimate, SE^2).",
            "interpretation": "A positive optimism gap means selecting on noisy estimates overstates benefit (winner's curse)."}


def value_of_information(p: Problem, c: OptConfig, draws=200, seed=42) -> dict:
    """Expected value of perfect information about the effects, per customer, ignoring budget/capacity (an upper bound).
    EVPI_i = E[max(0, max_a V_ia)] - max(0, max_a E[V_ia]),  V_ia = value*ARR_ia - cost_a,  ARR_ia ~ N(est, SE^2) independently."""
    rng = np.random.default_rng(seed)
    cost = np.array([c.costs[a] for a in p.arms])
    el = p.elig & np.array([a in c.available for a in p.arms])[None, :]
    mu = np.where(el, c.value_per_event * p.arr - cost, -np.inf)
    sd = np.where(el, c.value_per_event * p.se, 0.0)
    best_now = np.maximum(0.0, mu.max(axis=1))
    # subsample for tractability
    n = len(mu)
    idx = rng.choice(n, size=min(n, 4000), replace=False)
    sim = np.where(np.isfinite(mu[idx])[None], mu[idx][None] + sd[idx][None] * rng.standard_normal((draws, len(idx), mu.shape[1])), -np.inf)
    perfect = np.maximum(0.0, sim.max(axis=2)).mean(axis=0)
    evpi = perfect - best_now[idx]
    scale = n / len(idx)
    return {"evpi_total_unconstrained": float(evpi.sum() * scale), "share_of_customers_with_positive_evpi": float((evpi > 1e-6).mean()),
            "evpi_per_customer_mean": float(evpi.mean()), "draws": draws, "customers_sampled": int(len(idx)),
            "interpretation": ("Upper bound on what perfect knowledge of the treatment effects could add (ignoring budget/capacity). "
                               "It is NOT the value of any specific experiment (that requires EVSI with the experiment's sample size and cost)."),
            "method": "Monte Carlo, independent Normal estimation errors"}
