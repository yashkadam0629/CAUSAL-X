"""Layers D-E: intervention decision engine and constrained optimisation (sparse MILP via SciPy/HiGHS).

Decision variable  x[i,a] in {0,1}: customer i receives treatment a (a in the three active interventions; "none" = all zero).
Benefit of (i,a)   B[i,a] = value_per_event * (ARR[i,a] - z*se[i,a])      (z=0 -> point estimate; z>0 -> conservative)
Net benefit        N[i,a] = B[i,a] - cost[a]                              (cost is subtracted exactly once, here)
Objectives         net_benefit:    max sum x*N
                   avoided_events: max sum x*(ARR - z*se)                s.t. same constraints (cost only constrains via budget)
                   min_cost:       min sum x*cost                        s.t. sum x*(ARR - z*se) >= target_events
Constraints        sum_a x[i,a] <= 1 (mutually exclusive), budget, agent capacity (slot-weighted), channel capacity,
                   per-intervention caps, total contacts, eligibility (variables for ineligible pairs do not exist),
                   minimum benefit / max uncertainty (pairs failing the filters do not exist), optional per-group coverage minima.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from ..data.schema import AGENT_SLOTS, DEFAULT_COSTS, DEFAULT_VALUE_PER_EVENT, RESOURCE, TREATMENTS

TOL = 1e-6


@dataclass
class OptConfig:
    budget: float = 100_000.0
    agent_capacity: float = 60.0  # agent "slots" (counselling=1.0, repayment_plan=1.5)
    channel_capacity: int = 10_000
    per_intervention_cap: dict = field(default_factory=dict)  # arm -> max customers
    max_contacts: int | None = None
    costs: dict = field(default_factory=lambda: {a: DEFAULT_COSTS[a] for a in TREATMENTS})
    value_per_event: float = DEFAULT_VALUE_PER_EVENT
    available: list = field(default_factory=lambda: list(TREATMENTS))
    min_arr: float = 0.0  # minimum estimated absolute risk reduction
    max_se: float | None = None  # maximum acceptable standard error of ARR
    z: float = 0.0  # >0 => conservative objective: penalise each estimate by z standard errors
    objective: str = "net_benefit"
    target_events: float | None = None  # for min_cost
    coverage_min: dict = field(default_factory=dict)  # group value -> minimum share of that group's eligible customers contacted
    time_limit: float = 20.0
    mip_gap: float = 2e-3  # relative optimality gap; reported with every run


@dataclass
class Problem:
    ids: np.ndarray
    risk: np.ndarray
    arr: np.ndarray  # n x A
    se: np.ndarray
    elig: np.ndarray  # n x A bool
    group: np.ndarray | None = None
    true_arr: np.ndarray | None = None  # synthetic benchmark only
    arms: list = field(default_factory=lambda: list(TREATMENTS))


def candidate_mask(p: Problem, c: OptConfig):
    A = len(p.arms)
    cols = np.array([a in c.available for a in p.arms])
    m = p.elig & cols[None, :]
    elig_only = m.copy()
    unc = np.zeros_like(m)
    if c.max_se is not None:
        unc = m & (p.se > c.max_se)
        m = m & ~unc
    if c.min_arr > 0:
        m = m & ((p.arr - c.z * p.se) >= c.min_arr)
    return m, elig_only, unc


def _unit_values(p: Problem, c: OptConfig):
    cost = np.array([c.costs[a] for a in p.arms])
    adj = p.arr - c.z * p.se
    gross = c.value_per_event * adj
    net = gross - cost[None, :]
    return adj, gross, net, cost


def _resource_rows(p: Problem, c: OptConfig):
    slots = np.array([AGENT_SLOTS.get(a, 0.0) for a in p.arms])
    is_channel = np.array([RESOURCE.get(a) == "channel" for a in p.arms])
    return slots, is_channel


def solve_milp(p: Problem, c: OptConfig) -> dict:
    t0 = time.perf_counter()
    n, A = p.arr.shape
    mask, elig_only, unc = candidate_mask(p, c)
    adj, gross, net, cost = _unit_values(p, c)
    slots, is_channel = _resource_rows(p, c)
    # prune pairs that can never help (net objective without coverage constraints): keeps the problem sparse
    keep = mask.copy()
    if c.objective == "net_benefit" and not c.coverage_min:
        keep &= net > 0
    elif c.objective == "avoided_events":
        keep &= adj > 0
    i_idx, a_idx = np.nonzero(keep)
    m = len(i_idx)
    meta = {"n_variables": int(m), "n_customers": int(n)}
    if m == 0:
        return _empty(p, c, "optimal", "No candidate (customer, action) pair passes eligibility, benefit and uncertainty filters.", t0, meta, elig_only, unc)
    if c.objective == "net_benefit":
        obj = -net[i_idx, a_idx]
    elif c.objective == "avoided_events":
        obj = -adj[i_idx, a_idx]
    elif c.objective == "min_cost":
        obj = cost[a_idx]
    else:
        raise ValueError("objective must be net_benefit, avoided_events or min_cost")
    rows, cols, vals, lo, hi, names = [], [], [], [], [], []
    r = 0

    def add(row_cols, row_vals, lb, ub, name):
        nonlocal r
        rows.extend([r] * len(row_cols)); cols.extend(row_cols); vals.extend(row_vals)
        lo.append(lb); hi.append(ub); names.append(name); r += 1

    # one action per customer: sparse block built vectorised
    uniq, inv = np.unique(i_idx, return_inverse=True)
    one_rows = sparse.csr_matrix((np.ones(m), (inv, np.arange(m))), shape=(len(uniq), m))
    add(range(m), cost[a_idx], -np.inf, c.budget, "budget")
    ag = slots[a_idx]
    if (ag > 0).any():
        add(np.nonzero(ag > 0)[0], ag[ag > 0], -np.inf, c.agent_capacity, "agent_capacity")
    ch = is_channel[a_idx]
    if ch.any():
        add(np.nonzero(ch)[0], np.ones(ch.sum()), -np.inf, c.channel_capacity, "channel_capacity")
    for a, cap in c.per_intervention_cap.items():
        if a in p.arms and cap is not None:
            sel = np.nonzero(a_idx == p.arms.index(a))[0]
            if len(sel):
                add(sel, np.ones(len(sel)), -np.inf, cap, f"cap_{a}")
    if c.max_contacts is not None:
        add(range(m), np.ones(m), -np.inf, c.max_contacts, "max_contacts")
    if c.objective == "min_cost":
        if c.target_events is None:
            raise ValueError("target_events is required for min_cost")
        add(range(m), adj[i_idx, a_idx], c.target_events, np.inf, "target_events")
    infeasible_cov = []
    if c.coverage_min and p.group is not None:
        for g, share in c.coverage_min.items():
            n_el = int((elig_only.any(axis=1) & (p.group == g)).sum())
            need = int(np.ceil(share * n_el))
            sel = np.nonzero(p.group[i_idx] == g)[0]
            if need > 0:
                if len(sel) < need:
                    infeasible_cov.append(f"group {g}: needs {need} contacts but only {len(sel)} feasible candidates")
                add(sel, np.ones(len(sel)), need, np.inf, f"coverage_{g}")
    if infeasible_cov:
        return _empty(p, c, "infeasible", "Coverage requirement exceeds feasible candidates: " + "; ".join(infeasible_cov), t0, meta, elig_only, unc)
    extra = sparse.csr_matrix((vals, (rows, cols)), shape=(r, m))
    Amat = sparse.vstack([one_rows, extra]).tocsr()
    lb = np.concatenate([np.full(len(uniq), -np.inf), lo]); ub = np.concatenate([np.ones(len(uniq)), hi])
    res = milp(c=obj, constraints=LinearConstraint(Amat, lb, ub), integrality=np.ones(m), bounds=Bounds(0, 1),
               options={"time_limit": c.time_limit, "mip_rel_gap": c.mip_gap})
    status_map = {0: "optimal", 1: "time_limit_or_iteration_limit", 2: "infeasible", 3: "unbounded", 4: "solver_error"}
    status = status_map.get(res.status, "solver_error")
    if res.x is None:
        msg = {"infeasible": "No allocation satisfies all constraints. Relax the budget/capacity/coverage/target or loosen filters.",
               }.get(status, res.message)
        return _empty(p, c, status, msg, t0, meta, elig_only, unc)
    x = res.x > 0.5
    # shadow prices from the LP relaxation (marginal value per unit of constraint relaxation)
    shadow = {}
    try:
        con_rows = [(i, n_) for i, n_ in enumerate(names) if np.isfinite(hi[i])]
        if con_rows and c.objective != "min_cost":
            Aub = sparse.vstack([one_rows, extra[[i for i, _ in con_rows]]]).tocsr()
            bub = np.concatenate([np.ones(len(uniq)), [hi[i] for i, _ in con_rows]])
            lp = linprog(c=obj, A_ub=Aub, b_ub=bub, bounds=(0, 1), method="highs")
            if lp.status == 0:
                marg = lp.ineqlin.marginals[len(uniq):]
                shadow = {n_: float(-mv) for (_, n_), mv in zip(con_rows, marg)}
    except Exception:  # shadow prices are informative only
        shadow = {}
    sel_i, sel_a = i_idx[x], a_idx[x]
    usage = {"budget": float(cost[sel_a].sum()), "agent_capacity": float(slots[sel_a].sum()), "channel_capacity": int(is_channel[sel_a].sum())}
    limits = {"budget": c.budget, "agent_capacity": c.agent_capacity, "channel_capacity": c.channel_capacity}
    binding = [k for k in usage if np.isfinite(limits[k]) and limits[k] - usage[k] <= max(1e-6, 1e-4 * limits[k])]
    for a, cap in c.per_intervention_cap.items():
        if cap is not None and a in p.arms and int((sel_a == p.arms.index(a)).sum()) >= cap:
            binding.append(f"cap_{a}")
    if c.max_contacts is not None and len(sel_i) >= c.max_contacts:
        binding.append("max_contacts")
    meta.update({"mip_gap_reported": getattr(res, "mip_gap", None), "shadow_prices": shadow})
    return _finish(p, c, sel_i, sel_a, status, "", t0, meta, elig_only, unc, binding)


def _empty(p, c, status, msg, t0, meta, elig_only, unc):
    return _finish(p, c, np.array([], int), np.array([], int), status, msg, t0, meta, elig_only, unc, [])


def _finish(p, c, sel_i, sel_a, status, msg, t0, meta, elig_only, unc, binding):
    out = {"selected_i": sel_i, "selected_a": sel_a, "status": status, "message": msg, "runtime_s": time.perf_counter() - t0,
           "binding_constraints": binding, "meta": meta, "excluded_by_eligibility": int((~elig_only.any(axis=1)).sum()),
           "excluded_by_uncertainty": int((unc.any(axis=1) & ~(elig_only & ~unc).any(axis=1)).sum())}
    return out


# ------------------------------------------------------------------------------------------
# Baseline strategies (heuristics, clearly labelled as such)
# ------------------------------------------------------------------------------------------
class _Ledger:
    def __init__(self, p: Problem, c: OptConfig):
        self.p, self.c = p, c
        self.cost = np.array([c.costs[a] for a in p.arms])
        self.slots, self.is_channel = _resource_rows(p, c)
        self.spent = 0.0; self.agent = 0.0; self.chan = 0; self.contacts = 0
        self.counts = np.zeros(len(p.arms), int)
        self.assigned = np.zeros(len(p.ids), bool)
        self.si, self.sa = [], []

    def can(self, i, a):
        if self.assigned[i]:
            return False
        c = self.c
        if self.spent + self.cost[a] > c.budget + 1e-9 or self.agent + self.slots[a] > c.agent_capacity + 1e-9:
            return False
        if self.is_channel[a] and self.chan + 1 > c.channel_capacity:
            return False
        cap = c.per_intervention_cap.get(self.p.arms[a])
        if cap is not None and self.counts[a] + 1 > cap:
            return False
        if c.max_contacts is not None and self.contacts + 1 > c.max_contacts:
            return False
        return True

    def take(self, i, a):
        self.assigned[i] = True; self.spent += self.cost[a]; self.agent += self.slots[a]
        self.chan += int(self.is_channel[a]); self.contacts += 1; self.counts[a] += 1
        self.si.append(i); self.sa.append(a)


def _greedy_pairs(p, c, order_pairs):
    L = _Ledger(p, c)
    for i, a in order_pairs:
        if L.can(i, a):
            L.take(i, a)
    return np.array(L.si, int), np.array(L.sa, int)


def policy_assignments(name: str, p: Problem, c: OptConfig, rng: np.random.Generator | None = None, naive_arr: np.ndarray | None = None):
    """Return (selected_i, selected_a, label) for a heuristic baseline."""
    adj, gross, net, cost = _unit_values(p, c)
    mask, _, _ = candidate_mask(p, OptConfig(**{**c.__dict__, "min_arr": 0.0, "max_se": None}))
    if name == "no_intervention":
        return np.array([], int), np.array([], int)
    if name == "random":
        order = rng.permutation(len(p.ids))
        pairs = []
        for i in order:
            opts = np.nonzero(mask[i])[0]
            if len(opts):
                pairs.append((i, int(rng.choice(opts))))
        return _greedy_pairs(p, c, pairs)
    if name in ("highest_risk_first_intensive", "highest_risk_first_cheapest"):
        order = np.argsort(-p.risk)
        pairs = []
        for i in order:
            opts = np.nonzero(mask[i])[0]
            if len(opts):
                a = opts[np.argmax(cost[opts])] if name.endswith("intensive") else opts[np.argmin(cost[opts])]
                pairs.append((i, int(a)))
        L = _Ledger(p, c)
        for i, a in pairs:  # intensive: fall back to cheaper eligible actions when capacity/budget blocks
            if L.can(i, a):
                L.take(i, a)
            elif name.endswith("intensive"):
                for a2 in sorted(np.nonzero(mask[i])[0], key=lambda k: -cost[k]):
                    if L.can(i, int(a2)):
                        L.take(i, int(a2)); break
        return np.array(L.si, int), np.array(L.sa, int)
    if name in ("gross_benefit_first", "net_benefit_first"):
        key = np.where(mask, gross if name.startswith("gross") else net, -np.inf)
        flat = np.argsort(-key, axis=None)
        i_, a_ = np.unravel_index(flat, key.shape)
        ok = np.isfinite(key[i_, a_]) & (net[i_, a_] > 0 if name.startswith("net") else gross[i_, a_] > 0)
        return _greedy_pairs(p, c, zip(i_[ok].tolist(), a_[ok].tolist()))
    raise ValueError(name)


# ------------------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------------------
def summarize(p: Problem, c: OptConfig, sel_i, sel_a, label: str, status="heuristic", message="", runtime=0.0,
              binding=None, excl_elig=None, excl_unc=0, extra=None) -> dict:
    cost = np.array([c.costs[a] for a in p.arms])
    slots, is_ch = _resource_rows(p, c)
    n_sel = len(sel_i)
    arr = p.arr[sel_i, sel_a] if n_sel else np.array([])
    se = p.se[sel_i, sel_a] if n_sel else np.array([])
    spend = float(cost[sel_a].sum()) if n_sel else 0.0
    events = float(arr.sum())
    gross = events * c.value_per_event
    agent_used = float(slots[sel_a].sum()) if n_sel else 0.0
    out = {
        "strategy": label, "n_selected": int(n_sel), "budget_used": spend, "budget_remaining": float(c.budget - spend),
        "expected_avoided_events": events, "gross_benefit": gross, "net_benefit": gross - spend,
        "avg_benefit_per_selected": float(gross / n_sel) if n_sel else 0.0,
        "benefit_cost_ratio": float(gross / spend) if spend > 0 else None,
        "agent_slots_used": agent_used, "agent_utilisation": agent_used / c.agent_capacity if c.agent_capacity else None,
        "channel_used": int(is_ch[sel_a].sum()) if n_sel else 0,
        "mix": {p.arms[k]: int((sel_a == k).sum()) for k in range(len(p.arms))},
        # standard error of the *aggregate* assuming independent estimate errors (documented simplification)
        "events_se_independent": float(np.sqrt((se ** 2).sum())) if n_sel else 0.0,
        "mean_selected_risk": float(p.risk[sel_i].mean()) if n_sel else None,
        "solver_status": status, "message": message, "runtime_s": runtime,
        "binding_constraints": binding or [], "excluded_by_eligibility": excl_elig, "excluded_by_uncertainty": excl_unc,
    }
    if p.true_arr is not None:
        t_events = float(p.true_arr[sel_i, sel_a].sum()) if n_sel else 0.0
        out["synthetic_truth"] = {"label": "SYNTHETIC GROUND TRUTH (benchmark only)", "avoided_events": t_events,
                                  "gross_benefit": t_events * c.value_per_event, "net_benefit": t_events * c.value_per_event - spend}
    if extra:
        out.update(extra)
    return out


def optimise(p: Problem, c: OptConfig) -> tuple[dict, np.ndarray, np.ndarray]:
    r = solve_milp(p, c)
    s = summarize(p, c, r["selected_i"], r["selected_a"], "optimised" if c.z == 0 else f"conservative (z={c.z})", status=r["status"],
                  message=r["message"], runtime=r["runtime_s"], binding=r["binding_constraints"], excl_elig=r["excluded_by_eligibility"],
                  excl_unc=r["excluded_by_uncertainty"], extra={"n_variables": r["meta"].get("n_variables"),
                                                                "shadow_prices": r["meta"].get("shadow_prices", {})})
    return s, r["selected_i"], r["selected_a"]


STRATEGY_LABELS = {
    "no_intervention": "No intervention", "random": "Random eligible (mean of repeated draws)",
    "highest_risk_first_intensive": "Highest-risk-first (most intensive eligible action)",
    "highest_risk_first_cheapest": "Highest-risk-first (cheapest eligible action)",
    "gross_benefit_first": "Highest estimated gross benefit first", "net_benefit_first": "Highest estimated net benefit first",
}


def compare_strategies(p: Problem, c: OptConfig, seed=42, random_draws=50, conservative_z=1.28, naive_arr: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    results = []
    allocations = {}
    _, elig_only, _ = candidate_mask(p, c)
    excl = int((~elig_only.any(axis=1)).sum())
    for name in ("no_intervention", "highest_risk_first_cheapest", "highest_risk_first_intensive", "gross_benefit_first", "net_benefit_first"):
        t0 = time.perf_counter()
        si, sa = policy_assignments(name, p, c, rng)
        results.append(summarize(p, c, si, sa, STRATEGY_LABELS[name], runtime=time.perf_counter() - t0, excl_elig=excl))
        allocations[name] = (si, sa)
    # random: repeated draws -> mean and sd of outcomes
    draws = []
    t0 = time.perf_counter()
    for _ in range(random_draws):
        si, sa = policy_assignments("random", p, c, rng)
        draws.append(summarize(p, c, si, sa, "random"))
    keys = ["n_selected", "budget_used", "expected_avoided_events", "gross_benefit", "net_benefit"]
    r_si, r_sa = policy_assignments("random", p, c, np.random.default_rng(seed))
    allocations["random"] = (r_si, r_sa)
    rnd = summarize(p, c, r_si, r_sa, STRATEGY_LABELS["random"], excl_elig=excl)
    for k in keys:
        rnd[k] = float(np.mean([d[k] for d in draws])); rnd[f"{k}_sd_over_draws"] = float(np.std([d[k] for d in draws]))
    rnd["net_benefit_p05_p95"] = [float(np.percentile([d["net_benefit"] for d in draws], 5)), float(np.percentile([d["net_benefit"] for d in draws], 95))]
    rnd["budget_remaining"] = c.budget - rnd["budget_used"]; rnd["runtime_s"] = time.perf_counter() - t0
    rnd["message"] = f"mean over {random_draws} random draws (seed {seed})"
    results.append(rnd)
    s, si, sa = optimise(p, c)
    results.append(s); allocations["optimised"] = (si, sa)
    if conservative_z > 0:
        c2 = OptConfig(**{**c.__dict__, "z": conservative_z})
        s2, si2, sa2 = optimise(p, c2)
        # report the conservative allocation's *point-estimate* benefit for comparability, objective used the lower bound
        s2 = summarize(p, c, si2, sa2, f"Conservative (maximise lower bound, z={conservative_z})", status=s2["solver_status"],
                       message=s2["message"], runtime=s2["runtime_s"], binding=s2["binding_constraints"], excl_elig=excl,
                       extra={"objective_note": "Optimised against ARR - z*SE per pair (box uncertainty set); benefit columns use point estimates."})
        results.append(s2); allocations["conservative"] = (si2, sa2)
    if naive_arr is not None:
        pn = Problem(p.ids, p.risk, naive_arr, np.zeros_like(naive_arr), p.elig, p.group, p.true_arr, p.arms)
        _, nsi, nsa = optimise(pn, c)
        allocations["naive_optimiser"] = (nsi, nsa)
        results.append(summarize(p, c, nsi, nsa, "Optimiser on naive (unadjusted, risk-decile) effects", status="optimal",
                                 excl_elig=excl, extra={"objective_note": "Simpler baseline: raw treated-vs-untreated differences within risk deciles."}))
    keys = ["no_intervention", "highest_risk_first_cheapest", "highest_risk_first_intensive", "gross_benefit_first", "net_benefit_first",
            "random", "optimised"] + (["conservative"] if conservative_z > 0 else []) + (["naive_optimiser"] if naive_arr is not None else [])
    for k, r in zip(keys, results):
        r["key"] = k
    return {"strategies": results, "allocations": allocations}
