import numpy as np
import pytest

from app.data.schema import AGENT_SLOTS, TREATMENTS
from app.optimization.engine import OptConfig, Problem, compare_strategies, optimise, policy_assignments, solve_milp, summarize
from app.simulation.scenarios import SCENARIO_CATALOGUE, apply_overrides, budget_frontier, monte_carlo, run_scenario, value_of_information, winners_curse


def make_problem(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    risk = rng.beta(2, 5, n)
    arr = np.column_stack([rng.normal(0.05, 0.04, n), rng.normal(0.05, 0.05, n), rng.normal(0.06, 0.06, n)])
    se = np.full((n, 3), 0.02)
    elig = rng.random((n, 3)) < [0.97, 0.7, 0.5]
    return Problem(np.array([f"C{i}" for i in range(n)]), risk, arr, se, elig, rng.choice(["R1", "R2"], n), arr.copy())


CFG = OptConfig(budget=30000, agent_capacity=25, channel_capacity=400)


def used(p, c, si, sa):
    cost = np.array([c.costs[a] for a in p.arms])
    slots = np.array([AGENT_SLOTS.get(a, 0) for a in p.arms])
    return cost[sa].sum(), slots[sa].sum(), (sa == 0).sum()


def test_respects_budget_capacity_eligibility_and_exclusivity():
    p = make_problem()
    s, si, sa = optimise(p, CFG)
    spend, slots, chan = used(p, CFG, si, sa)
    assert s["solver_status"] == "optimal"
    assert spend <= CFG.budget + 1e-6 and slots <= CFG.agent_capacity + 1e-6 and chan <= CFG.channel_capacity
    assert p.elig[si, sa].all(), "an ineligible action was recommended"
    assert len(set(si.tolist())) == len(si), "customer received more than one action"
    assert s["budget_used"] == pytest.approx(spend)


def test_cost_counted_once_net_equals_gross_minus_spend():
    p = make_problem()
    s, si, sa = optimise(p, CFG)
    assert s["net_benefit"] == pytest.approx(s["gross_benefit"] - s["budget_used"])
    assert s["gross_benefit"] == pytest.approx(p.arr[si, sa].sum() * CFG.value_per_event)


def test_optimiser_beats_all_heuristics_on_its_own_objective():
    p = make_problem()
    r = compare_strategies(p, CFG, random_draws=10)
    by = {s["key"]: s for s in r["strategies"]}
    for k in ("no_intervention", "random", "highest_risk_first_cheapest", "highest_risk_first_intensive", "gross_benefit_first", "net_benefit_first"):
        assert by["optimised"]["net_benefit"] >= by[k]["net_benefit"] - 1e-6, k
    assert by["conservative"]["solver_status"] == "optimal"
    assert by["random"]["net_benefit_sd_over_draws"] >= 0 and "mean over" in by["random"]["message"]


def test_heuristics_obey_constraints_too():
    p = make_problem()
    rng = np.random.default_rng(1)
    for name in ("random", "highest_risk_first_intensive", "highest_risk_first_cheapest", "gross_benefit_first", "net_benefit_first"):
        si, sa = policy_assignments(name, p, CFG, rng)
        spend, slots, chan = used(p, CFG, si, sa)
        assert spend <= CFG.budget + 1e-6 and slots <= CFG.agent_capacity + 1e-6 and chan <= CFG.channel_capacity, name
        assert p.elig[si, sa].all() and len(set(si.tolist())) == len(si), name


def test_unavailable_intervention_never_selected():
    p = make_problem()
    c = OptConfig(**{**CFG.__dict__, "available": ["standard_reminder"]})
    s, si, sa = optimise(p, c)
    assert set(sa.tolist()) <= {0}


def test_infeasible_coverage_is_reported_not_called_success():
    p = make_problem()
    c = OptConfig(**{**CFG.__dict__, "coverage_min": {"R1": 1.0}, "budget": 100.0})
    s, si, sa = optimise(p, c)
    assert s["solver_status"] == "infeasible" and s["n_selected"] == 0 and s["message"]


def test_coverage_constraint_is_met_when_feasible():
    p = make_problem()
    c = OptConfig(**{**CFG.__dict__, "budget": 200000, "agent_capacity": 500, "coverage_min": {"R2": 0.5}})
    s, si, sa = optimise(p, c)
    n_el = int((p.elig.any(axis=1) & (p.group == "R2")).sum())
    assert (p.group[si] == "R2").sum() >= int(np.ceil(0.5 * n_el))


def test_uncertainty_cap_and_min_benefit_exclude_pairs():
    p = make_problem()
    p.se[:, 0] = 0.2  # reminders very uncertain
    s, si, sa = optimise(p, OptConfig(**{**CFG.__dict__, "max_se": 0.05}))
    assert 0 not in set(sa.tolist()) and s["excluded_by_uncertainty"] >= 0
    s2, si2, sa2 = optimise(p, OptConfig(**{**CFG.__dict__, "min_arr": 0.08}))
    assert (p.arr[si2, sa2] >= 0.08).all()


def test_conservative_objective_uses_lower_bound():
    p = make_problem()
    p.se[:, 2] = 0.15  # repayment plan: noisy
    s0, _, sa0 = optimise(p, CFG)
    s1, _, sa1 = optimise(p, OptConfig(**{**CFG.__dict__, "z": 1.28}))
    assert (sa1 == 2).sum() <= (sa0 == 2).sum()


def test_min_cost_objective_meets_target():
    p = make_problem()
    c = OptConfig(**{**CFG.__dict__, "objective": "min_cost", "target_events": 20.0, "budget": 1e9, "agent_capacity": 1e9, "channel_capacity": 10**6})
    s, si, sa = optimise(p, c)
    assert s["solver_status"] == "optimal" and p.arr[si, sa].sum() >= 20.0 - 1e-6
    c2 = OptConfig(**{**c.__dict__, "target_events": 1e6})
    assert optimise(p, c2)[0]["solver_status"] == "infeasible"


def test_reproducible_outputs():
    p = make_problem()
    a = optimise(p, CFG)[0]
    b = optimise(p, CFG)[0]
    assert a["net_benefit"] == pytest.approx(b["net_benefit"]) and a["n_selected"] == b["n_selected"]
    r1 = compare_strategies(p, CFG, seed=5, random_draws=5)["strategies"]
    r2 = compare_strategies(p, CFG, seed=5, random_draws=5)["strategies"]
    assert [x["net_benefit"] for x in r1] == [x["net_benefit"] for x in r2]


def test_scenarios_move_in_the_expected_direction():
    p = make_problem()
    base = run_scenario(p, CFG, {}, "base", mc_draws=50)["optimised"]["net_benefit"]
    assert run_scenario(p, CFG, SCENARIO_CATALOGUE["B_budget_cut"]["overrides"], "cut", mc_draws=50)["optimised"]["net_benefit"] <= base + 1e-6
    assert run_scenario(p, CFG, SCENARIO_CATALOGUE["F_weaker_effects"]["overrides"], "weak", mc_draws=50)["optimised"]["net_benefit"] < base
    assert run_scenario(p, CFG, SCENARIO_CATALOGUE["E_cost_increase"]["overrides"], "cost", mc_draws=50)["optimised"]["net_benefit"] <= base + 1e-6
    p2, c2 = apply_overrides(p, CFG, {"econ_shift": 0.5})
    assert (p2.risk > p.risk).all() and (p2.arr >= p.arr).all() or (p2.arr / p.arr > 1).mean() > 0.4


def test_monte_carlo_is_seeded_and_labelled():
    p = make_problem()
    s, si, sa = optimise(p, CFG)
    a, b = monte_carlo(p, CFG, si, sa, 100, 3), monte_carlo(p, CFG, si, sa, 100, 3)
    assert a == b and a["draws"] == 100 and "unmeasured confounding" in a["excludes"]
    lo, hi = a["net_benefit_p05_p95"]
    assert lo < a["net_benefit_mean"] < hi


def test_frontier_nondecreasing_and_voi_nonnegative_and_curse_positive():
    p = make_problem(800)
    f = budget_frontier(p, CFG, points=6)["points"]
    nets = [x["net_benefit"] for x in f]
    assert all(b >= a - 1e-6 for a, b in zip(nets, nets[1:]))
    v = value_of_information(p, CFG, draws=50)
    assert v["evpi_total_unconstrained"] >= 0
    rng = np.random.default_rng(5)
    p.se = rng.uniform(0.005, 0.08, p.se.shape)  # heterogeneous precision: the winner's curse needs noisy and precise estimates side by side
    cfg = OptConfig(**{**CFG.__dict__, "budget": 12000, "agent_capacity": 15, "channel_capacity": 150})
    w = winners_curse(p, cfg, trials=6)
    assert w["summary"]["unconstrained optimiser"]["mean_optimism_gap"] > 0
    assert w["summary"]["conservative optimiser"]["mean_optimism_gap"] < w["summary"]["unconstrained optimiser"]["mean_optimism_gap"]


def test_solver_failure_handling_invalid_objective():
    p = make_problem(100)
    with pytest.raises(ValueError):
        solve_milp(p, OptConfig(objective="nonsense"))
