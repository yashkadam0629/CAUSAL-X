# Optimisation methodology

## Formulation

Index customers *i*, actions *a ∈ {standard_reminder, counselling, repayment_plan}* (the "none" action is the absence of any selection).

```
x[i,a] ∈ {0,1}                             variables exist only for eligible, available, filter-passing pairs
ARR~[i,a] = ARR[i,a] - z · SE[i,a]         z = 0: point estimate; z > 0: lower confidence bound (conservative)
gross[i,a] = V · ARR~[i,a]                 V = value per avoided adverse event (analyst assumption)
net[i,a]   = gross[i,a] - c[a]             cost is subtracted exactly once, here

maximise   Σ x[i,a] · net[i,a]                              (objective "net_benefit")
   or      Σ x[i,a] · ARR~[i,a]                             (objective "avoided_events")
   or  min Σ x[i,a] · c[a]  s.t. Σ x·ARR~ ≥ target          (objective "min_cost")
s.t.       Σ_a x[i,a] ≤ 1                         ∀ i       (mutually exclusive actions)
           Σ c[a] x[i,a] ≤ Budget
           Σ s[a] x[i,a] ≤ AgentSlots             s = 1.0 counselling, 1.5 repayment plan
           Σ x[i,reminder] ≤ ChannelCapacity
           Σ_i x[i,a] ≤ Cap[a], Σ x ≤ MaxContacts (optional)
           Σ_{i∈g,a} x[i,a] ≥ ceil(share_g · |eligible_g|)   coverage minimum per operational group g (optional)
```

Filters remove variables rather than adding constraints: ineligible/unavailable pairs, `ARR~ < min_arr`, `SE > max_se` (reported as "uncertainty-related exclusions"), and (for the net objective with no coverage constraint) pairs with `net ≤ 0`, which keeps the model sparse (≈ 2.5 variables per decision customer after pruning).

Solver: `scipy.optimize.milp` (HiGHS) with a sparse CSR constraint matrix, relative gap 0.2 %, 20 s limit; the status is reported honestly (`optimal`, `infeasible`, `time_limit_or_iteration_limit`, ...). An infeasible model returns no allocation, no success flag, and advice. Shadow prices come from the LP relaxation (`linprog`/HiGHS marginals) and are reported as the marginal net benefit per extra unit of budget / agent slots / channel capacity. Binding constraints are those with slack ≤ 0.01 %.

## Conservative objective: what it means

Replacing each estimate by `ARR - z·SE` is a **box uncertainty set**: robust optimisation against every estimate being simultaneously at its lower confidence bound. It is deliberately pessimistic (stricter than the lower bound of the *aggregate* benefit when errors are independent) and not a probabilistic guarantee. Its purpose is to counter the **optimiser's curse**: choosing the largest of many noisy estimates systematically overstates benefit. The Scenario Studio quantifies this: errors are simulated around plausible truths, an optimiser is run on the noisy re-estimates, and claimed benefit is compared with benefit under the simulated truth; the conservative policy shows a smaller gap (tested in `test_optimization.py`).

## Strategies compared (all under identical constraints)

1. No intervention.
2. Random eligible selection (mean and spread over repeated draws).
3. Highest-risk-first, *cheapest* eligible action; and *most intensive* eligible action that capacity allows.
4. Highest estimated gross benefit first (greedy; ignores cost).
5. Highest estimated net benefit first (greedy; ignores interactions between constraints).
6. Optimised MILP.
7. Conservative MILP (z = 1.28).
8. Optimiser fed **naive** effects (raw treated-vs-untreated gaps within baseline-risk deciles): the simpler baseline model that shows what ignoring confounding costs.

Heuristics are labelled as heuristics. Every strategy reports: customers, spend, remaining budget, expected avoided events, gross/net benefit, benefit per selected customer, benefit-cost ratio, agent/channel use, mix, exclusions, binding constraints, status, runtime.

## Reported uncertainty

* `events_se_independent = sqrt(Σ SE²)` over selected pairs assumes independent estimation errors; positively correlated errors make it too small.
* Scenario Monte Carlo draws `ARR ~ N(estimate, SE²)` independently per selected pair for a fixed allocation (seeded; draw count shown).
* None of these include confounding, model-misspecification or cost/value uncertainty.

## Scenarios

A-J are implemented as overrides on the base problem (see `simulation/scenarios.py`). Economic deterioration shifts baseline risk on the logit scale and scales effects proportionally to baseline risk (an assumption stated in the UI). "Reliable effects only" caps SE at the median. The budget and agent-capacity frontiers re-solve the MILP over a grid and report finite-difference marginal value and the budget beyond which spending stops being worthwhile.

## Value of information

`EVPI_i = E[max(0, max_a V_ia)] − max(0, max_a E[V_ia])` with independent normal errors; summed over customers and **ignoring budget and capacity**, so it is an upper bound on what perfect knowledge of the effects could add. It is not the value of any specific experiment (that is EVSI and needs the experiment's size and cost).
