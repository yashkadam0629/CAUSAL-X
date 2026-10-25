# Experiment design

The Experiment Designer plans a *future* randomised experiment. It never approves anything: every protocol is stamped "DRAFT - not approved" and lists an ethics/operations checklist.

## Sample-size formulas (`experiment_design/sample_size.py`)

Two-sided normal approximations with allocation ratio `r = n_treat / n_control`:

* **Binary outcome** (unpooled two-proportion z-test): `n_c = (z_{1-α/2} + z_{power})² · [p0(1-p0) + p1(1-p1)/r] / δ²`, with `p1 = p0 + δ`.
* **Continuous outcome**: `n_c = (z_{1-α/2} + z_{power})² · σ² · (1 + 1/r) / δ²`.

Adjustments (each reported separately):

| Issue | Treatment in the calculator |
|---|---|
| Multiple comparisons (k arms vs control) | Bonferroni: α/k (conservative) |
| Non-delivery (ITT) | assignment effect diluted to `compliance × δ`, so `n` grows by ≈ `1/compliance²` |
| Attrition / unobserved outcomes | divide by `1 − attrition`; the protocol says to pre-specify handling and never impute favourable outcomes |
| Interference / clustering (households, agents) | design effect `1 + (m − 1)·ICC` |

Sanity checks in the test-suite: baseline 0.20, absolute lift 0.05, α=0.05, power 0.8 gives ≈1,090 per arm; continuous σ=1, δ=0.1 gives ≈1,570 per arm.

## Protocol contents

Objective, target population, eligibility, treatment, control, randomisation unit and stratification, allocation ratio, logging of assignment probability, primary/secondary outcomes, horizon, guardrails (complaints, opt-outs, contact fatigue, serious delinquency), sample-size derivation, duration, stopping rules, risks (contamination, missing outcomes, noncompliance, interference), ethics checklist. Export is JSON; saving stores it as an experiment record with an audit entry.

## Warnings and limits

Not covered: sequential/group-sequential boundaries (the stopping rule is a placeholder to be pre-specified), exact tests for rare events, cluster-level power beyond the design effect, covariate-adjusted power, Holm/Hochberg procedures. Ethical and regulatory approval is an institutional process that this tool cannot replace.
