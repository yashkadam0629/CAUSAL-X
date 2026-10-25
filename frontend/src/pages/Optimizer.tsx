import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend as RLegend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApp } from '../app/App'
import Gate from '../components/Gate'
import { Alert, Btn, Card, ChartTooltip, DataTable, ErrorBox, Field, Note, NumInput, PageHeader, Select, Stat, Tag } from '../components/ui'
import { useAction } from '../hooks/useAsync'
import { api } from '../services/api'
import { ARM_LABEL, fmt, label, money, pct } from '../utils/format'

const ACTIONS = ['standard_reminder', 'counselling', 'repayment_plan']

function Body() {
  const { current, setLastRunId } = useApp()
  const [c, setC] = useState<any>({ costs: { standard_reminder: 40, counselling: 250, repayment_plan: 500 }, available: [...ACTIONS], objective: 'net_benefit', min_arr: 0, z: 0 })
  const [cov, setCov] = useState<Record<string, number | undefined>>({})
  const [cap, setCap] = useState<Record<string, number | undefined>>({})
  const run = useAction(async () => {
    const coverage_min = Object.fromEntries(Object.entries(cov).filter(([, v]) => v).map(([k, v]) => [k, v]))
    const per_intervention_cap = Object.fromEntries(Object.entries(cap).filter(([, v]) => v).map(([k, v]) => [k, v]))
    const r = await api.post('/optimization/run', { dataset_id: current!.id, ...c, coverage_min: Object.keys(coverage_min).length ? coverage_min : undefined, per_intervention_cap: Object.keys(per_intervention_cap).length ? per_intervention_cap : undefined })
    if (r.run_id) setLastRunId(r.run_id)
    return r
  })
  const r = run.result
  const s = r?.summary
  const strat = r?.strategies?.filter((x: any) => x.key) ?? []
  const truth = !!s?.synthetic_truth
  const toggle = (a: string) => setC({ ...c, available: c.available.includes(a) ? c.available.filter((x: string) => x !== a) : [...c.available, a] })
  return (
    <>
      <PageHeader title="Intervention Optimizer" subtitle="Sparse mixed-integer programme (HiGHS): maximise estimated net benefit subject to budget, capacity, eligibility, mutual exclusivity and optional uncertainty and coverage constraints. Leave a field blank to use the data-scaled default." />
      <div className="grid gap-4 xl:grid-cols-[340px_1fr]">
        <Card title="Constraints & assumptions">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Budget" hint="blank = 18 × customers"><NumInput value={c.budget} onChange={v => setC({ ...c, budget: v })} /></Field>
            <Field label="Agent slots" hint="counselling 1.0, plan 1.5"><NumInput value={c.agent_capacity} onChange={v => setC({ ...c, agent_capacity: v })} /></Field>
            <Field label="Channel capacity"><NumInput value={c.channel_capacity} onChange={v => setC({ ...c, channel_capacity: v })} /></Field>
            <Field label="Max contacts"><NumInput value={c.max_contacts} onChange={v => setC({ ...c, max_contacts: v })} /></Field>
            <Field label="Value / avoided event" hint="analyst assumption"><NumInput value={c.value_per_event} onChange={v => setC({ ...c, value_per_event: v })} /></Field>
            <Field label="Min est. ARR"><NumInput value={c.min_arr} step={0.005} onChange={v => setC({ ...c, min_arr: v ?? 0 })} /></Field>
            <Field label="Max SE (uncertainty cap)"><NumInput value={c.max_se} step={0.005} onChange={v => setC({ ...c, max_se: v })} /></Field>
            <Field label="Conservatism z" hint="0 = point estimates"><NumInput value={c.z} step={0.1} min={0} max={4} onChange={v => setC({ ...c, z: v ?? 0 })} /></Field>
          </div>
          <div className="mt-3"><Field label="Objective"><Select value={c.objective} onChange={v => setC({ ...c, objective: v })} options={[{ value: 'net_benefit', label: 'Maximise net benefit' }, { value: 'avoided_events', label: 'Maximise avoided events (budget-constrained)' }, { value: 'min_cost', label: 'Minimise cost for a target' }]} /></Field></div>
          {c.objective === 'min_cost' && <div className="mt-2"><Field label="Target avoided events"><NumInput value={c.target_events} onChange={v => setC({ ...c, target_events: v })} /></Field></div>}
          <fieldset className="mt-3"><legend className="mb-1 text-xs font-medium text-ink2">Available interventions, unit cost and cap</legend>
            {ACTIONS.map(a => (
              <div key={a} className="mb-1 grid grid-cols-[1fr_70px_70px] items-center gap-2 text-xs">
                <label className="flex items-center gap-1"><input type="checkbox" checked={c.available.includes(a)} onChange={() => toggle(a)} />{ARM_LABEL[a]}</label>
                <NumInput value={c.costs[a]} onChange={v => setC({ ...c, costs: { ...c.costs, [a]: v ?? 0 } })} /><NumInput value={cap[a]} onChange={v => setCap({ ...cap, [a]: v })} />
              </div>))}
            <Note>Column 2: unit cost (assumption). Column 3: max customers (optional).</Note></fieldset>
          <fieldset className="mt-3"><legend className="mb-1 text-xs font-medium text-ink2">Coverage: minimum share of each region's eligible customers contacted</legend>
            <div className="grid grid-cols-5 gap-1">{['R1', 'R2', 'R3', 'R4', 'R5'].map(g => <div key={g}><div className="text-center text-[10px] text-muted">{g}</div><NumInput value={cov[g]} step={0.05} min={0} max={1} onChange={v => setCov({ ...cov, [g]: v })} /></div>)}</div>
            <Note>Region is an operational coverage group, never a model input.</Note></fieldset>
          <div className="mt-4"><Btn busy={run.busy} onClick={() => run.run()}>Run optimisation & compare strategies</Btn></div>
          {run.error != null && <div className="mt-2"><ErrorBox error={run.error} /></div>}
        </Card>
        <div className="min-w-0 space-y-4">
          {!r && <Alert tone="info">Runs the MILP, then compares it with no intervention, random, highest-risk-first (two variants), gross- and net-benefit-first heuristics, a conservative policy and an optimiser fed naive unadjusted effects. Takes ~10–20 s.</Alert>}
          {r && (
            <>
              {s.solver_status !== 'optimal' && <Alert>Solver status <b>{s.solver_status}</b>. {s.message} {(r.advice ?? []).join(' ')}</Alert>}
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <Stat label="Solver" value={s.solver_status} hint={`${fmt(s.runtime_s, 2)} s · ${s.n_variables} variables`} />
                <Stat label="Customers selected" value={s.n_selected.toLocaleString()} kind="estimated" />
                <Stat label="Budget used" value={money(s.budget_used)} hint={`remaining ${money(s.budget_remaining)}`} kind="assumption" />
                <Stat label="Net benefit" value={money(s.net_benefit)} kind="estimated" tone={s.net_benefit > 0 ? 'good' : 'bad'} hint={`gross ${money(s.gross_benefit)}`} />
                <Stat label="Avoided events" value={fmt(s.expected_avoided_events, 1)} kind="estimated" hint={`± ${fmt(s.events_se_independent, 1)} SE`} />
                <Stat label="Benefit / cost" value={fmt(s.benefit_cost_ratio, 2)} hint="gross benefit ÷ spend" />
                <Stat label="Agent utilisation" value={pct(s.agent_utilisation, 0)} hint={`${fmt(s.agent_slots_used, 1)} slots`} />
                <Stat label="Binding constraints" value={s.binding_constraints.join(', ') || 'none'} hint="active at the optimum" />
              </div>
              {Object.keys(s.shadow_prices ?? {}).length > 0 && <Alert tone="info">Marginal value of relaxing a constraint by one unit (LP relaxation shadow prices): {Object.entries(s.shadow_prices).filter(([, v]) => (v as number) > 1e-6).map(([k, v]) => `${label(k)} ≈ ${fmt(v as number, 2)}`).join('; ') || 'none positive — constraints are slack'}.</Alert>}
              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="Intervention distribution"><div className="h-52"><ResponsiveContainer><BarChart data={Object.entries(s.mix).map(([a, n]) => ({ name: ARM_LABEL[a], n }))}><CartesianGrid vertical={false} /><XAxis dataKey="name" /><YAxis /><Tooltip content={<ChartTooltip />} />
                  <Bar dataKey="n" name="Customers" barSize={34} radius={[4, 4, 0, 0]}>{ACTIONS.map((_, i) => <Cell key={i} fill={`var(--s${i + 1})`} />)}</Bar></BarChart></ResponsiveContainer></div></Card>
                <Card title="Exclusion reasons" subtitle="Why customers received no intervention">
                  <DataTable cols={[{ key: 'k', header: 'Reason' }, { key: 'v', header: 'Customers', align: 'right', render: (x: any) => x.v.toLocaleString() }]} rows={Object.entries(r.exclusion_reasons).map(([k, v]) => ({ k, v }))} /></Card>
              </div>
              <Card title="Strategy comparison" subtitle="Identical budget and capacity. Heuristics are labelled as such; random is a mean over repeated draws.">
                <div className="h-64"><ResponsiveContainer><BarChart data={strat.map((x: any) => ({ name: x.strategy.replace(/ \(.*\)/, ''), est: Math.round(x.net_benefit), truth: x.synthetic_truth ? Math.round(x.synthetic_truth.net_benefit) : undefined }))} layout="vertical" margin={{ left: 0 }}>
                  <CartesianGrid horizontal={false} /><XAxis type="number" tickFormatter={v => v / 1000 + 'k'} /><YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 10 }} /><Tooltip content={<ChartTooltip fmt={money} />} /><RLegend />
                  <Bar dataKey="est" name="Estimated net" fill="var(--s1)" barSize={9} radius={[0, 4, 4, 0]} />{truth && <Bar dataKey="truth" name="Synthetic truth" fill="var(--s3)" barSize={9} radius={[0, 4, 4, 0]} />}</BarChart></ResponsiveContainer></div>
                <DataTable cols={[{ key: 'strategy', header: 'Strategy' }, { key: 'n_selected', header: 'Selected', align: 'right', render: (x: any) => Math.round(x.n_selected).toLocaleString() }, { key: 'budget_used', header: 'Spend', align: 'right', render: (x: any) => money(x.budget_used) },
                  { key: 'expected_avoided_events', header: 'Avoided events', align: 'right', render: (x: any) => fmt(x.expected_avoided_events, 1) }, { key: 'net_benefit', header: 'Net (est.)', align: 'right', render: (x: any) => money(x.net_benefit) },
                  { key: 'truth', header: 'Net (synthetic truth)', align: 'right', render: (x: any) => (x.synthetic_truth ? money(x.synthetic_truth.net_benefit) : '—') }, { key: 'solver_status', header: 'Method' }]} rows={strat} />
                {strat.find((x: any) => x.objective_note) && <Note>{strat.filter((x: any) => x.objective_note).map((x: any) => `${x.strategy}: ${x.objective_note}`).join(' ')}</Note>}
              </Card>
              <Card title="Top allocations by net benefit" subtitle={r.run_id ? `Saved as ${r.run_id} · policy ${r.policy_version} · recommendations stored in Audit & Governance` : 'Not saved'}>
                <DataTable cols={[{ key: 'customer_id', header: 'Customer' }, { key: 'action', header: 'Action', render: (x: any) => ARM_LABEL[x.action] }, { key: 'risk', header: 'Risk', align: 'right', render: (x: any) => pct(x.risk) },
                  { key: 'arr', header: 'ARR ± SE', align: 'right', render: (x: any) => `${fmt(x.arr, 3)} ± ${fmt(x.se, 3)}` }, { key: 'cost', header: 'Cost', align: 'right', render: (x: any) => money(x.cost) }, { key: 'net_benefit', header: 'Net', align: 'right', render: (x: any) => money(x.net_benefit) }]} rows={r.allocation_preview} />
              </Card>
              {r.fairness_audit && <Card title="Fairness audit of this allocation (descriptive, audit-only attributes)" subtitle="Not used in optimisation. Groups below n=100 are suppressed.">
                {Object.values(r.fairness_audit).map((a: any) => <div key={a.group_attribute} className="mb-3"><h3 className="text-xs font-semibold">{label(a.group_attribute)}</h3>
                  <DataTable cols={[{ key: 'group', header: 'Group' }, { key: 'n', header: 'n', align: 'right' }, { key: 'sel', header: 'Selection rate (95% CI)', align: 'right', render: (g: any) => (g.suppressed ? 'suppressed' : `${pct(g.selection_rate)} [${pct(g.selection_rate_ci[0], 0)}–${pct(g.selection_rate_ci[1], 0)}]`) },
                    { key: 'o', header: 'Observed rate', align: 'right', render: (g: any) => (g.suppressed ? '' : pct(g.observed_outcome_rate)) }, { key: 'p', header: 'Mean predicted', align: 'right', render: (g: any) => (g.suppressed ? '' : pct(g.mean_predicted_risk)) }, { key: 'b', header: 'Mean est. benefit', align: 'right', render: (g: any) => (g.suppressed ? '' : fmt(g.mean_estimated_benefit, 3)) }]} rows={a.groups} /></div>)}
                <Note>{Object.values(r.fairness_audit)[0] && (Object.values(r.fairness_audit)[0] as any).caveats.join(' ')}</Note></Card>}
              <Note>{r.limitations.join(' ')}</Note>
            </>)}
        </div>
      </div>
    </>)
}
export default function Optimizer() { return <Gate><Body /></Gate> }
