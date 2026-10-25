import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend as RLegend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApp } from '../app/App'
import Gate from '../components/Gate'
import { Alert, Btn, Card, ChartTooltip, DataTable, ErrorBox, Field, Note, NumInput, PageHeader, TextInput } from '../components/ui'
import { useAction, useLoad } from '../hooks/useAsync'
import { api } from '../services/api'
import { download, fmt, money, pct } from '../utils/format'

function Body() {
  const { current } = useApp()
  const cat = useLoad(() => api.get('/scenarios/catalogue'), [])
  const saved = useLoad(() => api.get('/scenarios', { dataset_id: current!.id }), [current?.id])
  const [sel, setSel] = useState<string[]>(['A_base', 'B_budget_cut', 'C_budget_expansion', 'D_economic_deterioration', 'F_weaker_effects', 'G_limited_agents', 'J_conservative'])
  const [custom, setCustom] = useState<any>({ budget_mult: 1, effect_mult: 1, cost_mult: 1, econ_shift: 0, agent_mult: 1 })
  const [name, setName] = useState('')
  const [useCustom, setUseCustom] = useState(false)
  const run = useAction(async () => {
    const overrides = Object.fromEntries(Object.entries(custom).filter(([, v]) => v !== undefined && !(v === 1 || v === 0)))
    return api.post('/scenarios/run', { dataset_id: current!.id, scenarios: sel, custom: useCustom ? { name: 'Custom scenario', overrides } : undefined, mc_draws: 200, save_name: name || undefined })
  })
  const adv = useAction(() => api.post('/scenarios/advanced', { dataset_id: current!.id }))
  const [loadedSaved, setLoadedSaved] = useState<any>(null)
  const res = (loadedSaved ?? run.result)?.results as any[] | undefined
  const bars = res?.map(r => ({ name: r.label.replace(/\(.*\)/, '').slice(0, 34), net: Math.round(r.optimised.net_benefit), rf: Math.round(r.highest_risk_first.net_benefit), lo: Math.round(r.optimised.net_benefit - r.monte_carlo.net_benefit_p05_p95?.[0]), hi: Math.round(r.monte_carlo.net_benefit_p05_p95?.[1] - r.optimised.net_benefit) }))
  const a = adv.result
  const sc = (k: string) => (
    <label key={k} className="flex items-center gap-1.5 text-xs"><input type="checkbox" checked={sel.includes(k)} onChange={e => setSel(e.target.checked ? [...sel, k] : sel.filter(x => x !== k))} />{cat.data[k].label}</label>)
  return (
    <>
      <PageHeader title="Financial Scenario Studio" subtitle="Re-optimise under changed assumptions. All outputs are SIMULATED from model estimates; uncertainty bands propagate estimation error only." />
      <div className="grid gap-4 xl:grid-cols-[320px_1fr]">
        <Card title="Scenarios">
          <div className="space-y-1.5">{cat.data && Object.keys(cat.data).map(sc)}</div>
          <label className="mt-3 flex items-center gap-1.5 text-xs font-medium"><input type="checkbox" checked={useCustom} onChange={e => setUseCustom(e.target.checked)} />Add custom scenario</label>
          {useCustom && <div className="mt-2 grid grid-cols-2 gap-2">
            <Field label="Budget ×"><NumInput value={custom.budget_mult} step={0.1} onChange={v => setCustom({ ...custom, budget_mult: v })} /></Field><Field label="Effects ×"><NumInput value={custom.effect_mult} step={0.1} onChange={v => setCustom({ ...custom, effect_mult: v })} /></Field>
            <Field label="Costs ×"><NumInput value={custom.cost_mult} step={0.1} onChange={v => setCustom({ ...custom, cost_mult: v })} /></Field><Field label="Agent capacity ×"><NumInput value={custom.agent_mult} step={0.1} onChange={v => setCustom({ ...custom, agent_mult: v })} /></Field>
            <Field label="Baseline risk shift (logit)"><NumInput value={custom.econ_shift} step={0.1} onChange={v => setCustom({ ...custom, econ_shift: v })} /></Field></div>}
          <div className="mt-3"><Field label="Save comparison as (optional)"><TextInput value={name} onChange={setName} placeholder="Q4 stress set" /></Field></div>
          <div className="mt-3 flex gap-2"><Btn busy={run.busy} onClick={() => run.run()} disabled={sel.length === 0 && !useCustom}>Run scenarios</Btn></div>
          {run.error != null && <div className="mt-2"><ErrorBox error={run.error} /></div>}
          {saved.data?.length > 0 && <div className="mt-4"><h3 className="text-xs font-semibold">Saved scenario sets</h3>{saved.data.map((s: any) => <button key={s.scenario_id} className="block text-left text-xs text-accent underline" onClick={async () => setLoadedSaved((await api.get(`/scenarios/${s.scenario_id}`)).result)}>{s.name} ({s.n_results})</button>)}</div>}
        </Card>
        <div className="min-w-0 space-y-4">
          {!res && <Alert tone="info">Select scenarios and run. Each is re-optimised and compared with highest-risk-first, then 200 Monte Carlo draws propagate effect-estimation uncertainty through the chosen allocation.</Alert>}
          {res && <>
            <Card title="Net benefit by scenario" subtitle={`Optimised (error bars: Monte Carlo 5–95%) vs highest-risk-first. ${(loadedSaved ?? run.result).uncertainty}`} actions={<Btn variant="ghost" onClick={() => download('scenarios.json', JSON.stringify(loadedSaved ?? run.result, null, 2))}>Export</Btn>}>
              <div className="h-72"><ResponsiveContainer><BarChart data={bars} layout="vertical" margin={{ left: 0 }}><CartesianGrid horizontal={false} /><XAxis type="number" tickFormatter={v => v / 1000 + 'k'} /><YAxis type="category" dataKey="name" width={200} tick={{ fontSize: 10 }} />
                <Tooltip content={<ChartTooltip fmt={money} />} /><RLegend /><Bar dataKey="net" name="Optimised" fill="var(--s1)" barSize={9} radius={[0, 4, 4, 0]}><ErrorBar dataKey="hi" width={3} direction="x" stroke="var(--ink-2)" /></Bar>
                <Bar dataKey="rf" name="Highest-risk-first" fill="var(--s2)" barSize={9} radius={[0, 4, 4, 0]} /></BarChart></ResponsiveContainer></div>
            </Card>
            <Card title="Scenario table">
              <DataTable cols={[{ key: 'label', header: 'Scenario' }, { key: 'sel', header: 'Selected', align: 'right', render: (r: any) => r.optimised.n_selected.toLocaleString() }, { key: 'sp', header: 'Spend', align: 'right', render: (r: any) => money(r.optimised.budget_used) },
                { key: 'ev', header: 'Avoided events', align: 'right', render: (r: any) => fmt(r.optimised.expected_avoided_events, 1) }, { key: 'net', header: 'Net benefit', align: 'right', render: (r: any) => money(r.optimised.net_benefit) },
                { key: 'bcr', header: 'Benefit/cost', align: 'right', render: (r: any) => fmt(r.optimised.benefit_cost_ratio, 2) }, { key: 'mc', header: 'MC 5–95% (net)', align: 'right', render: (r: any) => `${money(r.monte_carlo.net_benefit_p05_p95?.[0])} – ${money(r.monte_carlo.net_benefit_p05_p95?.[1])}` },
                { key: 'p', header: 'P(net>0)', align: 'right', render: (r: any) => fmt(r.monte_carlo.prob_net_positive, 2) }, { key: 'adv', header: 'vs risk-first', align: 'right', render: (r: any) => money(r.net_benefit_advantage_over_risk_first) },
                { key: 'bind', header: 'Binding', render: (r: any) => r.optimised.binding_constraints.join(', ') || '—' }, { key: 'drv', header: 'Key drivers', render: (r: any) => r.drivers.join('; ') }]} rows={res} />
              <Note>Monte Carlo excludes unmeasured confounding, correlated errors and cost/value uncertainty. Economic-deterioration scenarios assume effects scale proportionally with baseline risk.</Note></Card></>}
          <Card title="Frontiers, stress tests & value of information" subtitle="Budget and capacity frontiers, fixed-policy stress tests, optimiser's-curse check, value of perfect information"
            actions={<Btn variant="ghost" busy={adv.busy} onClick={() => adv.run()}>Run advanced analyses (~30 s)</Btn>}>
            {adv.error != null && <ErrorBox error={adv.error} />}
            {a && <div className="min-w-0 space-y-4">
              <div className="grid gap-4 lg:grid-cols-2">
                <div><h3 className="text-xs font-semibold">Budget vs estimated net benefit</h3><div className="h-52"><ResponsiveContainer><LineChart data={a.budget_frontier.points}><CartesianGrid vertical={false} /><XAxis dataKey="budget" tickFormatter={v => v / 1000 + 'k'} /><YAxis tickFormatter={v => v / 1000 + 'k'} /><Tooltip content={<ChartTooltip fmt={money} />} />
                  <Line dataKey="net_benefit" name="Net benefit" stroke="var(--s1)" strokeWidth={2} dot={{ r: 2 }} /><Line dataKey="spend" name="Spend" stroke="var(--s2)" strokeWidth={2} dot={false} strokeDasharray="4 3" /><RLegend /></LineChart></ResponsiveContainer></div><Note>{a.budget_frontier.diminishing_returns_note}</Note></div>
                <div><h3 className="text-xs font-semibold">Agent capacity vs estimated net benefit</h3><div className="h-52"><ResponsiveContainer><LineChart data={a.capacity_frontier.points}><CartesianGrid vertical={false} /><XAxis dataKey="agent_capacity" /><YAxis tickFormatter={v => v / 1000 + 'k'} /><Tooltip content={<ChartTooltip fmt={money} />} />
                  <Line dataKey="net_benefit" name="Net benefit" stroke="var(--s3)" strokeWidth={2} dot={{ r: 2 }} /></LineChart></ResponsiveContainer></div></div></div>
              <div><h3 className="text-xs font-semibold">Stress test of fixed allocations (no re-optimisation)</h3>
                <DataTable cols={[{ key: 'stress', header: 'Stress' }, { key: 'optimised_net_benefit', header: 'Optimised', align: 'right', render: (r: any) => money(r.optimised_net_benefit) }, { key: 'conservative_net_benefit', header: 'Conservative', align: 'right', render: (r: any) => money(r.conservative_net_benefit) }, { key: 'highest_risk_first_net_benefit', header: 'Highest-risk-first', align: 'right', render: (r: any) => money(r.highest_risk_first_net_benefit) }]} rows={a.stress_test.rows} /><Note>{a.stress_test.note}</Note></div>
              <div><h3 className="text-xs font-semibold">Conservative vs unconstrained allocation: optimiser's curse</h3>
                <DataTable cols={[{ key: 'k', header: 'Policy' }, { key: 'c', header: 'Claimed net', align: 'right', render: (r: any) => money(r.mean_claimed_net_benefit) }, { key: 'r', header: 'Realised net', align: 'right', render: (r: any) => money(r.mean_realised_net_benefit) }, { key: 'g', header: 'Optimism gap', align: 'right', render: (r: any) => money(r.mean_optimism_gap) }]} rows={Object.entries(a.winners_curse.summary).map(([k, v]: any) => ({ k, ...v }))} />
                <Note>{a.winners_curse.interpretation} {a.conservative_objective}</Note></div>
              <Alert tone="info"><b>Value of information:</b> perfect knowledge of effects would add at most ≈ {money(a.value_of_information.evpi_total_unconstrained)} (EVPI upper bound, unconstrained); {pct(a.value_of_information.share_of_customers_with_positive_evpi, 0)} of customers have a non-zero value. {a.value_of_information.interpretation}</Alert>
            </div>}
          </Card>
        </div>
      </div>
    </>)
}
export default function Scenarios() { return <Gate><Body /></Gate> }
