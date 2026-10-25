import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend as RLegend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { useApp } from '../app/App'
import Gate from '../components/Gate'
import { Alert, Btn, Card, ChartTooltip, DataTable, ErrorBox, Field, NumInput, Note, PageHeader, Select, Stat, Tag, TextInput } from '../components/ui'
import { useAction, useLoad } from '../hooks/useAsync'
import { api } from '../services/api'
import { ARM_LABEL, download, fmt, label } from '../utils/format'

const EST = [['aipw', 'AIPW (doubly robust)'], ['ipw', 'Inverse probability weighting'], ['regression_adjustment', 'Regression adjustment'], ['difference_in_means', 'Raw difference in means']]
const OUTCOMES = ['delinquency_30d', 'missed_payment', 'delinquency_60d', 'default_event']

function Body() {
  const { current } = useApp()
  const meta = useLoad(() => api.get('/metadata'), [])
  const [f, setF] = useState<any>({ treatment: 'standard_reminder', comparison: 'none', outcome: 'delinquency_30d', estimator: 'aipw', subgroup: '', trim_low: undefined, bootstrap: 60 })
  const [groups, setGroups] = useState<Record<string, boolean>>({})
  const [expName, setExpName] = useState('')
  const [saved, setSaved] = useState<string | null>(null)
  const isUpload = current!.source === 'upload'
  const groupsOn = (meta.data?.feature_groups ? Object.keys(meta.data.feature_groups) : []).filter(g => groups[g] !== false)
  const covariates = isUpload ? undefined : groupsOn.flatMap(g => meta.data.feature_groups[g])
  const body = () => ({ dataset_id: current!.id, ...f, subgroup: f.subgroup || undefined, covariates })
  const run = useAction(async () => {
    const [est, cmp] = await Promise.all([api.post('/causal/estimate', body()), api.post('/causal/compare', { ...body(), bootstrap: 20 })])
    return { est, cmp }
  })
  const sens = useAction(() => api.post('/causal/sensitivity-analysis', body()))
  const r = run.result?.est, cmp = run.result?.cmp
  const arms = ['none', 'standard_reminder', 'counselling', 'repayment_plan']
  const armOpts = isUpload ? [] : arms.map(a => ({ value: a, label: ARM_LABEL[a] }))
  const d = r?.diagnostics
  const love = d ? [...d.smd].sort((a: any, b: any) => Math.abs(b.smd_unadjusted) - Math.abs(a.smd_unadjusted)).slice(0, 14).map((s: any) => ({ name: s.covariate, before: Math.abs(s.smd_unadjusted), after: Math.abs(s.smd_weighted) })) : []
  const ov = d ? d.overlap_histogram.bins.map((b: number, i: number) => ({ bin: b.toFixed(2), treated: d.overlap_histogram.treated[i], comparison: d.overlap_histogram.comparison[i] })) : []
  const ests = cmp?.estimators?.filter((e: any) => e.status === 'ok').map((e: any) => ({ name: label(e.estimator), ate: e.ate, lo: e.ate - e.ci_low, hi: e.ci_high - e.ate })) ?? []
  const truth = r?.synthetic_benchmark?.true_ate_itt
  return (
    <>
      <PageHeader title="Causal Lab" subtitle="Define the causal question explicitly, estimate with several estimators, inspect overlap and balance, then stress the conclusion. Estimates of observational data are conditional on untestable assumptions." />
      <div className="grid gap-4 xl:grid-cols-[320px_1fr]">
        <Card title="Causal question">
          <div className="space-y-3">
            <Field label="Treatment"><Select value={f.treatment} onChange={v => setF({ ...f, treatment: v })} options={isUpload ? [{ value: f.treatment, label: f.treatment }] : armOpts.filter(a => a.value !== 'none')} /></Field>
            {isUpload && <TextInput value={f.treatment} onChange={v => setF({ ...f, treatment: v })} placeholder="treatment label in the data" />}
            <Field label="Comparison condition">{isUpload ? <TextInput value={f.comparison} onChange={v => setF({ ...f, comparison: v })} /> : <Select value={f.comparison} onChange={v => setF({ ...f, comparison: v })} options={armOpts.filter(a => a.value !== f.treatment)} />}</Field>
            <Field label="Outcome (lower is better)" hint="Horizon: 30 days after the decision date (time zero)">{isUpload ? <TextInput value={f.outcome} onChange={v => setF({ ...f, outcome: v })} /> : <Select value={f.outcome} onChange={v => setF({ ...f, outcome: v })} options={OUTCOMES.map(o => ({ value: o, label: label(o) }))} />}</Field>
            <Field label="Estimator"><Select value={f.estimator} onChange={v => setF({ ...f, estimator: v })} options={EST.map(([v, l]) => ({ value: v, label: l }))} /></Field>
            {!isUpload && <>
              <Field label="Population"><Select value={f.subgroup} onChange={v => setF({ ...f, subgroup: v })} options={[{ value: '', label: 'All eligible' }, { value: 'high_baseline_risk', label: 'High baseline risk (top third)' }, { value: 'low_baseline_risk', label: 'Low baseline risk (bottom third)' }]} /></Field>
              <fieldset><legend className="mb-1 text-xs font-medium text-ink2">Pre-treatment covariate groups</legend>
                {Object.keys(meta.data?.feature_groups ?? {}).map(g => <label key={g} className="mr-3 inline-flex items-center gap-1 text-xs"><input type="checkbox" checked={groups[g] !== false} onChange={e => setGroups({ ...groups, [g]: e.target.checked })} />{label(g)}</label>)}
                <Note>Outcome and post-treatment variables are blocked. Age band and region are excluded from the model by policy.</Note></fieldset>
              <Field label="Overlap trimming (propensity ≥ x and ≤ 1−x)" hint="Changes the estimand to the trimmed population"><NumInput value={f.trim_low} onChange={v => setF({ ...f, trim_low: v })} step={0.01} min={0} max={0.4} /></Field></>}
            <Btn busy={run.busy} onClick={() => run.run()}>Run analysis</Btn>
            {run.error != null && <ErrorBox error={run.error} />}
          </div>
        </Card>
        <div className="min-w-0 space-y-4">
          {!r && !run.busy && <Alert tone="info">Pick a treatment contrast and run. The result shows the estimate, its interval, diagnostics, and why it may — or may not — be causally interpretable.</Alert>}
          {r && r.status !== 'ok' && <Alert>Insufficient evidence: {r.reason}</Alert>}
          {r?.status === 'ok' && (
            <>
              <div className="grid gap-3 md:grid-cols-4">
                <Stat label={`ATE (${label(r.estimator)})`} value={fmt(r.ate, 4)} kind="estimated" hint="Δ in adverse-outcome probability; negative = beneficial" />
                <Stat label="95% interval" value={`[${fmt(r.ci_low, 3)}, ${fmt(r.ci_high, 3)}]`} hint={r.se_method ?? 'Wald'} />
                <Stat label="ATT" value={r.att !== undefined ? fmt(r.att, 4) : 'n/a'} hint={r.att_note ?? 'Effect among those assigned'} kind="estimated" />
                <Stat label="Sample" value={r.n.toLocaleString()} hint={`${d.n_treated} treated / ${d.n_comparison} comparison; ESS ${fmt(d.ess_treated, 0)} / ${fmt(d.ess_comparison, 0)}`} />
              </div>
              {r.synthetic_benchmark && <Alert tone="info"><b>SYNTHETIC BENCHMARK</b> (ground truth, never a model input): true ITT effect {fmt(r.synthetic_benchmark.true_ate_itt, 4)}; estimation error {fmt(r.synthetic_benchmark.estimation_error, 4)}; truth {r.synthetic_benchmark.covered_by_ci ? 'is' : 'is NOT'} inside the interval.</Alert>}
              <Card title="Can this be read causally?" subtitle={`Status: ${label(r.interpretability.status)}`}>
                <ul className="list-disc space-y-1 pl-5 text-xs text-ink2">{r.interpretability.reasons.map((x: string) => <li key={x}>{x}</li>)}</ul>
                <p className="mt-2 text-xs font-medium">What would strengthen it</p><ul className="list-disc pl-5 text-xs text-ink2">{r.interpretability.what_would_strengthen_it.map((x: string) => <li key={x}>{x}</li>)}</ul>
              </Card>
              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="Estimator comparison" subtitle="Point estimate with 95% interval (20-draw bootstrap here for speed)"><div className="h-60">
                  <ResponsiveContainer><BarChart data={ests} layout="vertical" margin={{ left: 0 }}><CartesianGrid horizontal={false} /><XAxis type="number" /><YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 10 }} />
                    <Tooltip content={<ChartTooltip fmt={(v: number) => fmt(v, 4)} />} /><ReferenceLine x={0} stroke="var(--axis)" />{truth !== undefined && <ReferenceLine x={truth} stroke="var(--s3)" strokeDasharray="4 3" label={{ value: 'truth', fontSize: 10 }} />}
                    <Bar dataKey="ate" name="ATE" fill="var(--s1)" barSize={10} radius={[0, 4, 4, 0]}><ErrorBar dataKey="hi" width={4} stroke="var(--ink-2)" direction="x" /></Bar></BarChart></ResponsiveContainer></div>
                  <Note>Agreement between estimators does not prove validity: they share the same exchangeability assumption. Dashed line = synthetic truth.</Note></Card>
                <Card title="Covariate balance (Love plot)" subtitle="|Standardised mean difference|; 0.10 is a convention, not proof"><div className="h-60">
                  <ResponsiveContainer><BarChart data={love} layout="vertical" margin={{ left: 0 }}><CartesianGrid horizontal={false} /><XAxis type="number" /><YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 9 }} />
                    <Tooltip content={<ChartTooltip fmt={(v: number) => fmt(v, 3)} />} /><RLegend /><ReferenceLine x={0.1} stroke="var(--s8)" strokeDasharray="3 3" />
                    <Bar dataKey="before" name="Unadjusted" fill="var(--s2)" barSize={5} /><Bar dataKey="after" name="After weighting" fill="var(--s1)" barSize={5} /></BarChart></ResponsiveContainer></div></Card>
                <Card title="Propensity-score overlap" subtitle={`Share outside [0.05, 0.95]: ${fmt(d['share_outside_0.05_0.95'] * 100, 1)}% · max weight ${fmt(d.max_weight, 1)}`}><div className="h-52">
                  <ResponsiveContainer><BarChart data={ov}><CartesianGrid vertical={false} /><XAxis dataKey="bin" interval={3} /><YAxis /><Tooltip content={<ChartTooltip />} /><RLegend />
                    <Bar dataKey="treated" name="Treatment" fill="var(--s1)" /><Bar dataKey="comparison" name="Comparison" fill="var(--s2)" /></BarChart></ResponsiveContainer></div></Card>
                <Card title="Population & diagnostics">
                  <ul className="space-y-1 text-xs text-ink2">
                    <li>Rule: {r.population.rule}</li><li>Rows: {r.population.rows_in_two_arms} in two arms → {r.population.after_eligibility} eligible → {r.population.final} analysed ({r.population.trimmed_out} trimmed, {r.population.ps_clipped} propensities clipped at {r.spec.clip_ps})</li>
                    <li>Raw outcome rate: treatment {fmt(d.raw_outcome_means.treatment * 100, 1)}% vs comparison {fmt(d.raw_outcome_means.comparison * 100, 1)}%</li>
                    <li>Max |SMD| before {fmt(d.max_abs_smd_unadjusted, 2)} → after {fmt(d.max_abs_smd_weighted, 2)}</li></ul>
                  <div className="mt-2 space-y-1">{r.warnings.map((w: string) => <Alert key={w}>{w}</Alert>)}</div></Card>
              </div>
              <Card title="Sensitivity analysis" subtitle="Robustness checks (AIPW reference). They probe, but cannot prove, the identification assumptions." actions={<Btn variant="ghost" busy={sens.busy} onClick={() => sens.run()}>Run sensitivity analysis</Btn>}>
                {sens.error != null && <ErrorBox error={sens.error} />}
                {sens.result && <Sens s={sens.result} />}
                {!sens.result && <Note>Covariate specifications, alternative propensity models, overlap restriction, placebo outcome, alternative outcomes and E-value.</Note>}
              </Card>
              <Card title="Save & export">
                <div className="flex flex-wrap items-end gap-2">
                  <div className="w-64"><Field label="Experiment name"><TextInput value={expName} onChange={setExpName} placeholder="Reminder vs none (AIPW)" /></Field></div>
                  <Btn variant="ghost" disabled={expName.length < 2} onClick={async () => { const x = await api.post('/causal/experiments', { dataset_id: current!.id, name: expName, config: body(), results: { ate: r.ate, ci: [r.ci_low, r.ci_high], n: r.n, estimator: r.estimator } }); setSaved(x.manifest_hash.slice(0, 12)) }}>Save configuration</Btn>
                  <Btn variant="ghost" onClick={() => download('causal_result.json', JSON.stringify(r, null, 2))}>Export JSON</Btn>
                  {saved && <span className="text-xs text-[var(--good-ink)]">Saved · manifest hash {saved}</span>}
                </div>
              </Card>
            </>)}
        </div>
      </div>
    </>)
}

function Sens({ s }: { s: any }) {
  const rows = (items: any[]) => <DataTable cols={[{ key: 'label', header: 'Specification' }, { key: 'ate', header: 'ATE', align: 'right', render: (x: any) => (x.status === 'ok' ? fmt(x.ate, 4) : x.reason || x.status) },
    { key: 'ci', header: '95% CI', align: 'right', render: (x: any) => (x.status === 'ok' ? `[${fmt(x.ci_low, 3)}, ${fmt(x.ci_high, 3)}]` : '') }, { key: 'n', header: 'n', align: 'right' }]} rows={items} />
  const p = s.placebo_outcome
  return (
    <div className="space-y-3">
      <Alert tone={s.stability_indicator === 'stable' ? 'ok' : 'warn'}>Stability indicator: <b>{s.stability_indicator}</b> (spread of ATE across specifications {fmt(s.spread_of_ate_across_specs, 4)}).</Alert>
      <div><h3 className="text-xs font-semibold">Alternative covariate specifications</h3>{rows(s.covariate_specifications)}</div>
      <div><h3 className="text-xs font-semibold">Alternative propensity models</h3>{rows(s.propensity_models)}</div>
      <div><h3 className="text-xs font-semibold">Overlap restriction (changes estimand)</h3>{rows(s.overlap_restriction)}</div>
      <div><h3 className="text-xs font-semibold">Alternative outcome definitions</h3>{rows(s.alternative_outcomes)}</div>
      <div><h3 className="text-xs font-semibold">Placebo outcome (pre-treatment)</h3>
        {p.status === 'ok' ? <Alert tone={p.flags_confounding ? 'warn' : 'ok'}>Estimate {fmt(p.ate, 4)} [{fmt(p.ci_low, 3)}, {fmt(p.ci_high, 3)}] — {p.flags_confounding ? 'non-zero: suggests residual confounding.' : 'consistent with zero.'} {p.interpretation}</Alert> : <Note>{p.reason}</Note>}</div>
      {s.e_value && <div><h3 className="text-xs font-semibold">E-value for unmeasured confounding</h3><p className="text-xs text-ink2">Risk ratio {fmt(s.e_value.risk_ratio, 3)}; E-value (point) {fmt(s.e_value.e_value_point, 2)}, (CI limit) {fmt(s.e_value.e_value_ci_limit, 2)}. {s.e_value.interpretation}</p></div>}
    </div>)
}
export default function CausalLab() { return <Gate needModels={false}><Body /></Gate> }
