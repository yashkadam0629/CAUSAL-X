import { Bar, BarChart, CartesianGrid, ErrorBar, Legend as RLegend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApp } from '../app/App'
import Gate from '../components/Gate'
import { Alert, Btn, Card, ChartTooltip, DataTable, ErrorBox, Note, PageHeader, Stat } from '../components/ui'
import { useAction } from '../hooks/useAsync'
import { api } from '../services/api'
import { fmt, pct } from '../utils/format'

function Body() {
  const { current } = useApp()
  const run = useAction(() => api.post('/policies/evaluate', { dataset_id: current!.id }))
  const r = run.result
  const rows = r?.policies?.filter((p: any) => p.strategy !== 'No intervention') ?? []
  const hasTruth = rows.some((p: any) => p.synthetic_truth_events_prevented !== null && p.synthetic_truth_events_prevented !== undefined)
  const chart = rows.map((p: any) => ({ name: p.strategy.replace(/ \(.*\)/, ''), dr: p.dr_estimated_events_prevented, model: p.model_estimated_events_prevented, truth: p.synthetic_truth_events_prevented ?? undefined, err: (p.dr_ci[1] - p.dr_ci[0]) / 2 }))
  return (
    <>
      <PageHeader title="Policy Evaluation" subtitle="How would each policy have performed on cohorts that no model was trained on? Doubly robust off-policy evaluation on logged outcomes, kept separate from the model-estimated benefit the optimiser used.">
        <Btn busy={run.busy} onClick={() => run.run()}>Evaluate all policies</Btn>
      </PageHeader>
      {run.error != null && <ErrorBox error={run.error} />}
      {!r && !run.busy && <Alert tone="info">Evaluation uses held-out cohorts 10–11. Policies were optimised from predictions for these customers but never saw their outcomes.</Alert>}
      {r && (
        <div className="space-y-4">
          <div className="grid gap-3 md:grid-cols-4">
            <Stat label="Evidence type" value={r.evidence_type.includes('OBSERVATIONAL') ? 'Observational OPE' : 'Logged-randomisation OPE'} hint={r.propensity_source} />
            <Stat label="Evaluated customers" value={r.n.toLocaleString()} hint={`cohorts ${r.evaluation_cohorts.join(', ')}`} />
            <Stat label="Training / validation" value="0-7 / 8-9" hint="models fitted & selected here" />
            <Stat label="Evaluation" value="10-11" hint="untouched chronological holdout" />
          </div>
          <Card title="Events prevented vs no intervention" subtitle="DR estimate with 95% interval; model estimate used by the optimiser; synthetic truth shown only because the data are synthetic.">
            <div className="h-72"><ResponsiveContainer><BarChart data={chart} layout="vertical" margin={{ left: 0 }}><CartesianGrid horizontal={false} /><XAxis type="number" /><YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 10 }} /><Tooltip content={<ChartTooltip fmt={(v: number) => fmt(v, 1)} />} /><RLegend />
              <Bar dataKey="model" name="Model-estimated" fill="var(--s1)" barSize={8} radius={[0, 4, 4, 0]} />
              <Bar dataKey="dr" name="Off-policy (DR) estimate" fill="var(--s2)" barSize={8} radius={[0, 4, 4, 0]}><ErrorBar dataKey="err" width={3} direction="x" stroke="var(--ink-2)" /></Bar>
              {hasTruth && <Bar dataKey="truth" name="Synthetic truth" fill="var(--s3)" barSize={8} radius={[0, 4, 4, 0]} />}</BarChart></ResponsiveContainer></div>
          </Card>
          <Card title="Policy table">
            <DataTable cols={[{ key: 'strategy', header: 'Policy' }, { key: 'n_selected', header: 'Selected', align: 'right', render: (p: any) => Math.round(p.n_selected).toLocaleString() }, { key: 'budget_used', header: 'Spend', align: 'right', render: (p: any) => Math.round(p.budget_used).toLocaleString() },
              { key: 'm', header: 'Model est.', align: 'right', render: (p: any) => fmt(p.model_estimated_events_prevented, 1) }, { key: 'd', header: 'DR est. [95% CI]', align: 'right', render: (p: any) => `${fmt(p.dr_estimated_events_prevented, 1)} [${fmt(p.dr_ci[0], 0)}, ${fmt(p.dr_ci[1], 0)}]` },
              { key: 't', header: 'Synthetic truth', align: 'right', render: (p: any) => (p.synthetic_truth_events_prevented != null ? fmt(p.synthetic_truth_events_prevented, 1) : '—') }, { key: 'e', header: 'DR error vs truth', align: 'right', render: (p: any) => (p.dr_error_vs_truth != null ? fmt(p.dr_error_vs_truth, 1) : '—') },
              { key: 'c', header: 'Clipped weights', align: 'right', render: (p: any) => pct(p.clipped_share) }]} rows={r.policies} />
            <Note>DR intervals reflect sampling variance only. Policies that choose actions rarely logged for similar customers have high variance (see clipped weights).</Note></Card>
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Assumptions"><ul className="list-disc space-y-1 pl-5 text-xs text-ink2">{r.assumptions.map((a: string) => <li key={a}>{a}</li>)}</ul></Card>
            <Card title="Separation of data roles & limitations"><ul className="list-disc space-y-1 pl-5 text-xs text-ink2">{Object.entries(r.separation).map(([k, v]) => <li key={k}><b>{k.replace(/_/g, ' ')}</b>: {v as string}</li>)}{r.limitations.map((l: string) => <li key={l}>{l}</li>)}</ul></Card>
          </div>
          <Alert tone="warn">{r.evidence_type}. This is not a randomised trial result of these policies. An experiment on a randomised holdout is the only way to verify them.</Alert>
        </div>)}
    </>)
}
export default function PolicyEval() { return <Gate><Body /></Gate> }
