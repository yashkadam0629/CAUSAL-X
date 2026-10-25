import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend as RLegend, ReferenceLine, ComposedChart } from 'recharts'
import { useApp } from '../app/App'
import Gate from '../components/Gate'
import { Alert, Card, ChartTooltip, DataTable, ErrorBox, Note, PageHeader, Spinner, Tabs } from '../components/ui'
import { useLoad } from '../hooks/useAsync'
import { api } from '../services/api'
import { ARM_LABEL, fmt, label } from '../utils/format'

function Body() {
  const { current } = useApp()
  const [tab, setTab] = useState('risk')
  const { data, error, loading } = useLoad(() => api.get(`/models/overview/${current!.id}`), [current?.id])
  const exps = useLoad(() => api.get('/causal/experiments'), [])
  const models = useLoad(() => api.get('/models', { dataset_id: current!.id }), [current?.id])
  if (loading) return <Spinner />
  if (error) return <ErrorBox error={error} />
  const d = data, rk = d.risk
  const cal = rk.metrics.test.calibration.map((c: any) => ({ ...c, perfect: c.mean_predicted }))
  const imp = rk.importance.slice(0, 12)
  return (
    <>
      <PageHeader title="Model Laboratory" subtitle={`Model version ${d.model_version}. ${d.metric_note}`} />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: 'risk', label: 'Predictive risk models' }, { id: 'cate', label: 'Treatment-effect (CATE) models' }, { id: 'registry', label: 'Registry & experiments' }]} />
      {tab === 'risk' && <div className="space-y-4">
        <Alert tone="info">Selected: <b>{rk.chosen}</b> — {rk.config.selection_rule}. Splits: train {rk.config.split.train_cohorts}, validation {rk.config.split.validation_cohorts}, test {rk.config.split.test_cohorts} (chronological). Preprocessing is fitted on training rows only.</Alert>
        <Card title="Model comparison (test = untouched cohorts 10-11)">
          <DataTable cols={[{ key: 'model', header: 'Model' }, { key: 'vb', header: 'Val Brier', align: 'right', render: (c: any) => fmt(c.validation.brier, 4) }, { key: 'auc', header: 'Test ROC-AUC', align: 'right', render: (c: any) => fmt(c.test.roc_auc, 3) }, { key: 'pr', header: 'Test PR-AUC', align: 'right', render: (c: any) => fmt(c.test.pr_auc, 3) },
            { key: 'b', header: 'Test Brier', align: 'right', render: (c: any) => fmt(c.test.brier, 4) }, { key: 'll', header: 'Log loss', align: 'right', render: (c: any) => fmt(c.test.log_loss, 3) }, { key: 'p', header: 'Precision', align: 'right', render: (c: any) => fmt(c.test.precision, 3) }, { key: 'r', header: 'Recall', align: 'right', render: (c: any) => fmt(c.test.recall, 3) },
            { key: 'f', header: 'F1', align: 'right', render: (c: any) => fmt(c.test.f1, 3) }, { key: 'th', header: 'Threshold', align: 'right', render: (c: any) => fmt(c.test.threshold, 2) }, { key: 'ec', header: 'Expected cost', align: 'right', render: (c: any) => Math.round(c.test.expected_cost).toLocaleString() }]} rows={rk.comparison} />
          <Note>Winner chosen on validation Brier (calibrated probabilities matter for financial decisions), not accuracy. Threshold minimises expected cost on validation with FN cost {rk.config.fn_cost} and FP cost {rk.config.fp_cost} (analyst assumptions).</Note></Card>
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="Calibration (test)" subtitle="Mean predicted vs observed rate by decile"><div className="h-64"><ResponsiveContainer><ComposedChart data={cal}><CartesianGrid /><XAxis dataKey="mean_predicted" tickFormatter={v => fmt(v, 2)} /><YAxis tickFormatter={v => fmt(v, 2)} /><Tooltip content={<ChartTooltip fmt={(v: number) => fmt(v, 3)} />} /><RLegend />
            <Line dataKey="perfect" name="Perfect calibration" stroke="var(--muted)" strokeDasharray="4 3" dot={false} /><Line dataKey="observed_rate" name="Observed" stroke="var(--s1)" strokeWidth={2} dot={{ r: 3 }} /></ComposedChart></ResponsiveContainer></div></Card>
          <Card title="Feature importance (mean |SHAP|)" subtitle="Describes the prediction, not causal effects"><div className="h-64"><ResponsiveContainer><BarChart data={imp} layout="vertical" margin={{ left: 0 }}><CartesianGrid horizontal={false} /><XAxis type="number" /><YAxis type="category" dataKey="feature" width={170} tick={{ fontSize: 10 }} /><Tooltip content={<ChartTooltip fmt={(v: number) => fmt(v, 3)} />} /><Bar dataKey="mean_abs_shap" name="mean |SHAP|" fill="var(--s1)" barSize={9} radius={[0, 4, 4, 0]} /></BarChart></ResponsiveContainer></div></Card>
        </div>
        <Card title="Feature groups & provenance"><div className="grid gap-3 md:grid-cols-4">{Object.entries(d.feature_groups).map(([g, fs]) => <div key={g}><h3 className="text-xs font-semibold">{label(g)}</h3><ul className="text-[11px] text-ink2">{(fs as string[]).map(f => <li key={f}>{f}</li>)}</ul></div>)}</div><Note>Excluded from all models and the optimiser by policy: age_band (audit-only), region_group (coverage / audit only), and all outcome or post-treatment fields.</Note></Card>
      </div>}
      {tab === 'cate' && <Card title="Treatment-effect models (DR-learner, cross-fitted)" subtitle="Metrics here are uplift metrics (see Uplift Intelligence); they are not comparable with the predictive metrics above.">
        <DataTable cols={[{ key: 'arm', header: 'Intervention', render: (r: any) => ARM_LABEL[r.arm] }, { key: 'n_treated', header: 'Treated (train)', align: 'right' }, { key: 'n_comparison', header: 'Comparison (train)', align: 'right' }, { key: 'ate', header: 'Pseudo-outcome ATE', align: 'right', render: (r: any) => fmt(r.ate_from_pseudo_outcomes, 4) }, { key: 'bootstrap_B', header: 'Bootstrap B', align: 'right' }, { key: 'features', header: 'Features', align: 'right' }]}
          rows={Object.entries(d.cate).map(([arm, v]: any) => ({ arm, ...v }))} />
        <Note>{Object.values(d.cate)[0] && (Object.values(d.cate)[0] as any).method}. {Object.values(d.cate)[0] && (Object.values(d.cate)[0] as any).uncertainty_note}</Note></Card>}
      {tab === 'registry' && <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Model registry"><DataTable cols={[{ key: 'model_id', header: 'Model ID' }, { key: 'kind', header: 'Kind' }, { key: 'name', header: 'Name' }, { key: 'seed', header: 'Seed', align: 'right' }, { key: 'created_at', header: 'Created', render: (r: any) => String(r.created_at).slice(0, 16) }]} rows={models.data ?? []} empty="Models register after a training job." /></Card>
        <Card title="Saved experiments"><DataTable cols={[{ key: 'name', header: 'Name' }, { key: 'kind', header: 'Kind' }, { key: 'hash', header: 'Manifest hash', render: (r: any) => (r.manifest?.manifest_hash ?? '').slice(0, 10) }, { key: 'created_at', header: 'Saved', render: (r: any) => String(r.created_at).slice(0, 16) }]} rows={exps.data ?? []} empty="No saved experiments yet — save one from the Causal Lab." /></Card>
        <Card title="Pipeline timings (s)" className="lg:col-span-2"><Note>Reproducibility: seed {d.risk.config.seed}; feature list and dataset fingerprint are hashed into the model version.</Note></Card>
      </div>}
    </>)
}
export default function ModelLab() { return <Gate><Body /></Gate> }
