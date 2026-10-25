import { useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend as RLegend } from 'recharts'
import { useApp } from '../app/App'
import Gate from '../components/Gate'
import { Alert, Btn, Card, ChartTooltip, DataTable, ErrorBox, Field, Modal, Note, PageHeader, Pager, Select, Spinner, Tabs, TextInput } from '../components/ui'
import { useAction, useLoad } from '../hooks/useAsync'
import { api } from '../services/api'
import { ARM_LABEL, fmt, money, pct } from '../utils/format'

function Decisions() {
  const { current, lastRunId } = useApp()
  const [page, setPage] = useState(1)
  const [status, setStatus] = useState('')
  const [q, setQ] = useState('')
  const [sel, setSel] = useState<any>(null)
  const [rv, setRv] = useState<any>({ reviewer: '', outcome: 'approve', final_action: 'none', reason: '' })
  const { data, error, loading, reload } = useLoad(() => api.get('/audit/decisions', { dataset_id: current!.id, page, page_size: 15, status, customer_id: q }), [current?.id, page, status, q])
  const submit = useAction(async () => { await api.post(`/audit/decisions/${sel.decision_id}/review`, { ...rv, final_action: rv.outcome === 'override' ? rv.final_action : undefined }); setSel(null); reload() })
  return (
    <Card title="Recommendation history" subtitle="Each record stores model, policy and data versions plus its explanation. Reviews keep the original recommendation and the final decision."
      actions={<a className="text-xs text-accent underline" href={`/api/v1/audit/decisions/export?dataset_id=${current!.id}`}>Export CSV</a>}>
      <div className="mb-3 grid grid-cols-3 gap-3"><Field label="Status"><Select value={status} onChange={v => { setPage(1); setStatus(v) }} options={[{ value: '', label: 'All' }, ...['pending_review', 'not_required', 'approved', 'overridden', 'rejected'].map(s => ({ value: s, label: s.replace('_', ' ') }))]} /></Field>
        <Field label="Customer ID contains"><TextInput value={q} onChange={v => { setPage(1); setQ(v) }} /></Field></div>
      {loading && !data ? <Spinner /> : error ? <ErrorBox error={error} /> : <>
        <DataTable onRowClick={r => setSel(r)} empty="No recommendations yet. Run and save an optimisation to create auditable decisions." cols={[{ key: 'customer_id', header: 'Customer' }, { key: 'recommended_action', header: 'Recommended', render: (r: any) => ARM_LABEL[r.recommended_action] }, { key: 'net_benefit', header: 'Net (est.)', align: 'right', render: (r: any) => money(r.net_benefit) },
          { key: 'model_version', header: 'Model' }, { key: 'policy_version', header: 'Policy' }, { key: 'data_version', header: 'Data' }, { key: 'status', header: 'Review', render: (r: any) => r.status.replace('_', ' ') },
          { key: 'ov', header: 'Final decision', render: (r: any) => (r.reviews.length ? `${ARM_LABEL[r.reviews.at(-1).final_action] ?? r.reviews.at(-1).final_action} by ${r.reviews.at(-1).reviewer}` : '—') }, { key: 'timestamp', header: 'Time', render: (r: any) => String(r.timestamp).slice(0, 16) }]} rows={data.items} />
        <Pager page={page} total={data.total} size={15} onPage={setPage} /></>}
      <Modal open={!!sel} onClose={() => setSel(null)} title={`Decision ${sel?.decision_id ?? ''}`}>
        {sel && <div className="space-y-3 text-xs">
          <p><b>{ARM_LABEL[sel.recommended_action]}</b> — {sel.reason}</p>
          {sel.human_review_required && <Alert>{sel.explanation.human_review_reasons.map((r: string) => <div key={r}>• {r}</div>)}</Alert>}
          <DataTable cols={[{ key: 'label', header: 'Candidate' }, { key: 'e', header: 'Eligible', render: (c: any) => (c.eligible ? 'yes' : 'no') }, { key: 'a', header: 'ARR', align: 'right', render: (c: any) => fmt(c.estimated_absolute_risk_reduction.value, 3) }, { key: 'n', header: 'Net', align: 'right', render: (c: any) => money(c.estimated_net_benefit.value) }]} rows={sel.explanation.candidate_interventions} />
          {sel.reviews.length > 0 && <div><h3 className="font-semibold">Review trail</h3>{sel.reviews.map((v: any, i: number) => <p key={i}>{String(v.reviewed_at).slice(0, 16)} · {v.reviewer}: {v.outcome} ({v.original_action} → {v.final_action}) — {v.reason}</p>)}</div>}
          <div className="rounded-md border border-line p-3"><h3 className="mb-2 font-semibold">Human review</h3>
            <div className="grid grid-cols-2 gap-3"><Field label="Reviewer"><TextInput value={rv.reviewer} onChange={v => setRv({ ...rv, reviewer: v })} /></Field>
              <Field label="Outcome"><Select value={rv.outcome} onChange={v => setRv({ ...rv, outcome: v })} options={[{ value: 'approve', label: 'Approve' }, { value: 'override', label: 'Override with another action' }, { value: 'reject', label: 'Reject (no intervention)' }]} /></Field>
              {rv.outcome === 'override' && <Field label="Final action (must be eligible)"><Select value={rv.final_action} onChange={v => setRv({ ...rv, final_action: v })} options={['none', 'standard_reminder', 'counselling', 'repayment_plan'].map(a => ({ value: a, label: ARM_LABEL[a] }))} /></Field>}
              <div className="col-span-2"><Field label="Reason (required)"><TextInput value={rv.reason} onChange={v => setRv({ ...rv, reason: v })} /></Field></div></div>
            <div className="mt-2"><Btn busy={submit.busy} disabled={rv.reviewer.length < 2 || rv.reason.length < 3} onClick={() => submit.run()}>Record decision</Btn></div>{submit.error != null && <div className="mt-2"><ErrorBox error={submit.error} /></div>}</div>
        </div>}
      </Modal>
    </Card>)
}

function Monitoring() {
  const { current } = useApp()
  const { data, error, loading } = useLoad(async () => ({ sum: await api.get('/monitoring/summary', { dataset_id: current!.id }), drift: await api.get('/monitoring/drift', { dataset_id: current!.id }) }), [current?.id])
  if (loading) return <Spinner />
  if (error) return <ErrorBox error={error} />
  const { sum, drift } = data!
  return (
    <div className="space-y-4">
      {sum.warnings.length === 0 ? <Alert tone="ok">No drift warnings at the configured thresholds.</Alert> : sum.warnings.map((w: string) => <Alert key={w}>{w}</Alert>)}
      <Card title="Recommended investigation"><ul className="list-disc pl-5 text-xs text-ink2">{sum.recommended_investigation.map((r: string) => <li key={r}>{r}</li>)}</ul><Note>Models are never retrained or redeployed automatically; changes need an approved workflow.</Note></Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Feature drift (PSI, cohorts 0-7 vs 10-11)" subtitle="Feature drift ≠ prediction drift ≠ concept drift ≠ treatment-effect drift">
          <DataTable cols={[{ key: 'feature', header: 'Feature' }, { key: 'psi', header: 'PSI', align: 'right', render: (f: any) => fmt(f.psi, 3) }, { key: 'status', header: 'Status' }, { key: 'ks', header: 'KS p', align: 'right', render: (f: any) => (f.ks_pvalue != null ? f.ks_pvalue.toExponential(1) : '—') }, { key: 'mc', header: 'Missing ref→cur', align: 'right', render: (f: any) => `${pct(f.missing_ref)}→${pct(f.missing_cur)}` }]} rows={drift.feature_drift.slice(0, 10)} />
          <Note>Prediction (risk-score) PSI: {fmt(drift.prediction_psi, 3)} ({drift.prediction_status}). Thresholds 0.10 / 0.25 are conventions.</Note></Card>
        <Card title="Treatment-effect stability (reminder vs none)" subtitle="AIPW ATE by period with 95% CI"><DataTable cols={[{ key: 'period', header: 'Period' }, { key: 'ate', header: 'ATE', align: 'right', render: (r: any) => fmt(r.ate, 4) }, { key: 'ci', header: '95% CI', align: 'right', render: (r: any) => `[${fmt(r.ci_low, 3)}, ${fmt(r.ci_high, 3)}]` }, { key: 'n', header: 'n', align: 'right' }]} rows={sum.effect_stability} /><Note>{sum.effect_stability_note}</Note></Card>
        <Card title="Outcome, calibration and treatment mix by cohort" className="lg:col-span-2"><div className="h-60"><ResponsiveContainer><LineChart data={drift.monthly}><CartesianGrid vertical={false} /><XAxis dataKey="cohort_month" /><YAxis tickFormatter={v => pct(v, 0)} /><Tooltip content={<ChartTooltip fmt={(v: number) => pct(v)} />} /><RLegend />
          <Line dataKey="outcome_rate" name="Observed rate" stroke="var(--s1)" strokeWidth={2} dot={false} /><Line dataKey="mean_predicted_risk" name="Mean predicted" stroke="var(--s2)" strokeWidth={2} dot={false} /><Line dataKey="treated_share" name="Treated share" stroke="var(--s3)" strokeWidth={2} dot={false} /><Line dataKey="delivery_rate" name="Delivery rate" stroke="var(--s7)" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></div></Card>
      </div>
    </div>)
}

function AuditLog() {
  const [page, setPage] = useState(1)
  const { data, loading, error } = useLoad(() => api.get('/audit/log', { page, page_size: 25 }), [page])
  if (loading && !data) return <Spinner />
  if (error) return <ErrorBox error={error} />
  return <Card title="Audit log" subtitle="Dataset, model, policy, assumption and review events"><DataTable cols={[{ key: 'created_at', header: 'Time', render: (r: any) => String(r.created_at).slice(0, 19) }, { key: 'event', header: 'Event' }, { key: 'entity_id', header: 'Entity' }, { key: 'detail', header: 'Detail', render: (r: any) => <span className="text-muted">{JSON.stringify(r.detail).slice(0, 110)}</span> }]} rows={data.items} /><Pager page={page} total={data.total} size={25} onPage={setPage} /></Card>
}

function Body() {
  const [tab, setTab] = useState('decisions')
  return (
    <>
      <PageHeader title="Decision Audit & Governance" subtitle="Reviewable recommendations, human overrides with reasons, an immutable-style event log, and monitoring. This workspace does not certify regulatory compliance." />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: 'decisions', label: 'Decisions & review' }, { id: 'monitor', label: 'Monitoring & drift' }, { id: 'log', label: 'Audit log' }]} />
      {tab === 'decisions' && <Decisions />}{tab === 'monitor' && <Monitoring />}{tab === 'log' && <AuditLog />}
    </>)
}
export default function Governance() { return <Gate><Body /></Gate> }
