import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useApp } from '../app/App'
import CustomerPanel from '../components/CustomerPanel'
import Gate from '../components/Gate'
import { Card, DataTable, ErrorBox, Field, NumInput, PageHeader, Pager, Select, Spinner, Tag, TextInput } from '../components/ui'
import { useLoad } from '../hooks/useAsync'
import { api } from '../services/api'
import { ARM_LABEL, fmt, money, pct } from '../utils/format'

const SEGMENTS = ['high_benefit', 'moderate_benefit', 'low_benefit', 'uncertain_effect', 'possibly_counterproductive', 'outside_supported_population']
const opt = (xs: string[], all = 'All') => [{ value: '', label: all }, ...xs.map(x => ({ value: x, label: x.replace(/_/g, ' ') }))]

function Body() {
  const { current } = useApp()
  const [sp, setSp] = useSearchParams()
  const [page, setPage] = useState(1)
  const [f, setF] = useState<Record<string, any>>({ response_segment: sp.get('segment') ?? '', split: 'test', sort: 'best_net' })
  const [open, setOpen] = useState<string | null>(null)
  const set = (k: string, v: any) => { setPage(1); setF(s => ({ ...s, [k]: v })) }
  const { data, error, loading } = useLoad(() => api.get('/customers', { dataset_id: current!.id, page, page_size: 20, desc: true, ...f }), [current?.id, page, JSON.stringify(f)])
  return (
    <>
      <PageHeader title="Portfolio Explorer" subtitle="Synthetic customers with loan details, predicted risk and estimated intervention economics. Click a row for the full decision explanation." />
      <Card className="mb-3">
        <div className="grid gap-3 md:grid-cols-4 xl:grid-cols-6">
          <Field label="Search ID"><TextInput value={f.q ?? ''} onChange={v => set('q', v)} placeholder="SYN-CUST-…" /></Field>
          <Field label="Loan type"><Select value={f.loan_type ?? ''} onChange={v => set('loan_type', v)} options={opt(['personal', 'auto', 'credit_line', 'small_business'])} /></Field>
          <Field label="Risk band"><Select value={f.risk_band ?? ''} onChange={v => set('risk_band', v)} options={opt(['low', 'medium', 'high', 'very_high'])} /></Field>
          <Field label="Customer segment"><Select value={f.customer_segment ?? ''} onChange={v => set('customer_segment', v)} options={opt(['stable', 'watchlist', 'stressed'])} /></Field>
          <Field label="Response segment"><Select value={f.response_segment ?? ''} onChange={v => { setSp(v ? { segment: v } : {}); set('response_segment', v) }} options={opt(SEGMENTS)} /></Field>
          <Field label="Eligible for"><Select value={f.eligible_action ?? ''} onChange={v => set('eligible_action', v)} options={opt(['standard_reminder', 'counselling', 'repayment_plan'])} /></Field>
          <Field label="Uncertainty"><Select value={f.uncertainty ?? ''} onChange={v => set('uncertainty', v)} options={opt(['low', 'moderate', 'high'])} /></Field>
          <Field label="Region group"><Select value={f.region_group ?? ''} onChange={v => set('region_group', v)} options={opt(['R1', 'R2', 'R3', 'R4', 'R5'])} /></Field>
          <Field label="Min prior delinquencies"><NumInput value={f.min_delinq} onChange={v => set('min_delinq', v)} min={0} /></Field>
          <Field label="Min est. benefit (ARR)"><NumInput value={f.min_arr} onChange={v => set('min_arr', v)} step={0.01} /></Field>
          <Field label="Cohorts"><Select value={f.split ?? ''} onChange={v => set('split', v)} options={[{ value: '', label: 'All' }, { value: 'test', label: 'Decision (10-11)' }, { value: 'validation', label: 'Validation (8-9)' }, { value: 'train', label: 'Training (0-7)' }]} /></Field>
          <Field label="Sort by"><Select value={f.sort} onChange={v => set('sort', v)} options={[{ value: 'best_net', label: 'Est. net benefit' }, { value: 'risk', label: 'Predicted risk' }, { value: 'best_arr', label: 'Est. benefit' }, { value: 'outstanding_balance', label: 'Balance' }]} /></Field>
        </div>
      </Card>
      <Card>
        {loading && !data ? <Spinner /> : error ? <ErrorBox error={error} /> : (
          <>
            <DataTable onRowClick={r => setOpen(r.customer_id)} empty="No customers match these filters." cols={[
              { key: 'customer_id', header: 'Customer' }, { key: 'loan_type', header: 'Loan' }, { key: 'outstanding_balance', header: 'Balance', align: 'right', render: r => money(r.outstanding_balance) },
              { key: 'debt_to_income_ratio', header: 'DTI', align: 'right', render: r => fmt(r.debt_to_income_ratio) }, { key: 'historical_delinquency_count', header: 'Prior delinq.', align: 'right' },
              { key: 'risk', header: 'Risk (pred.)', align: 'right', render: r => pct(r.risk) }, { key: 'eligible_actions', header: 'Eligible', render: r => r.eligible_actions.split(',').filter(Boolean).map((a: string) => ARM_LABEL[a].split(' ')[0]).join(', ') || '—' },
              { key: 'best_action', header: 'Best action', render: r => (r.best_action === 'none' ? '—' : ARM_LABEL[r.best_action]) }, { key: 'best_arr', header: 'Est. ARR', align: 'right', render: r => fmt(r.best_arr, 3) },
              { key: 'best_cost', header: 'Cost', align: 'right', render: r => (r.best_action === 'none' ? '—' : money(r.best_cost)) }, { key: 'best_net', header: 'Net (est.)', align: 'right', render: r => money(r.best_net) },
              { key: 'uncertainty', header: 'Uncertainty' }, { key: 'data_completeness', header: 'Complete', align: 'right', render: r => pct(r.data_completeness, 0) }, { key: 'response_segment', header: 'Segment', render: r => r.response_segment.replace(/_/g, ' ') },
            ]} rows={data.items} />
            <Pager page={page} total={data.total} size={20} onPage={setPage} />
            <p className="mt-2 text-[11px] text-muted"><Tag kind="predicted" /> risk · <Tag kind="estimated" /> ARR and net · <Tag kind="observed" /> balance, DTI, delinquencies. Sorting applies to the whole filtered set.</p>
          </>)}
      </Card>
      <CustomerPanel customerId={open} onClose={() => setOpen(null)} />
    </>)
}
export default function Portfolio() { return <Gate><Body /></Gate> }
