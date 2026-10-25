import { useApp } from '../app/App'
import { useLoad } from '../hooks/useAsync'
import { api } from '../services/api'
import { ARM_LABEL, fmt, money, pct } from '../utils/format'
import { Alert, DataTable, ErrorBox, Modal, Note, Spinner, Tag } from './ui'

export default function CustomerPanel({ customerId, onClose }: { customerId: string | null; onClose: () => void }) {
  const { current } = useApp()
  const { data: ex, error, loading } = useLoad(() => (customerId ? api.get(`/customers/${customerId}/explanation`, { dataset_id: current!.id }) : Promise.resolve(null)), [customerId, current?.id])
  const { data: hist } = useLoad(() => (customerId ? api.get(`/customers/${customerId}/repayment-history`, { dataset_id: current!.id }).catch(() => null) : Promise.resolve(null)), [customerId, current?.id])
  return (
    <Modal open={!!customerId} onClose={onClose} title={`Decision explanation — ${customerId}`}>
      {loading && <Spinner label="Computing explanation…" />}
      {error && <ErrorBox error={error} />}
      {ex && (
        <div className="space-y-4 text-sm">
          <Alert tone="info">{ex.disclaimer} Individual values are <b>model estimates</b>, not known facts about this person.</Alert>
          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-md border border-line p-2"><div className="text-xs text-muted">Baseline risk <Tag kind="predicted" /></div><div className="text-xl font-semibold">{pct(ex.baseline_risk.value)}</div><Note>{ex.baseline_risk.target}, {ex.baseline_risk.model}</Note></div>
            <div className="rounded-md border border-line p-2"><div className="text-xs text-muted">Recommended action <Tag kind="estimated" /></div><div className="text-xl font-semibold">{ARM_LABEL[ex.recommended_action]}</div><Note>Segment: {ex.response_segment.replace(/_/g, ' ')}</Note></div>
            <div className="rounded-md border border-line p-2"><div className="text-xs text-muted">Human review</div><div className={`text-xl font-semibold ${ex.human_review_required ? 'text-crit' : ''}`}>{ex.human_review_required ? 'Required' : 'Not required'}</div></div>
          </div>
          {ex.human_review_reasons.length > 0 && <Alert>{ex.human_review_reasons.map((r: string) => <div key={r}>• {r}</div>)}</Alert>}
          <div>
            <h3 className="mb-1 text-xs font-semibold text-ink">Counterfactual explorer — every candidate action vs no additional intervention</h3>
            <DataTable cols={[
              { key: 'label', header: 'Action' }, { key: 'eligible', header: 'Eligible', render: (c: any) => (c.eligible ? 'yes' : 'no') },
              { key: 'arr', header: 'Est. ARR (80% interval)', render: (c: any) => `${fmt(c.estimated_absolute_risk_reduction.value, 3)} [${fmt(c.estimated_absolute_risk_reduction.interval_80[0], 3)}, ${fmt(c.estimated_absolute_risk_reduction.interval_80[1], 3)}]` },
              { key: 'cost', header: 'Cost', align: 'right', render: (c: any) => money(c.intervention_cost.value) }, { key: 'gross', header: 'Gross', align: 'right', render: (c: any) => money(c.gross_benefit.value) },
              { key: 'net', header: 'Net', align: 'right', render: (c: any) => money(c.estimated_net_benefit.value) }, { key: 'u', header: 'Uncertainty', render: (c: any) => c.uncertainty_status.replace(/_/g, ' ') },
            ]} rows={ex.candidate_interventions.map((c: any) => ({ ...c, arr: c.estimated_absolute_risk_reduction.value }))} />
            <Note>ARR = absolute risk reduction (positive = fewer adverse outcomes). Net = ARR × value per avoided event − cost (cost counted once). Break-even ARR = cost ÷ value.</Note>
          </div>
          <div><h3 className="text-xs font-semibold">Why this action?</h3><p className="text-xs text-ink2">{ex.selection_reason}</p>
            <ul className="mt-1 list-disc pl-5 text-xs text-ink2">{ex.alternatives_not_selected.map((a: any) => <li key={a.action}><b>{ARM_LABEL[a.action]}</b>: {a.reason_not_selected}</li>)}</ul>
            <h3 className="mt-2 text-xs font-semibold">What would change the decision</h3><ul className="list-disc pl-5 text-xs text-ink2">{ex.what_would_change_the_decision.map((w: string) => <li key={w}>{w}</li>)}</ul></div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-line p-2"><h3 className="text-xs font-semibold">Predictive explanation (SHAP) <Tag kind="predicted" /></h3><Note>{ex.explanations.predictive.type}</Note>
              {ex.explanations.predictive.top_drivers.map((d: any) => <div key={d.feature} className="tnum flex justify-between text-xs"><span>{d.feature}</span><span className={d.shap_log_odds > 0 ? 'text-crit' : 'text-[var(--good-ink)]'}>{d.shap_log_odds > 0 ? '+' : ''}{fmt(d.shap_log_odds, 3)}</span></div>)}</div>
            <div className="rounded-md border border-line p-2"><h3 className="text-xs font-semibold">Causal explanation <Tag kind="estimated" /></h3><Note>{ex.explanations.causal.type}</Note>
              <ul className="mt-1 list-disc pl-4 text-[11px] text-ink2">{ex.explanations.causal.assumptions.map((a: string) => <li key={a}>{a}</li>)}</ul><Note>{ex.explanations.causal.limits}</Note></div>
          </div>
          {hist && <div><h3 className="mb-1 text-xs font-semibold">Repayment history before the decision date <Tag kind="observed" /></h3>
            <DataTable cols={[{ key: 'scheduled_due_date', header: 'Due', render: (r: any) => String(r.scheduled_due_date).slice(0, 10) }, { key: 'scheduled_amount', header: 'Amount', align: 'right', render: (r: any) => money(r.scheduled_amount) }, { key: 'days_past_due', header: 'Days past due', align: 'right' }, { key: 'repayment_status', header: 'Status' }]} rows={hist.repayments} /></div>}
        </div>)}
    </Modal>
  )
}
