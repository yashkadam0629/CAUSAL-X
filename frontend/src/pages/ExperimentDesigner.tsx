import { useState } from 'react'
import { useApp } from '../app/App'
import { Alert, Btn, Card, ErrorBox, Field, Note, NumInput, PageHeader, Select, Stat, TextInput } from '../components/ui'
import { useAction } from '../hooks/useAsync'
import { api } from '../services/api'
import { download, fmt } from '../utils/format'

export default function ExperimentDesigner() {
  const { current } = useApp()
  const [f, setF] = useState<any>({ outcome_type: 'binary', baseline: 0.25, mde_absolute: -0.03, alpha: 0.05, power: 0.8, allocation_ratio: 1, comparisons_vs_control: 1, attrition: 0.05, compliance: 0.9, icc: 0, cluster_size: 1,
    objective: 'Estimate the causal effect of a standard reminder on 30-day delinquency', treatment: 'standard_reminder', target_population: 'Active-loan customers contactable by SMS/e-mail', duration_days: 60, name: '' })
  const calc = useAction(() => api.post('/experiments/sample-size', f))
  const design = useAction(() => api.post('/experiments/design', { ...f, name: f.name || undefined, dataset_id: current?.id }))
  const set = (k: string) => (v: any) => setF({ ...f, [k]: v })
  const c = calc.result ?? design.result?.sample_size
  const p = design.result
  return (
    <>
      <PageHeader title="Experiment Designer" subtitle="Plan a future randomised experiment. Sample sizes use the formula for the chosen outcome type and report every adjustment. This tool supports human-led planning; it never approves an experiment." />
      <Alert>Draft only. Institutional, compliance, fairness and ethics review must approve any experiment before customers are affected.</Alert>
      <div className="mt-3 grid gap-4 xl:grid-cols-[360px_1fr]">
        <Card title="Design inputs">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Outcome type"><Select value={f.outcome_type} onChange={set('outcome_type')} options={[{ value: 'binary', label: 'Binary (rate)' }, { value: 'continuous', label: 'Continuous (mean)' }]} /></Field>
            <Field label="Baseline rate / mean"><NumInput value={f.baseline} step={0.01} onChange={set('baseline')} /></Field>
            <Field label="MDE (absolute, signed)" hint="negative = reduction"><NumInput value={f.mde_absolute} step={0.005} onChange={set('mde_absolute')} /></Field>
            {f.outcome_type === 'continuous' && <Field label="Std deviation"><NumInput value={f.sd} onChange={set('sd')} /></Field>}
            <Field label="Significance α"><NumInput value={f.alpha} step={0.01} onChange={set('alpha')} /></Field>
            <Field label="Power"><NumInput value={f.power} step={0.05} onChange={set('power')} /></Field>
            <Field label="Allocation ratio (T:C)"><NumInput value={f.allocation_ratio} step={0.25} onChange={set('allocation_ratio')} /></Field>
            <Field label="Treatment arms vs control" hint="Bonferroni"><NumInput value={f.comparisons_vs_control} min={1} onChange={set('comparisons_vs_control')} /></Field>
            <Field label="Attrition / unobserved"><NumInput value={f.attrition} step={0.01} onChange={set('attrition')} /></Field>
            <Field label="Compliance (delivery)" hint="ITT dilution"><NumInput value={f.compliance} step={0.05} onChange={set('compliance')} /></Field>
            <Field label="ICC (interference)"><NumInput value={f.icc} step={0.01} onChange={set('icc')} /></Field>
            <Field label="Cluster size"><NumInput value={f.cluster_size} min={1} onChange={set('cluster_size')} /></Field>
          </div>
          <div className="mt-3 space-y-3">
            <Field label="Objective"><TextInput value={f.objective} onChange={set('objective')} /></Field>
            <Field label="Target population"><TextInput value={f.target_population} onChange={set('target_population')} /></Field>
            <Field label="Duration (days)"><NumInput value={f.duration_days} onChange={set('duration_days')} /></Field>
            <Field label="Save protocol as (optional)"><TextInput value={f.name} onChange={set('name')} /></Field>
          </div>
          <div className="mt-3 flex gap-2"><Btn busy={calc.busy} onClick={() => calc.run()} variant="ghost">Calculate sample size</Btn><Btn busy={design.busy} onClick={() => design.run()}>Generate protocol</Btn></div>
          {(calc.error != null || design.error != null) && <div className="mt-2"><ErrorBox error={calc.error ?? design.error} /></div>}
        </Card>
        <div className="min-w-0 space-y-4">
          {c && <>
            <div className="grid gap-3 md:grid-cols-4">
              <Stat label="Control n" value={c.n_control.toLocaleString()} /><Stat label="Treatment n (per arm)" value={c.n_treatment_per_arm.toLocaleString()} /><Stat label="Total" value={c.n_total.toLocaleString()} />
              <Stat label="Inflation factor" value={fmt(c.adjustments.inflation_for_attrition_and_clustering, 2)} hint="attrition × design effect" /></div>
            <Card title="How it was calculated" subtitle={c.formula}>
              <ul className="list-disc space-y-1 pl-5 text-xs text-ink2">
                <li>Before adjustments: {c.n_before_adjustments.control.toLocaleString()} control / {c.n_before_adjustments.treatment.toLocaleString()} treatment.</li>
                <li>α after Bonferroni: {fmt(c.adjustments.alpha_after_bonferroni, 4)}; effective MDE after non-compliance: {fmt(c.adjustments.effective_mde_after_noncompliance, 4)}; design effect {fmt(c.adjustments.design_effect, 2)}.</li>
                {c.notes.map((n: string) => <li key={n}>{n}</li>)}</ul></Card></>}
          {p && <Card title="Draft protocol" actions={<Btn variant="ghost" onClick={() => download('experiment_protocol.json', JSON.stringify(p, null, 2))}>Export JSON</Btn>}>
            <Alert>{p.status}</Alert>
            <dl className="mt-3 grid gap-x-4 gap-y-2 text-xs md:grid-cols-2">
              {[['Objective', p.objective], ['Target population', p.target_population], ['Treatment', p.treatment], ['Control', p.control], ['Randomisation unit', p.randomization.unit], ['Stratification', p.randomization.stratification.join(', ')], ['Primary outcome', `${p.primary_outcome} (${p.outcome_horizon_days} days)`], ['Duration', `${p.duration_assumption_days} days`]].map(([k, v]) => <div key={k as string}><dt className="text-muted">{k}</dt><dd className="font-medium text-ink">{v}</dd></div>)}
            </dl>
            <div className="mt-3 grid gap-4 md:grid-cols-3">
              <div><h3 className="text-xs font-semibold">Guardrail metrics</h3><ul className="list-disc pl-5 text-xs text-ink2">{p.guardrails.map((g: string) => <li key={g}>{g}</li>)}</ul></div>
              <div><h3 className="text-xs font-semibold">Risks</h3><ul className="list-disc pl-5 text-xs text-ink2">{Object.entries(p.risks).map(([k, v]) => <li key={k}><b>{k}</b>: {v as string}</li>)}{p.additional_risks.map((r: string) => <li key={r}>{r}</li>)}</ul></div>
              <div><h3 className="text-xs font-semibold">Ethics & operations checklist</h3><ul className="list-disc pl-5 text-xs text-ink2">{p.ethics_checklist.map((g: string) => <li key={g}>{g}</li>)}</ul></div></div>
            <Note>Stopping rules: {p.stopping_rules.join('; ')}. {p.experiment_id ? `Saved as ${p.experiment_id}.` : ''}</Note></Card>}
        </div>
      </div>
    </>)
}
