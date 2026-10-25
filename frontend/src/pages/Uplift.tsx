import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend as RLegend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { useApp } from '../app/App'
import CustomerPanel from '../components/CustomerPanel'
import Gate from '../components/Gate'
import { Alert, Card, ChartTooltip, DataTable, ErrorBox, Field, NumInput, Note, PageHeader, Select, Spinner, Stat, Tag } from '../components/ui'
import { useLoad } from '../hooks/useAsync'
import { api } from '../services/api'
import { ARM_LABEL, fmt, label, pct } from '../utils/format'

const SEG_COLOR: Record<string, string> = { high_benefit: 'var(--s1)', moderate_benefit: 'var(--s3)', low_benefit: 'var(--s4)', uncertain_effect: 'var(--muted)', possibly_counterproductive: 'var(--s8)', outside_supported_population: 'var(--axis)' }

function Body() {
  const { current } = useApp()
  const nav = useNavigate()
  const [th, setTh] = useState({ high: 0.08, moderate: 0.03, z: 1.28 })
  const [arm, setArm] = useState('standard_reminder')
  const [open, setOpen] = useState<string | null>(null)
  const seg = useLoad(() => api.get('/uplift/segments', { dataset_id: current!.id, ...th }), [current?.id, th.high, th.moderate, th.z])
  const met = useLoad(() => api.get('/uplift/metrics', { dataset_id: current!.id }), [current?.id])
  if (seg.loading && !seg.data) return <Spinner />
  if (seg.error) return <ErrorBox error={seg.error} />
  const s = seg.data
  const ev = met.data?.evaluation?.[arm]
  const curve = ev?.status === 'ok' ? ev.rankers.dr_learner.curve.map((p: any, i: number) => ({ fraction: p.fraction, dr: p.cum_benefit, t: ev.rankers.t_learner.curve[i].cum_benefit, risk: ev.rankers.baseline_risk_only.curve[i].cum_benefit, random: p.fraction * ev.total_benefit_per_unit })) : []
  const segData = Object.entries(s.counts).map(([k, v]) => ({ k, name: label(k), n: v as number }))
  const groups = Object.keys(SEG_COLOR).map(k => ({ k, pts: s.scatter.filter((p: any) => p.segment === k) })).filter(g => g.pts.length)
  return (
    <>
      <PageHeader title="Uplift Intelligence" subtitle="Who benefits from which action? Risk tells you who is likely to default; uplift estimates who changes outcome because of an intervention. Segments are heuristic groupings of model estimates." />
      <Alert tone="info">Across decision-cohort customers the correlation between baseline risk and best-action estimated benefit is <b>{fmt(s.risk_benefit_correlation, 2)}</b>: {Math.abs(s.risk_benefit_correlation) < 0.6 ? 'risk is only a partial guide to who benefits.' : 'risk and benefit are closely related for this action mix, but ranking by risk still ignores cost and uncertainty.'}</Alert>
      <div className="mt-3 grid gap-4 lg:grid-cols-2">
        <Card title="Risk vs estimated benefit" subtitle={`Each point is a customer's best eligible action (n=${s.scatter.length} sampled). Click a point for the explanation. Estimates, not facts.`}>
          <div className="h-80" role="img" aria-label="Scatter of baseline risk versus estimated benefit"><ResponsiveContainer>
            <ScatterChart margin={{ left: 0 }}><CartesianGrid /><XAxis type="number" dataKey="risk" name="Baseline risk" tickFormatter={v => pct(v, 0)} label={{ value: 'Baseline risk (predicted)', position: 'insideBottom', offset: -4, fontSize: 11 }} />
              <YAxis type="number" dataKey="arr" name="Est. ARR" tickFormatter={v => fmt(v, 2)} label={{ value: 'Estimated benefit (ARR)', angle: -90, position: 'insideLeft', fontSize: 11 }} /><ZAxis range={[18, 18]} />
              <ReferenceLine y={0} stroke="var(--axis)" /><ReferenceLine x={s.risk_median} stroke="var(--axis)" strokeDasharray="4 3" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }: any) => active && payload?.length ? <div className="rounded border border-line bg-surface p-2 text-xs"><b>{payload[0].payload.customer_id}</b><br />risk {pct(payload[0].payload.risk)} · ARR {fmt(payload[0].payload.arr, 3)} ± {fmt(payload[0].payload.se, 3)}<br />{ARM_LABEL[payload[0].payload.action]} · {label(payload[0].payload.segment)}</div> : null} />
              <RLegend />
              {groups.map(g => <Scatter key={g.k} name={label(g.k)} data={g.pts} fill={SEG_COLOR[g.k]} fillOpacity={0.75} onClick={(p: any) => setOpen(p.customer_id)} cursor="pointer" />)}
            </ScatterChart></ResponsiveContainer></div>
          <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-ink2">
            <span>High risk, high benefit: <b>{s.quadrants.high_risk_high_benefit}</b></span><span>High risk, low benefit: <b>{s.quadrants.high_risk_low_benefit}</b></span>
            <span>Moderate/low risk, high benefit: <b>{s.quadrants.moderate_risk_high_benefit}</b></span><span>Low risk, low benefit: <b>{s.quadrants.low_risk_low_benefit}</b></span></div>
        </Card>
        <Card title="Response segments" subtitle="Click a bar to open those customers in the Portfolio Explorer.">
          <div className="grid grid-cols-3 gap-2"><Field label="High benefit ≥ ARR"><NumInput value={th.high} step={0.01} onChange={v => setTh({ ...th, high: v ?? 0.08 })} /></Field><Field label="Moderate ≥ ARR"><NumInput value={th.moderate} step={0.01} onChange={v => setTh({ ...th, moderate: v ?? 0.03 })} /></Field><Field label="Interval z"><NumInput value={th.z} step={0.1} onChange={v => setTh({ ...th, z: v ?? 1.28 })} /></Field></div>
          <div className="mt-2 h-60"><ResponsiveContainer><BarChart data={segData} layout="vertical" margin={{ left: 0 }}><CartesianGrid horizontal={false} /><XAxis type="number" /><YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 10 }} /><Tooltip content={<ChartTooltip />} />
            <Bar dataKey="n" name="Customers" barSize={14} radius={[0, 4, 4, 0]} cursor="pointer" onClick={(d: any) => nav(`/portfolio?segment=${d.k}`)}>{segData.map(d => <Cell key={d.k} fill={SEG_COLOR[d.k]} />)}</Bar></BarChart></ResponsiveContainer></div>
          <Note>{s.thresholds.how_chosen}. Counterproductive = the upper bound of the interval is below zero.</Note>
        </Card>
      </div>
      <Card className="mt-4" title="Uplift evaluation on held-out cohorts" subtitle="IPW-weighted uplift (Qini) curve; accuracy is not used. Cumulative incremental events prevented per eligible customer, targeting top-ranked first."
        actions={<div className="w-52"><Select value={arm} onChange={setArm} options={['standard_reminder', 'counselling', 'repayment_plan'].map(a => ({ value: a, label: ARM_LABEL[a] }))} /></div>}>
        {met.loading ? <Spinner /> : ev?.status !== 'ok' ? <Alert>Insufficient evidence: {ev?.reason ?? 'no evaluation'}</Alert> : (
          <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
            <div className="h-72"><ResponsiveContainer><LineChart data={curve}><CartesianGrid vertical={false} /><XAxis dataKey="fraction" tickFormatter={v => pct(v, 0)} /><YAxis tickFormatter={v => fmt(v, 3)} /><Tooltip content={<ChartTooltip fmt={(v: number) => fmt(v, 4)} />} /><RLegend />
              <Line dataKey="dr" name="DR-learner" stroke="var(--s1)" strokeWidth={2} dot={false} /><Line dataKey="t" name="T-learner" stroke="var(--s3)" strokeWidth={2} dot={false} /><Line dataKey="risk" name="Risk-only ranking" stroke="var(--s2)" strokeWidth={2} dot={false} /><Line dataKey="random" name="Random" stroke="var(--muted)" strokeDasharray="4 3" dot={false} /></LineChart></ResponsiveContainer></div>
            <div className="space-y-2">
              <DataTable cols={[{ key: 'name', header: 'Ranker' }, { key: 'qini', header: 'Qini', align: 'right', render: (r: any) => fmt(r.qini, 4) }, { key: 'ci', header: '95% bootstrap CI', align: 'right', render: (r: any) => `[${fmt(r.ci[0], 4)}, ${fmt(r.ci[1], 4)}]` }]}
                rows={Object.entries(ev.rankers).map(([k, v]: any) => ({ name: label(k), qini: v.qini, ci: v.qini_ci }))} />
              <Note>{ev.method}. {ev.caveat}</Note>
              {ev.synthetic_benchmark && <Alert tone="info"><b>SYNTHETIC BENCHMARK</b>: correlation of predicted vs true ARR {fmt(ev.synthetic_benchmark.corr_pred_vs_true_arr, 2)} (risk-only: {fmt(ev.synthetic_benchmark.corr_risk_vs_true_arr, 2)}); RMSE {fmt(ev.synthetic_benchmark.rmse, 3)}.</Alert>}
            </div>
            <div className="lg:col-span-2"><h3 className="mb-1 text-xs font-semibold">Treatment-effect calibration (quintiles of predicted ARR)</h3>
              <div className="h-48"><ResponsiveContainer><ComposedChart data={ev.calibration}><CartesianGrid vertical={false} /><XAxis dataKey="quintile" /><YAxis tickFormatter={v => fmt(v, 2)} /><Tooltip content={<ChartTooltip fmt={(v: number) => fmt(v, 4)} />} /><RLegend />
                <Bar dataKey="mean_predicted_arr" name="Predicted ARR" fill="var(--s1)" barSize={18} radius={[3, 3, 0, 0]} /><Bar dataKey="ipw_realised_arr" name="IPW-realised ARR" fill="var(--s2)" barSize={18} radius={[3, 3, 0, 0]} /></ComposedChart></ResponsiveContainer></div>
              <Note>Realised ARR per quintile is a noisy IPW estimate (n≈{ev.calibration[0].n} each); judge the ordering, not exact levels.</Note></div>
          </div>)}
      </Card>
      <CustomerPanel customerId={open} onClose={() => setOpen(null)} />
    </>)
}
export default function Uplift() { return <Gate><Body /></Gate> }
