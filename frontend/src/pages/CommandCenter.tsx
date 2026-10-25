import { Bar, BarChart, CartesianGrid, Cell, Legend as RLegend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApp } from '../app/App'
import Gate from '../components/Gate'
import { Alert, Card, ChartTooltip, ErrorBox, Note, PageHeader, Spinner, Stat } from '../components/ui'
import { useLoad } from '../hooks/useAsync'
import { api } from '../services/api'
import { ARM_LABEL, fmt, money, pct } from '../utils/format'

const hist = (h: { edges: number[]; counts: number[] }, d = 2) => h.counts.map((c, i) => ({ x: `${h.edges[i].toFixed(d)}`, count: c }))

function Body() {
  const { current } = useApp()
  const { data, error, loading } = useLoad(() => api.get('/overview', { dataset_id: current!.id }), [current?.id])
  if (loading) return <Spinner label="Running the optimiser and strategy comparison…" />
  if (error) return <ErrorBox error={error} />
  const o = data, k = o.kpis
  const strat = o.strategies.filter((s: any) => s.key)
  const opt = strat.find((s: any) => s.key === 'optimised'), rf = strat.find((s: any) => s.key === 'highest_risk_first_intensive')
  const hasTruth = !!opt?.synthetic_truth
  const stratData = strat.map((s: any) => ({ name: s.strategy.replace(/ \(.*\)/, ''), est: Math.round(s.net_benefit), truth: s.synthetic_truth ? Math.round(s.synthetic_truth.net_benefit) : undefined }))
  const mix = Object.entries(o.mix).map(([a, n]) => ({ name: ARM_LABEL[a], n }))
  return (
    <>
      <PageHeader title="Executive Command Center" subtitle={`Dataset “${o.dataset}” (${o.scenario} assignment). Decision cohorts are the held-out months. Every figure is labelled observed, predicted, estimated or simulated.`}>
        <div className="text-right text-[11px] text-muted">model <b className="text-ink">{o.model_version}</b> · policy <b className="text-ink">{o.policy_version}</b></div>
      </PageHeader>
      {o.warnings?.length > 0 && <div className="mb-3 space-y-1">{o.warnings.map((w: string) => <Alert key={w}>{w}</Alert>)}</div>}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
        <Stat label="Customers analysed" value={k.total_customers.toLocaleString()} kind="observed" hint={`${k.decision_customers.toLocaleString()} in decision cohorts`} />
        <Stat label="Eligible (≥1 action)" value={k.eligible_customers.toLocaleString()} kind="observed" hint="Rule-based eligibility, decision cohorts" />
        <Stat label="Baseline adverse risk" value={pct(k.baseline_risk)} kind="predicted" hint={`Observed rate ${pct(k.observed_adverse_rate_decision_cohorts)}`} />
        <Stat label="Recommended interventions" value={k.n_recommended.toLocaleString()} kind="estimated" hint="Selected by the optimiser" />
        <Stat label="Avoided events (expected)" value={fmt(k.expected_avoided_events, 1)} kind="estimated" hint={`± ${fmt(k.avoided_events_se, 1)} SE (independent-error approx.)`} />
        <Stat label="Net benefit" value={money(k.net_benefit)} kind="estimated" tone={k.net_benefit > 0 ? 'good' : 'bad'} hint="ARR × value − cost; value is an assumption" />
        <Stat label="Budget" value={money(k.budget)} kind="assumption" />
        <Stat label="Budget utilisation" value={pct(k.budget_utilisation)} hint={`${money(k.budget_used)} spent`} />
        <Stat label="Risk model test AUC" value={fmt(k.risk_model_test_auc, 3)} kind="observed" hint="Held-out cohorts 10-11" />
        <Stat label="Data quality score" value={k.data_quality_score ?? 'run validation'} hint="From last validation" />
        <Stat label="Model version" value={o.model_version} />
        <Stat label="Policy version" value={o.policy_version} hint="Hash of the active configuration" />
      </div>

      {opt && rf && (
        <Card className="mt-4" title="Optimised policy vs highest-risk-first" subtitle="Same budget and capacity. Highest-risk-first spends on the riskiest customers; the optimiser spends where estimated net benefit is largest.">
          <div className="grid gap-3 md:grid-cols-3">
            <Stat label="Optimised: net benefit" value={money(opt.net_benefit)} kind="estimated" hint={`${opt.n_selected} customers, spend ${money(opt.budget_used)}`} />
            <Stat label="Highest-risk-first: net benefit" value={money(rf.net_benefit)} kind="estimated" hint={`${rf.n_selected} customers, spend ${money(rf.budget_used)}`} />
            <Stat label="Estimated advantage" value={money(opt.net_benefit - rf.net_benefit)} kind="estimated" tone={opt.net_benefit > rf.net_benefit ? 'good' : 'bad'}
              hint={`Uncertainty: optimiser events ±${fmt(opt.events_se_independent, 1)}, risk-first ±${fmt(rf.events_se_independent, 1)} (SE)`} />
          </div>
          {hasTruth && <p className="mt-2 text-xs text-ink2"><span className="mr-1 rounded bg-emerald-100 px-1 text-[10px] font-semibold text-emerald-800">SYNTHETIC BENCHMARK</span>Under known ground truth: optimised {money(opt.synthetic_truth.net_benefit)} vs risk-first {money(rf.synthetic_truth.net_benefit)}.</p>}
        </Card>)}

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card title="Strategy comparison — net benefit" subtitle="Blue: model-estimated. Aqua: synthetic ground truth (benchmark only).">
          <div className="h-72" role="img" aria-label="Net benefit by strategy">
            <ResponsiveContainer><BarChart data={stratData} layout="vertical" margin={{ left: 0 }}>
              <CartesianGrid horizontal={false} /><XAxis type="number" tickFormatter={v => (v / 1000).toFixed(0) + 'k'} /><YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 10 }} />
              <Tooltip content={<ChartTooltip fmt={money} />} /><RLegend />
              <Bar dataKey="est" name="Estimated" fill="var(--s1)" radius={[0, 4, 4, 0]} barSize={9} />
              {hasTruth && <Bar dataKey="truth" name="Synthetic truth" fill="var(--s3)" radius={[0, 4, 4, 0]} barSize={9} />}
            </BarChart></ResponsiveContainer></div>
        </Card>
        <Card title="Intervention allocation (optimised)" subtitle="Customers assigned per action"><div className="h-72">
          <ResponsiveContainer><BarChart data={mix}><CartesianGrid vertical={false} /><XAxis dataKey="name" /><YAxis /><Tooltip content={<ChartTooltip />} />
            <Bar dataKey="n" name="Customers" radius={[4, 4, 0, 0]} barSize={36}>{mix.map((_, i) => <Cell key={i} fill={`var(--s${i + 1})`} />)}</Bar></BarChart></ResponsiveContainer></div></Card>
        <Card title="Baseline risk distribution" subtitle="Predicted probability of 30-day delinquency (predicted)"><div className="h-56">
          <ResponsiveContainer><BarChart data={hist(o.risk_histogram)}><CartesianGrid vertical={false} /><XAxis dataKey="x" interval={3} /><YAxis /><Tooltip content={<ChartTooltip />} />
            <Bar dataKey="count" name="Customers" fill="var(--s1)" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div></Card>
        <Card title="Estimated treatment-benefit distribution" subtitle="Best eligible action's estimated ARR (estimated; negative = possibly harmful)"><div className="h-56">
          <ResponsiveContainer><BarChart data={hist(o.benefit_histogram, 3)}><CartesianGrid vertical={false} /><XAxis dataKey="x" interval={3} /><YAxis /><Tooltip content={<ChartTooltip />} />
            <Bar dataKey="count" name="Customers" fill="var(--s2)" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div></Card>
        <Card title="Outcomes over time" subtitle="Observed adverse rate vs mean predicted risk by decision cohort (calibration drift check)" className="lg:col-span-2"><div className="h-60">
          <ResponsiveContainer><LineChart data={o.monthly}><CartesianGrid vertical={false} /><XAxis dataKey="cohort_month" /><YAxis tickFormatter={v => pct(v, 0)} domain={[0, 'auto']} />
            <Tooltip content={<ChartTooltip fmt={(v: number) => pct(v)} />} /><RLegend />
            <Line dataKey="outcome_rate" name="Observed rate" stroke="var(--s1)" strokeWidth={2} dot={{ r: 3 }} /><Line dataKey="mean_predicted_risk" name="Mean predicted" stroke="var(--s2)" strokeWidth={2} dot={{ r: 3 }} /></LineChart></ResponsiveContainer></div></Card>
      </div>
      {o.exclusion_reasons && <Card className="mt-4" title="Why customers were not selected"><div className="grid gap-2 md:grid-cols-3">{Object.entries(o.exclusion_reasons).map(([k, v]) => <div key={k} className="text-xs"><span className="tnum mr-1 font-semibold text-ink">{(v as number).toLocaleString()}</span><span className="text-muted">{k}</span></div>)}</div></Card>}
      <Note>Definitions: baseline risk = model-predicted adverse probability under historical practice; ARR = estimated absolute risk reduction vs no additional intervention; net benefit = ARR × value per avoided event − cost. Estimates exclude confounding uncertainty.</Note>
    </>)
}
export default function CommandCenter() { return <Gate><Body /></Gate> }
