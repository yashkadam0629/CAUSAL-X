import { useApp } from '../app/App'
import Gate from '../components/Gate'
import { Alert, Btn, Card, DataTable, ErrorBox, Note, PageHeader } from '../components/ui'
import { useAction, useLoad } from '../hooks/useAsync'
import { api } from '../services/api'

function Body() {
  const { current, lastRunId } = useApp()
  const list = useLoad(() => api.get('/reports'), [])
  const gen = useAction(async () => { const r = await api.post('/reports/generate', undefined, { dataset_id: current!.id, run_id: lastRunId ?? undefined }); list.reload(); return r })
  return (
    <>
      <PageHeader title="Executive Reports" subtitle="Generates a report from executed results only: summary, data, assumptions, predictive, causal and uplift results, policy comparison, scenarios, limitations and governance." />
      <Alert>Reports are decision-support documents. They are not an official audit, regulatory certification or legal opinion.</Alert>
      <Card className="mt-3" title="Generate" subtitle={lastRunId ? `Uses your latest saved optimisation run (${lastRunId}) plus the standard scenario set.` : 'No saved run in this session: a fresh optimisation with default assumptions is used.'}>
        <Btn busy={gen.busy} onClick={() => gen.run()}>Generate report (Markdown, HTML, PDF) — ~40 s</Btn>
        {gen.error != null && <div className="mt-2"><ErrorBox error={gen.error} /></div>}
        {gen.result && <p className="mt-2 text-xs text-[var(--good-ink)]">Generated {gen.result.markdown} with {gen.result.scenarios_included} scenarios{gen.result.pdf ? '' : ' (PDF unavailable — reportlab not installed)'}.</p>}
      </Card>
      <Card className="mt-4" title="Previous reports">
        <DataTable cols={[{ key: 'name', header: 'File', render: (r: any) => <a className="text-accent underline" href={`/api/v1/reports/${r.name}`} download>{r.name}</a> }, { key: 'size', header: 'KB', align: 'right', render: (r: any) => (r.size / 1024).toFixed(0) }, { key: 'modified', header: 'Created', render: (r: any) => new Date(r.modified * 1000).toLocaleString() }]} rows={list.data ?? []} empty="No reports yet." />
        <Note>CSV exports: decisions (Audit & Governance), quality report (Data & Quality), scenarios and causal results (JSON export buttons).</Note>
      </Card>
    </>)
}
export default function Reports() { return <Gate><Body /></Gate> }
