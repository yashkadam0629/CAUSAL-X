import { useRef, useState } from 'react'
import { useApp } from '../app/App'
import { Alert, Btn, Card, DataTable, ErrorBox, Field, Modal, Note, NumInput, PageHeader, Select, Stat, Tag, TextInput } from '../components/ui'
import { useAction, useLoad } from '../hooks/useAsync'
import { api, pollJob } from '../services/api'
import { download, fmt, pct } from '../utils/format'

const ROLES = ['customer_id', 'decision_date', 'treatment', 'outcome', 'outcome_date', 'cost']

export default function DataManagement() {
  const { datasets, current, setCurrentId, refresh } = useApp()
  const cfg = useLoad(() => api.get('/config'), [])
  const [gen, setGen] = useState<any>({ name: 'demo_obs', size: 'medium', scenario: 'observational', seed: 42 })
  const [prog, setProg] = useState<{ p: number; m: string } | null>(null)
  const generate = useAction(async () => {
    const j = await api.post('/datasets/generate', gen)
    setProg({ p: 0, m: 'queued' })
    try { await pollJob(j.job_id, (p, m) => setProg({ p, m })) } finally { setProg(null); await refresh() }
    return true
  })
  const train = useAction(async () => {
    const j = await api.post('/models/train', { dataset_id: current!.id, seed: 42, force: true })
    setProg({ p: 0, m: 'queued' })
    try { await pollJob(j.job_id, (p, m) => setProg({ p, m })) } finally { setProg(null); await refresh() }
    return true
  })
  const validate = useAction(() => api.post(`/datasets/${current!.id}/validate`))
  const [confirmDel, setConfirmDel] = useState(false)
  const del = useAction(async () => { await api.del(`/datasets/${current!.id}`, { confirm: true }); setConfirmDel(false); await refresh() })
  // upload
  const file = useRef<HTMLInputElement>(null)
  const [staged, setStaged] = useState<any>(null)
  const [map, setMap] = useState<Record<string, string>>({})
  const [upName, setUpName] = useState('uploaded_data')
  const [control, setControl] = useState('')
  const [upRes, setUpRes] = useState<any>(null)
  const upload = useAction(async (f: File) => { const r = await api.upload(f); setStaged(r); setMap(r.suggested_mapping); setUpRes(null); return r })
  const commit = useAction(async () => { const r = await api.post('/datasets/upload/commit', { upload_token: staged.upload_token, name: upName, mapping: map, control_label: control || undefined }); setUpRes(r.validation); setStaged(null); await refresh(); return r })
  const rep = validate.result ?? upRes
  return (
    <>
      <PageHeader title="Data Quality & Management" subtitle="Generate reproducible synthetic portfolios, upload CSVs with explicit column mapping, validate, and train models. Treatments and outcomes are never invented: unsupported analyses are reported as such." />
      {prog && <Alert tone="info"><div className="mb-1">Background job: {prog.m}</div><div className="h-2 w-full rounded bg-line"><div className="h-2 rounded bg-accent" style={{ width: `${Math.round(prog.p * 100)}%` }} /></div></Alert>}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Generate synthetic portfolio" subtitle="Transparent data-generating process; ground truth stored separately for benchmarks only.">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Name"><TextInput value={gen.name} onChange={v => setGen({ ...gen, name: v })} /></Field>
            <Field label="Size"><Select value={gen.size} onChange={v => setGen({ ...gen, size: v })} options={Object.entries(cfg.data?.size_presets ?? { small: 8000, medium: 40000 }).map(([k, v]) => ({ value: k, label: `${k} (${(v as number).toLocaleString()})` }))} /></Field>
            <Field label="Assignment scenario" hint="randomized = experiment; observational = confounded; hidden = unmeasured confounder"><Select value={gen.scenario} onChange={v => setGen({ ...gen, scenario: v })} options={['randomized', 'observational', 'observational_hidden'].map(s => ({ value: s, label: s.replace('_', ' ') }))} /></Field>
            <Field label="Seed"><NumInput value={gen.seed} onChange={v => setGen({ ...gen, seed: v ?? 42 })} /></Field>
          </div>
          <div className="mt-3 flex gap-2"><Btn busy={generate.busy} onClick={() => generate.run()}>Generate & train</Btn></div>
          {generate.error != null && <div className="mt-2"><ErrorBox error={generate.error} /></div>}
          <Note>Large (200k) is for performance testing and takes several minutes.</Note>
        </Card>
        <Card title="Datasets" subtitle="Select the active dataset in the header.">
          <DataTable onRowClick={(r: any) => setCurrentId(r.id)} cols={[{ key: 'name', header: 'Name', render: (r: any) => <span className={r.id === current?.id ? 'font-semibold text-accent' : ''}>{r.name}</span> }, { key: 'source', header: 'Source' }, { key: 'scenario', header: 'Scenario' }, { key: 'n_customers', header: 'Rows', align: 'right', render: (r: any) => r.n_customers.toLocaleString() },
            { key: 'versions', header: 'Versions', align: 'right' }, { key: 'fingerprint', header: 'Fingerprint', render: (r: any) => (r.fingerprint ?? '').slice(0, 10) }, { key: 'models_trained', header: 'Models', render: (r: any) => (r.models_trained ? r.model_version : r.source === 'upload' ? 'n/a' : 'untrained') }]} rows={datasets} empty="No datasets." />
          {current && <div className="mt-3 flex flex-wrap gap-2">
            <Btn variant="ghost" busy={validate.busy} onClick={() => validate.run()}>Validate “{current.name}”</Btn>
            {current.source === 'synthetic' && <Btn variant="ghost" busy={train.busy} onClick={() => train.run()}>{current.models_trained ? 'Retrain models' : 'Train models'}</Btn>}
            <Btn variant="danger" onClick={() => setConfirmDel(true)}>Delete…</Btn></div>}
          {(validate.error != null || train.error != null) && <div className="mt-2"><ErrorBox error={validate.error ?? train.error} /></div>}
        </Card>
      </div>
      <Modal open={confirmDel} onClose={() => setConfirmDel(false)} title="Delete dataset permanently?">
        <p className="text-sm">This permanently removes <b>{current?.name}</b>, its trained models and cached artifacts. Saved runs and audit records remain only if they reference no deleted rows (they are removed with the dataset). This cannot be undone.</p>
        <div className="mt-4 flex justify-end gap-2"><Btn variant="ghost" onClick={() => setConfirmDel(false)}>Cancel</Btn><Btn variant="danger" busy={del.busy} onClick={() => del.run()}>Delete dataset</Btn></div>{del.error != null && <ErrorBox error={del.error} />}
      </Modal>
      <Card className="mt-4" title="Upload a CSV" subtitle={`Single analysis table (one row per customer decision). Max ${cfg.data?.max_upload_mb ?? 50} MB. Map your columns to the roles below; unmapped roles disable the analyses that need them.`}>
        <input ref={file} type="file" accept=".csv" className="text-xs" aria-label="CSV file" onChange={e => { const f = e.target.files?.[0]; if (f) upload.run(f) }} />
        {upload.busy && <Note>Inspecting…</Note>}{upload.error != null && <div className="mt-2"><ErrorBox error={upload.error} /></div>}
        {staged && <div className="mt-3 space-y-3">
          <Alert tone="info">{staged.n_rows.toLocaleString()} rows × {staged.n_columns} columns. Suggested mapping pre-filled from column names — please verify.</Alert>
          <div className="grid gap-3 md:grid-cols-3">{ROLES.map(r => <Field key={r} label={r.replace('_', ' ') + (r === 'customer_id' ? ' *' : '')}><Select value={map[r] ?? ''} onChange={v => setMap({ ...map, [r]: v })} options={[{ value: '', label: '— not available —' }, ...staged.columns.map((c: any) => ({ value: c.column, label: c.column }))]} /></Field>)}
            <Field label="Control label in treatment column" hint="e.g. 'control' or 'none'"><TextInput value={control} onChange={setControl} /></Field>
            <Field label="Dataset name"><TextInput value={upName} onChange={setUpName} /></Field></div>
          <DataTable cols={[{ key: 'column', header: 'Column' }, { key: 'dtype', header: 'Type' }, { key: 'missing_pct', header: 'Missing', align: 'right', render: (c: any) => pct(c.missing_pct) }, { key: 'unique', header: 'Unique', align: 'right' }]} rows={staged.columns} />
          <Btn busy={commit.busy} onClick={() => commit.run()} disabled={!map.customer_id}>Validate & import</Btn>{commit.error != null && <ErrorBox error={commit.error} />}</div>}
      </Card>
      {rep && <Card className="mt-4" title="Data quality report" actions={<Btn variant="ghost" onClick={() => download('quality_report.json', JSON.stringify(rep, null, 2))}>Download</Btn>}>
        <div className="mb-3 grid gap-3 md:grid-cols-4"><Stat label="Quality score" value={rep.quality_score} hint="100 − penalties (error 15, warning 5)" />{rep.rows_in !== undefined && <Stat label="Rows in → analysable" value={`${rep.rows_in} → ${rep.rows_analysable}`} />}</div>
        <div className="space-y-1">{rep.findings.map((f: any, i: number) => <div key={i} className={`flex gap-2 rounded px-2 py-1 text-xs ${f.severity === 'error' ? 'bg-critbg' : f.severity === 'warning' ? 'bg-warnbg' : 'bg-surface2'}`}><b className="w-14 uppercase">{f.severity}</b><span>{f.message}</span></div>)}</div>
        {rep.exclusions?.length > 0 && <><h3 className="mt-3 text-xs font-semibold">Excluded rows</h3><ul className="list-disc pl-5 text-xs">{rep.exclusions.map((e: any) => <li key={e.reason}>{e.rows} rows: {e.reason}</li>)}</ul></>}
        <h3 className="mt-3 text-xs font-semibold">Supported analyses</h3>
        <DataTable cols={[{ key: 'k', header: 'Analysis' }, { key: 's', header: 'Supported', render: (r: any) => (r.supported ? 'yes' : 'NO') }, { key: 'n', header: 'Requirement', render: (r: any) => r.needs ?? '' }]} rows={Object.entries(rep.capabilities).map(([k, v]: any) => ({ k: k.replace(/_/g, ' '), ...v }))} />
      </Card>}
    </>)
}
