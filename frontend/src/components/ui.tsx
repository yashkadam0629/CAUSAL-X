import { ReactNode, useState } from 'react'
import { AlertTriangle, CheckCircle2, Info, Loader2, XCircle } from 'lucide-react'

export type Kind = 'observed' | 'predicted' | 'estimated' | 'simulated' | 'assumption' | 'synthetic'
const KIND_STYLE: Record<Kind, string> = {
  observed: 'bg-slate-100 text-slate-700', predicted: 'bg-blue-100 text-blue-800', estimated: 'bg-violet-100 text-violet-800',
  simulated: 'bg-amber-100 text-amber-800', assumption: 'bg-orange-100 text-orange-800', synthetic: 'bg-emerald-100 text-emerald-800',
}
export const Tag = ({ kind }: { kind: Kind }) => (
  <span className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${KIND_STYLE[kind]}`} title={`This quantity is ${kind}`}>{kind}</span>
)

export function Card({ title, subtitle, actions, children, className = '' }: { title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-line bg-surface p-4 ${className}`}>
      {(title || actions) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div><h2 className="text-sm font-semibold text-ink">{title}</h2>{subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}</div>
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}

export function Stat({ label, value, hint, kind, tone }: { label: string; value: ReactNode; hint?: ReactNode; kind?: Kind; tone?: 'good' | 'bad' }) {
  return (
    <div className="rounded-lg border border-line bg-surface p-3" title={typeof hint === 'string' ? hint : undefined}>
      <div className="flex items-center justify-between gap-2"><span className="text-xs text-muted">{label}</span>{kind && <Tag kind={kind} />}</div>
      <div className={`tnum mt-1 text-xl font-semibold ${tone === 'good' ? 'text-[var(--good-ink)]' : tone === 'bad' ? 'text-crit' : 'text-ink'}`}>{value}</div>
      {hint && <div className="mt-0.5 text-[11px] leading-snug text-muted">{hint}</div>}
    </div>
  )
}

export function Btn({ children, onClick, disabled, variant = 'primary', busy, type = 'button', title }: { children: ReactNode; onClick?: () => void; disabled?: boolean; variant?: 'primary' | 'ghost' | 'danger'; busy?: boolean; type?: 'button' | 'submit'; title?: string }) {
  const v = { primary: 'bg-accent text-white hover:opacity-90', ghost: 'border border-line bg-surface text-ink hover:bg-surface2', danger: 'bg-crit text-white hover:opacity-90' }[variant]
  return (
    <button type={type} title={title} onClick={onClick} disabled={disabled || busy} className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 ${v}`}>
      {busy && <Loader2 className="spin h-4 w-4" aria-hidden />}{children}
    </button>
  )
}

export const Spinner = ({ label = 'Loading…' }: { label?: string }) => (
  <div role="status" className="flex items-center gap-2 p-6 text-sm text-muted"><Loader2 className="spin h-4 w-4" aria-hidden />{label}</div>
)
export const ErrorBox = ({ error }: { error: unknown }) => (
  <div role="alert" className="flex gap-2 rounded-md border border-crit bg-critbg p-3 text-sm text-ink"><XCircle className="mt-0.5 h-4 w-4 shrink-0 text-crit" aria-hidden /><span>{(error as Error)?.message ?? String(error)}</span></div>
)
export const Empty = ({ title, children }: { title: string; children?: ReactNode }) => (
  <div className="rounded-lg border border-dashed border-line bg-surface p-8 text-center"><p className="font-medium text-ink">{title}</p><div className="mt-1 text-sm text-muted">{children}</div></div>
)
export const Alert = ({ tone = 'warn', children }: { tone?: 'warn' | 'info' | 'ok'; children: ReactNode }) => {
  const Icon = tone === 'ok' ? CheckCircle2 : tone === 'info' ? Info : AlertTriangle
  const cls = tone === 'warn' ? 'bg-warnbg border-[var(--warn)]' : tone === 'ok' ? 'bg-emerald-50 border-emerald-300 text-emerald-900' : 'bg-blue-50 border-blue-200 text-blue-900'
  return <div className={`flex gap-2 rounded-md border p-2.5 text-xs leading-relaxed ${cls}`}><Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden /><div>{children}</div></div>
}
export const Note = ({ children }: { children: ReactNode }) => <p className="text-xs leading-relaxed text-muted">{children}</p>

export const Field = ({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) => (
  <label className="block text-xs"><span className="mb-1 block font-medium text-ink2">{label}</span>{children}{hint && <span className="mt-0.5 block text-[11px] text-muted">{hint}</span>}</label>
)
const inputCls = 'w-full rounded-md border border-line bg-surface px-2 py-1.5 text-sm text-ink'
export const NumInput = ({ value, onChange, step, min, max }: { value: number | undefined; onChange: (v: number | undefined) => void; step?: number; min?: number; max?: number }) => (
  <input className={`${inputCls} tnum`} type="number" value={value ?? ''} step={step} min={min} max={max} onChange={e => onChange(e.target.value === '' ? undefined : Number(e.target.value))} />
)
export const Select = ({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) => (
  <select className={inputCls} value={value} onChange={e => onChange(e.target.value)}>{options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}</select>
)
export const TextInput = ({ value, onChange, placeholder }: { value: string; onChange: (v: string) => void; placeholder?: string }) => (
  <input className={inputCls} value={value} placeholder={placeholder} onChange={e => onChange(e.target.value)} />
)

export function Tabs({ tabs, value, onChange }: { tabs: { id: string; label: string }[]; value: string; onChange: (id: string) => void }) {
  return (
    <div role="tablist" className="mb-4 flex gap-1 border-b border-line">
      {tabs.map(t => (
        <button key={t.id} role="tab" aria-selected={value === t.id} onClick={() => onChange(t.id)}
          className={`-mb-px border-b-2 px-3 py-2 text-sm ${value === t.id ? 'border-accent font-semibold text-ink' : 'border-transparent text-muted hover:text-ink'}`}>{t.label}</button>
      ))}
    </div>
  )
}

export interface Col<T> { key: string; header: string; render?: (r: T) => ReactNode; align?: 'right'; sortable?: boolean }
export function DataTable<T extends Record<string, any>>({ cols, rows, onRowClick, empty = 'No rows' }: { cols: Col<T>[]; rows: T[]; onRowClick?: (r: T) => void; empty?: string }) {
  const [sort, setSort] = useState<{ k: string; asc: boolean } | null>(null)
  const sorted = sort ? [...rows].sort((a, b) => { const x = a[sort.k], y = b[sort.k]; return (x > y ? 1 : x < y ? -1 : 0) * (sort.asc ? 1 : -1) }) : rows
  if (!rows.length) return <p className="p-4 text-center text-sm text-muted">{empty}</p>
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead><tr className="border-b border-line text-muted">
          {cols.map(c => (
            <th key={c.key} className={`px-2 py-1.5 font-medium ${c.align === 'right' ? 'text-right' : ''}`}>
              {c.sortable === false ? c.header : <button className="font-medium hover:text-ink" onClick={() => setSort(s => ({ k: c.key, asc: s?.k === c.key ? !s.asc : false }))}>{c.header}{sort?.k === c.key ? (sort.asc ? ' ▲' : ' ▼') : ''}</button>}
            </th>))}
        </tr></thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={i} onClick={() => onRowClick?.(r)} className={`border-b border-line/60 ${onRowClick ? 'cursor-pointer hover:bg-surface2' : ''}`}>
              {cols.map(c => <td key={c.key} className={`tnum px-2 py-1.5 ${c.align === 'right' ? 'text-right' : ''}`}>{c.render ? c.render(r) : String(r[c.key] ?? '')}</td>)}
            </tr>))}
        </tbody>
      </table>
    </div>
  )
}

export const Pager = ({ page, total, size, onPage }: { page: number; total: number; size: number; onPage: (p: number) => void }) => {
  const pages = Math.max(1, Math.ceil(total / size))
  return (
    <div className="mt-2 flex items-center justify-between text-xs text-muted">
      <span>{total.toLocaleString()} rows</span>
      <div className="flex items-center gap-2">
        <Btn variant="ghost" disabled={page <= 1} onClick={() => onPage(page - 1)}>Prev</Btn><span className="tnum">Page {page} / {pages}</span>
        <Btn variant="ghost" disabled={page >= pages} onClick={() => onPage(page + 1)}>Next</Btn>
      </div>
    </div>
  )
}

export const PageHeader = ({ title, subtitle, children }: { title: string; subtitle: string; children?: ReactNode }) => (
  <div className="mb-4 flex flex-wrap items-end justify-between gap-3"><div><h1 className="text-lg font-semibold text-ink">{title}</h1><p className="max-w-3xl text-xs text-muted">{subtitle}</p></div>{children}</div>
)

export function Legend({ items }: { items: { name: string; color: string }[] }) {
  return <div className="mb-1 flex flex-wrap gap-3 text-[11px] text-ink2">{items.map(i => <span key={i.name} className="inline-flex items-center gap-1"><span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: i.color }} />{i.name}</span>)}</div>
}

export const Modal = ({ open, onClose, title, children }: { open: boolean; onClose: () => void; title: string; children: ReactNode }) =>
  !open ? null : (
    <div role="dialog" aria-modal="true" aria-label={title} className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/40 p-4" onClick={onClose}>
      <div className="mt-10 w-full max-w-3xl rounded-lg bg-surface p-5 shadow-xl" onClick={e => e.stopPropagation()}>
        <div className="mb-3 flex items-center justify-between"><h2 className="font-semibold">{title}</h2><button aria-label="Close" onClick={onClose} className="text-muted hover:text-ink">✕</button></div>
        {children}
      </div>
    </div>
  )

export const ChartTooltip = ({ active, payload, label, fmt }: any) =>
  !active || !payload?.length ? null : (
    <div className="rounded-md border border-line bg-surface px-2.5 py-1.5 text-xs shadow">
      {label !== undefined && <div className="mb-0.5 font-medium text-ink">{label}</div>}
      {payload.map((p: any) => <div key={p.name} className="tnum text-ink2"><span className="mr-1 inline-block h-2 w-2 rounded-sm" style={{ background: p.color || p.fill }} />{p.name}: {fmt ? fmt(p.value) : typeof p.value === 'number' ? p.value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : p.value}</div>)}
    </div>
  )
