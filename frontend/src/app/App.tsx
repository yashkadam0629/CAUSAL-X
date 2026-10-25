import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { Activity, BarChart3, Beaker, Database, FileText, FlaskConical, Gauge, LineChart, Moon, Network, ScrollText, Sun, Target, Users } from 'lucide-react'
import { api } from '../services/api'
import { ErrorBox, Spinner } from '../components/ui'
import CommandCenter from '../pages/CommandCenter'
import Portfolio from '../pages/Portfolio'
import CausalLab from '../pages/CausalLab'
import Uplift from '../pages/Uplift'
import Optimizer from '../pages/Optimizer'
import Scenarios from '../pages/Scenarios'
import PolicyEval from '../pages/PolicyEval'
import ExperimentDesigner from '../pages/ExperimentDesigner'
import ModelLab from '../pages/ModelLab'
import DataManagement from '../pages/DataManagement'
import Governance from '../pages/Governance'
import Reports from '../pages/Reports'

export interface DatasetRow { id: number; name: string; source: string; scenario: string | null; n_customers: number; fingerprint: string; models_trained: boolean; model_version: string | null }
interface Ctx { datasets: DatasetRow[]; current: DatasetRow | null; setCurrentId: (id: number) => void; refresh: () => Promise<void>; lastRunId: string | null; setLastRunId: (id: string | null) => void }
const AppCtx = createContext<Ctx>(null as unknown as Ctx)
export const useApp = () => useContext(AppCtx)

const NAV = [
  { to: '/', label: 'Command Center', icon: Gauge }, { to: '/portfolio', label: 'Portfolio Explorer', icon: Users }, { to: '/causal', label: 'Causal Lab', icon: FlaskConical },
  { to: '/uplift', label: 'Uplift Intelligence', icon: Network }, { to: '/optimizer', label: 'Intervention Optimizer', icon: Target }, { to: '/scenarios', label: 'Scenario Studio', icon: LineChart },
  { to: '/policy', label: 'Policy Evaluation', icon: BarChart3 }, { to: '/experiments', label: 'Experiment Designer', icon: Beaker }, { to: '/models', label: 'Model Laboratory', icon: Activity },
  { to: '/data', label: 'Data & Quality', icon: Database }, { to: '/governance', label: 'Audit & Governance', icon: ScrollText }, { to: '/reports', label: 'Executive Reports', icon: FileText },
]

export default function App() {
  const [datasets, setDatasets] = useState<DatasetRow[]>([])
  const [currentId, setCurrentId] = useState<number | null>(() => Number(localStorage.getItem('cx.ds')) || null)
  const [lastRunId, setLastRunId] = useState<string | null>(null)
  const [err, setErr] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const [dark, setDark] = useState(() => localStorage.getItem('cx.theme') === 'dark')

  const refresh = useCallback(async () => {
    try { const d: DatasetRow[] = await api.get('/datasets'); setDatasets(d); setErr(null)
      setCurrentId(c => (c && d.some(x => x.id === c) ? c : (d.find(x => x.models_trained) ?? d[0])?.id ?? null))
    } catch (e) { setErr(e) } finally { setLoading(false) }
  }, [])
  useEffect(() => { refresh() }, [refresh])
  useEffect(() => { document.documentElement.dataset.theme = dark ? 'dark' : 'light'; localStorage.setItem('cx.theme', dark ? 'dark' : 'light') }, [dark])
  useEffect(() => { if (currentId) localStorage.setItem('cx.ds', String(currentId)) }, [currentId])
  const current = datasets.find(d => d.id === currentId) ?? null

  return (
    <AppCtx.Provider value={{ datasets, current, setCurrentId: setCurrentId as any, refresh, lastRunId, setLastRunId }}>
      <div className="flex min-h-screen">
        <aside className="sticky top-0 hidden h-screen w-56 shrink-0 flex-col bg-nav p-3 text-[var(--nav-ink)] md:flex">
          <div className="mb-4 px-2"><div className="text-lg font-bold tracking-tight text-white">CAUSAL<span className="text-[#6da7ec]">-X</span></div><div className="text-[10px] leading-tight opacity-80">Predict outcomes. Estimate impact. Optimize decisions.</div></div>
          <nav className="flex-1 space-y-0.5 overflow-y-auto" aria-label="Main">
            {NAV.map(n => (
              <NavLink key={n.to} to={n.to} end={n.to === '/'} className={({ isActive }) => `flex items-center gap-2 rounded-md px-2 py-1.5 text-[13px] ${isActive ? 'bg-white/15 text-white' : 'hover:bg-white/10'}`}>
                <n.icon className="h-4 w-4" aria-hidden />{n.label}</NavLink>))}
          </nav>
          <p className="mt-2 px-1 text-[10px] leading-snug opacity-70">Decision support only. Not legal, suitability or regulatory advice. Synthetic data demonstrate methodology, not real-world effectiveness.</p>
        </aside>
        <div className="min-w-0 flex-1">
          <header className="sticky top-0 z-20 flex flex-wrap items-center gap-3 border-b border-line bg-surface/95 px-4 py-2 backdrop-blur">
            <label className="flex items-center gap-2 text-xs text-muted">Dataset
              <select aria-label="Dataset" className="rounded-md border border-line bg-surface px-2 py-1 text-sm text-ink" value={currentId ?? ''} onChange={e => setCurrentId(Number(e.target.value))}>
                {datasets.length === 0 && <option value="">none</option>}
                {datasets.map(d => <option key={d.id} value={d.id}>{d.name} · {d.scenario ?? d.source}{d.models_trained ? '' : ' (untrained)'}</option>)}
              </select></label>
            {current && <span className="text-[11px] text-muted">model <b className="text-ink">{current.model_version ?? '—'}</b> · {current.n_customers.toLocaleString()} customers · {current.source}</span>}
            <button aria-label="Toggle dark mode" className="ml-auto rounded-md border border-line p-1.5 text-muted hover:text-ink" onClick={() => setDark(d => !d)}>{dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</button>
          </header>
          <main className="p-4">
            {loading ? <Spinner /> : err ? <ErrorBox error={err} /> : (
              <Routes>
                <Route path="/" element={<CommandCenter />} /><Route path="/portfolio" element={<Portfolio />} /><Route path="/causal" element={<CausalLab />} />
                <Route path="/uplift" element={<Uplift />} /><Route path="/optimizer" element={<Optimizer />} /><Route path="/scenarios" element={<Scenarios />} />
                <Route path="/policy" element={<PolicyEval />} /><Route path="/experiments" element={<ExperimentDesigner />} /><Route path="/models" element={<ModelLab />} />
                <Route path="/data" element={<DataManagement />} /><Route path="/governance" element={<Governance />} /><Route path="/reports" element={<Reports />} />
                <Route path="*" element={<Navigate to="/" />} />
              </Routes>)}
          </main>
        </div>
      </div>
    </AppCtx.Provider>
  )
}
