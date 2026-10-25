import { useCallback, useEffect, useRef, useState } from 'react'

/** Run an async loader when deps change; exposes loading/error and a manual reload. */
export function useLoad<T>(loader: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const seq = useRef(0)
  const run = useCallback(() => {
    const id = ++seq.current
    setLoading(true)
    loader().then(d => { if (id === seq.current) { setData(d); setError(null) } }).catch(e => { if (id === seq.current) setError(e) }).finally(() => { if (id === seq.current) setLoading(false) })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  useEffect(() => { run() }, [run])
  return { data, error, loading, reload: run }
}

/** Imperative action with busy/error state (button-triggered). */
export function useAction<A extends unknown[], R>(fn: (...a: A) => Promise<R>) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [result, setResult] = useState<R | null>(null)
  const run = async (...a: A) => {
    setBusy(true); setError(null)
    try { const r = await fn(...a); setResult(r); return r } catch (e) { setError(e); return null } finally { setBusy(false) }
  }
  return { run, busy, error, result, setResult }
}
