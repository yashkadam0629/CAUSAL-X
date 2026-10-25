export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) { super(message); this.status = status }
}
const BASE = '/api/v1'

async function request<T>(method: string, path: string, body?: unknown, params?: Record<string, unknown>): Promise<T> {
  const qs = params
    ? '?' + Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '').map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join('&')
    : ''
  let res: Response
  try {
    res = await fetch(BASE + path + qs, { method, headers: body ? { 'Content-Type': 'application/json' } : undefined, body: body ? JSON.stringify(body) : undefined })
  } catch {
    throw new ApiError(0, 'Cannot reach the CAUSAL-X API. Is the backend running on port 8000?')
  }
  if (!res.ok) {
    let msg = res.statusText
    try {
      const j = await res.json()
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j)
    } catch { /* ignore */ }
    throw new ApiError(res.status, msg)
  }
  const ct = res.headers.get('content-type') || ''
  return (ct.includes('json') ? res.json() : res.text()) as Promise<T>
}

export const api = {
  get: <T = any>(path: string, params?: Record<string, unknown>) => request<T>('GET', path, undefined, params),
  post: <T = any>(path: string, body?: unknown, params?: Record<string, unknown>) => request<T>('POST', path, body ?? {}, params),
  del: <T = any>(path: string, params?: Record<string, unknown>) => request<T>('DELETE', path, undefined, params),
  upload: async (file: File) => {
    const fd = new FormData(); fd.append('file', file)
    const res = await fetch(BASE + '/datasets/upload', { method: 'POST', body: fd })
    if (!res.ok) { const j = await res.json().catch(() => ({})); throw new ApiError(res.status, j.detail || res.statusText) }
    return res.json()
  },
}

export async function pollJob(jobId: string, onProgress?: (p: number, m: string) => void): Promise<any> {
  for (;;) {
    const j = await api.get(`/jobs/${jobId}`)
    onProgress?.(j.progress, j.message)
    if (j.status === 'succeeded') return j.result
    if (j.status === 'failed') throw new ApiError(500, j.message)
    await new Promise(r => setTimeout(r, 1500))
  }
}
