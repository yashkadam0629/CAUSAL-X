export const fmt = (x: number | null | undefined, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? 'n/a' : x.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }))
export const pct = (x: number | null | undefined, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? 'n/a' : (x * 100).toFixed(d) + '%')
export const money = (x: number | null | undefined) => (x === null || x === undefined || Number.isNaN(x) ? 'n/a' : Math.round(x).toLocaleString())
export const label = (s: string) => s.replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase())
export const ARM_LABEL: Record<string, string> = { none: 'No intervention', standard_reminder: 'Standard reminder', counselling: 'Financial counselling', repayment_plan: 'Repayment-plan review' }
export const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)', 'var(--s7)', 'var(--s8)']
export function download(name: string, content: string, type = 'application/json') {
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([content], { type })); a.download = name; a.click()
}
