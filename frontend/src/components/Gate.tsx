import { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { useApp } from '../app/App'
import { Empty } from './ui'

/** Renders children only when a dataset with trained models is selected; otherwise an actionable empty state. */
export default function Gate({ children, needModels = true }: { children: ReactNode; needModels?: boolean }) {
  const { current } = useApp()
  if (!current) return <Empty title="No dataset yet">Go to <Link className="text-accent underline" to="/data">Data &amp; Quality</Link> and generate a synthetic portfolio.</Empty>
  if (needModels && !current.models_trained)
    return <Empty title={`Models not trained for “${current.name}”`}>Train them from <Link className="text-accent underline" to="/data">Data &amp; Quality</Link> (runs as a background job).</Empty>
  return <>{children}</>
}
