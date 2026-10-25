import { fireEvent, render, screen } from '@testing-library/react'
import { Alert, DataTable, Empty, ErrorBox, Pager, Stat, Tag } from './ui'

describe('ui kit', () => {
  it('labels quantities by provenance', () => {
    render(<Stat label="Net benefit" value="1,234" kind="estimated" />)
    expect(screen.getByText('estimated')).toBeInTheDocument()
    expect(screen.getByText('1,234')).toBeInTheDocument()
    render(<Tag kind="synthetic" />)
    expect(screen.getByText('synthetic')).toBeInTheDocument()
  })

  it('renders an explicit error and an empty state instead of blank space', () => {
    render(<><ErrorBox error={new Error('Models not trained')} /><Empty title="No rows" /></>)
    expect(screen.getByRole('alert')).toHaveTextContent('Models not trained')
    expect(screen.getByText('No rows')).toBeInTheDocument()
  })

  it('sorts table rows and reports clicks', () => {
    const clicked: string[] = []
    render(<DataTable cols={[{ key: 'id', header: 'ID' }, { key: 'v', header: 'Value' }]} rows={[{ id: 'a', v: 1 }, { id: 'b', v: 3 }, { id: 'c', v: 2 }]} onRowClick={r => clicked.push(r.id)} />)
    fireEvent.click(screen.getByRole('button', { name: /Value/ }))
    const cells = screen.getAllByRole('row').slice(1).map(r => r.textContent)
    expect(cells[0]).toContain('b')
    fireEvent.click(screen.getByText('c'))
    expect(clicked).toEqual(['c'])
  })

  it('pages and shows warnings', () => {
    const pages: number[] = []
    render(<><Pager page={2} total={95} size={20} onPage={p => pages.push(p)} /><Alert>Poor overlap</Alert></>)
    expect(screen.getByText('Page 2 / 5')).toBeInTheDocument()
    fireEvent.click(screen.getByText('Next'))
    expect(pages).toEqual([3])
    expect(screen.getByText('Poor overlap')).toBeInTheDocument()
  })
})
