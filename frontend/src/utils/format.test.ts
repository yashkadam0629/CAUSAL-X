import { fmt, pct, money, label } from './format'
describe('format helpers', () => {
  it('formats numbers and handles missing values explicitly', () => {
    expect(fmt(1.2345, 2)).toBe('1.23'); expect(fmt(null)).toBe('n/a'); expect(pct(0.256)).toBe('25.6%'); expect(money(1234.6)).toBe('1,235'); expect(label('standard_reminder')).toBe('Standard reminder')
  })
})
