import { describe, expect, it } from 'vitest';
import { formatMultiplier, formatNumber, formatPercent, formatPrice } from './formatting';

describe('formatPrice', () => {
  it('formats large stock prices with 2 decimal places', () => {
    expect(formatPrice(200)).toBe('$200.00');
    expect(formatPrice(1234.56)).toBe('$1,234.56');
  });

  it('formats standard stock prices with 2 decimal places', () => {
    expect(formatPrice(15.25)).toBe('$15.25');
    expect(formatPrice(1.0)).toBe('$1.00');
  });

  it('formats sub-dollar penny stocks (between $0.01 and $1.00) with at least 3 decimals', () => {
    expect(formatPrice(0.31)).toBe('$0.310');
    expect(formatPrice(0.315)).toBe('$0.315');
    expect(formatPrice(0.05)).toBe('$0.050');
  });

  it('formats micro-cap sub-penny stocks (< $0.01) with 4-6 meaningful decimals', () => {
    expect(formatPrice(0.0042)).toBe('$0.0042');
    expect(formatPrice(0.000123)).toBe('$0.000123');
  });

  it('handles negative prices, zero, and non-finite inputs safely', () => {
    expect(formatPrice(-0.31)).toBe('-$0.310');
    expect(formatPrice(0)).toBe('$0.00');
    expect(formatPrice(null)).toBe('—');
    expect(formatPrice(undefined)).toBe('—');
    expect(formatPrice(NaN)).toBe('—');
  });
});

describe('formatPercent and formatMultiplier', () => {
  it('formats percentages correctly', () => {
    expect(formatPercent(2.4)).toBe('+2.40%');
    expect(formatPercent(-6.5)).toBe('-6.50%');
    expect(formatPercent(0)).toBe('0.00%');
    expect(formatPercent(NaN)).toBe('—');
  });

  it('formats multipliers correctly', () => {
    expect(formatMultiplier(0.982)).toBe('0.982×');
    expect(formatMultiplier(1.011)).toBe('1.011×');
    expect(formatMultiplier(null)).toBe('—');
  });
});
