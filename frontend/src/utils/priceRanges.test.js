import { describe, expect, it } from 'vitest';
import {
  availableRanges,
  defaultRangeId,
  periodChange,
  sliceRangePoints,
} from './priceRanges';

function dailyHistory(n, startPrice = 100) {
  return {
    daily: Array.from({ length: n }, (_, i) => ({
      d: `2024-01-${String((i % 28) + 1).padStart(2, '0')}`,
      c: startPrice + i,
    })),
    intraday: [{ t: '2024-06-01T14:30:00+00:00', c: 150 }],
  };
}

describe('availableRanges', () => {
  it('shows every tab for deep history with intraday', () => {
    expect(availableRanges({ dailyCount: 2500, hasIntraday: true })).toEqual([
      '24H', '5D', '1M', '6M', '1Y', '5Y', 'MAX',
    ]);
  });

  it('hides 5Y for a 2024 IPO with ~500 sessions', () => {
    expect(availableRanges({ dailyCount: 500, hasIntraday: true })).toEqual([
      '24H', '5D', '1M', '6M', '1Y', 'MAX',
    ]);
  });

  it('hides 24H when intraday failed to load', () => {
    const ids = availableRanges({ dailyCount: 2500, hasIntraday: false });
    expect(ids).not.toContain('24H');
    expect(ids).toContain('MAX');
  });

  it('shows only MAX for a brand-new listing', () => {
    expect(availableRanges({ dailyCount: 2, hasIntraday: false })).toEqual(['MAX']);
  });

  it('shows nothing without history', () => {
    expect(availableRanges({ dailyCount: 0, hasIntraday: false })).toEqual([]);
  });
});

describe('defaultRangeId', () => {
  it('defaults to the previous week', () => {
    expect(defaultRangeId(['24H', '5D', '1M', 'MAX'])).toBe('5D');
  });

  it('falls back to 24H when the listing is days old', () => {
    expect(defaultRangeId(['24H', 'MAX'])).toBe('24H');
  });

  it('falls back to MAX for two sessions and no intraday', () => {
    expect(defaultRangeId(['MAX'])).toBe('MAX');
  });

  it('returns null with no ranges', () => {
    expect(defaultRangeId([])).toBeNull();
  });
});

describe('sliceRangePoints', () => {
  it('slices the last N daily sessions', () => {
    const { labels, prices, isIntraday } = sliceRangePoints(dailyHistory(100), '5D');
    expect(labels).toHaveLength(5);
    expect(prices).toHaveLength(5);
    expect(isIntraday).toBe(false);
    expect(prices[4]).toBe(199);
  });

  it('returns everything for MAX', () => {
    expect(sliceRangePoints(dailyHistory(3000), 'MAX').prices).toHaveLength(3000);
  });

  it('returns intraday bars for 24H', () => {
    const slice = sliceRangePoints(dailyHistory(100), '24H');
    expect(slice.isIntraday).toBe(true);
    expect(slice.labels[0]).toContain('T');
  });
});

describe('periodChange', () => {
  it('computes signed change and percent', () => {
    expect(periodChange([100, 110])).toEqual({ change: 10, changePct: 10, up: true });
    expect(periodChange([100, 90]).up).toBe(false);
  });

  it('is neutral on short series', () => {
    expect(periodChange([5])).toEqual({ change: 0, changePct: 0, up: true });
  });
});
