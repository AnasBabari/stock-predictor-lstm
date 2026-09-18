import { describe, expect, it } from 'vitest';
import { getSharedEstimatePresentation } from './estimateAdapter';

describe('getSharedEstimatePresentation', () => {
  const baseForecast = {
    ticker: 'MSFT',
    data_as_of: '2026-09-09',
    current_price: 491.65,
    lower_prices: [482.45, 479.5, 479.4, 480.14, 478.38, 473.39, 468.65],
    upper_prices: [495.1, 495.46, 498.41, 499.71, 499.61, 495.38, 492.41],
    future_dates: [
      '2026-09-10',
      '2026-09-11',
      '2026-09-14',
      '2026-09-15',
      '2026-09-16',
      '2026-09-17',
      '2026-09-18',
    ],
  };

  const baseHistory = {
    ticker: 'MSFT',
    asOf: '2026-09-09',
    daily: [
      { d: '2026-09-08', c: 493.95 },
      { d: '2026-09-09', c: 491.64999 },
    ],
  };

  it('computes exact midpoint series and matches final price and percentage change', () => {
    const res = getSharedEstimatePresentation({
      forecast: baseForecast,
      history: baseHistory,
      currencySymbol: '$',
    });

    expect(res.isAvailable).toBe(true);
    expect(res.isMismatch).toBe(false);
    expect(res.series).toHaveLength(7);
    expect(res.series[0]).toBeCloseTo((482.45 + 495.1) / 2, 4);
    expect(res.series[6]).toBeCloseTo((468.65 + 492.41) / 2, 4);
    expect(res.finalPrice).toBe(res.series[6]);
    expect(res.changePct).toBeCloseTo(((res.finalPrice / 491.65) - 1) * 100, 4);
    expect(res.direction).toBe('down');
  });

  it('marks unavailable when lower or upper bounds are missing or empty', () => {
    const withoutBounds = { ...baseForecast, lower_prices: [], upper_prices: [] };
    const res = getSharedEstimatePresentation({ forecast: withoutBounds });
    expect(res.isAvailable).toBe(false);
    expect(res.reason).toBe('missing_bounds');
    expect(res.series).toEqual([]);
  });

  it('detects ticker mismatch between forecast and history', () => {
    const mismatchedHistory = { ...baseHistory, ticker: 'AAPL' };
    const res = getSharedEstimatePresentation({
      forecast: baseForecast,
      history: mismatchedHistory,
    });
    expect(res.isAvailable).toBe(false);
    expect(res.isMismatch).toBe(true);
    expect(res.mismatchReason).toContain('Ticker mismatch');
  });

  it('detects origin date mismatch between forecast and history', () => {
    const mismatchedHistory = {
      ...baseHistory,
      asOf: '2026-09-08',
      daily: [{ d: '2026-09-08', c: 491.65 }],
    };
    const res = getSharedEstimatePresentation({
      forecast: baseForecast,
      history: mismatchedHistory,
    });
    expect(res.isAvailable).toBe(false);
    expect(res.isMismatch).toBe(true);
    expect(res.mismatchReason).toContain('Origin date mismatch');
  });

  it('detects origin price mismatch at display precision (like $315.34 vs $491.65)', () => {
    const mismatchedHistory = {
      ...baseHistory,
      daily: [
        { d: '2026-09-08', c: 310.0 },
        { d: '2026-09-09', c: 315.34 },
      ],
    };
    const res = getSharedEstimatePresentation({
      forecast: baseForecast,
      history: mismatchedHistory,
      currencySymbol: '$',
    });
    expect(res.isAvailable).toBe(false);
    expect(res.isMismatch).toBe(true);
    expect(res.mismatchReason).toContain('Origin price mismatch');
  });

  it('handles UK pence precision correctly for .L tickers', () => {
    const ukForecast = {
      ticker: 'SHEL.L',
      data_as_of: '2026-09-09',
      current_price: 2600.54,
      lower_prices: [2580.0, 2590.0],
      upper_prices: [2620.0, 2630.0],
      future_dates: ['2026-09-10', '2026-09-11'],
    };
    const ukHistory = {
      ticker: 'SHEL.L',
      asOf: '2026-09-09',
      daily: [{ d: '2026-09-09', c: 2600.51 }], // rounds to 2600.5p at 1 decimal
    };
    const res = getSharedEstimatePresentation({
      forecast: ukForecast,
      history: ukHistory,
      currencySymbol: 'p',
    });
    expect(res.isAvailable).toBe(true);
    expect(res.isMismatch).toBe(false);
  });
});
