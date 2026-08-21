import { describe, expect, it } from 'vitest';
import { buildPriceSeries, directionStatusText } from './chartSeries';

const base = {
  ticker: 'MSFT',
  historical_dates: ['2025-01-02', '2025-01-03', '2025-01-06'],
  historical_prices: [100, 101, 102],
  future_dates: ['2025-01-07', '2025-01-08'],
  predicted_prices: [102, 102], // decision path (flat baseline in fallback case)
  learned_prices: [104, 108.8],
  forecast_status: {
    state: 'experimental_no_demonstrated_edge',
    decision: 'persistence',
    alpha: 0,
    label: 'Experimental model did not beat persistence…',
  },
};

const promoted = {
  ...base,
  predicted_prices: [103.1, 104.2],
  forecast_status: { state: 'promoted', decision: 'model', alpha: 1, label: 'Promoted…' },
};

describe('buildPriceSeries — promoted', () => {
  const series = buildPriceSeries(promoted, 21, true);

  it('draws exactly historical + model forecast, labelled as a prediction', () => {
    expect(series).not.toBeNull();
    expect(series.datasets.map((d) => d.label)).toEqual(['Historical Price', 'Predicted Price']);
    expect(series.promoted).toBe(true);
    expect(series.annotation).toBeNull();
  });

  it('anchors the forecast line to the last historical close', () => {
    const decision = series.datasets[1].data;
    expect(decision[2]).toBe(102);
    expect(decision[3]).toBe(103.1);
    expect(decision[4]).toBe(104.2);
  });
});

describe('buildPriceSeries — fallback (decision = persistence)', () => {
  const series = buildPriceSeries(base, 21, false);

  it('labels the decision path as the no-change baseline, never as the model', () => {
    const labels = series.datasets.map((d) => d.label);
    expect(labels).toContain('No-change baseline');
    expect(labels).not.toContain('Predicted Price');
    expect(series.promoted).toBe(false);
  });

  it('still draws the raw learned path as a dashed diagnostic', () => {
    const learned = series.datasets.find((d) => d.label === 'Learned model (not promoted)');
    expect(learned).toBeDefined();
    expect(learned.borderDash.length).toBeGreaterThan(0);
    // Anchored to last close, then the model's own values.
    expect(learned.data[2]).toBe(102);
    expect(learned.data[3]).toBe(104);
    expect(learned.data[4]).toBe(108.8);
  });

  it('explains the substitution explicitly', () => {
    expect(series.annotation).toMatch(/no-change baseline/i);
    expect(series.annotation).toMatch(/did not beat persistence/i);
  });
});

describe('buildPriceSeries — guards', () => {
  it.each([
    [null],
    [{}],
    [{ ...base, predicted_prices: undefined }],
    [{ ...base, historical_dates: [] }],
    [{ ...base, future_dates: [] }],
  ])('returns null for incomplete payloads (%#)', (payload) => {
    expect(buildPriceSeries(payload, 21, true)).toBeNull();
  });
});

describe('directionStatusText', () => {
  it('uses base-rate language for non-promoted direction models', () => {
    expect(directionStatusText({ state: 'experimental_no_demonstrated_edge' })).toMatch(
      /base rate/
    );
  });
  it('is empty without status', () => {
    expect(directionStatusText(undefined)).toBe('');
  });
});
