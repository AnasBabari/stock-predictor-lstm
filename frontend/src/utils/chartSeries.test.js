import { describe, expect, it } from 'vitest';
import { buildPriceSeries, directionStatusText } from './chartSeries';

const samplePayload = {
  ticker: 'MSFT',
  historical_dates: ['2025-01-02', '2025-01-03', '2025-01-06'],
  historical_prices: [100, 101, 102],
  future_dates: ['2025-01-07', '2025-01-08'],
  predicted_prices: [104, 108.8], // Always the model forecast
  learned_prices: [104, 108.8],
  persistence_forecast: [102, 102],
  validation: {
    state: 'experimental',
    promoted: false,
    reasons: ['Relative RMSE did not beat persistence.'],
  },
  forecast_status: {
    state: 'experimental',
    decision: 'model',
    alpha: 1,
    label: 'Model forecast shown for research; validation gates were not met.',
  },
};

const promotedPayload = {
  ...samplePayload,
  predicted_prices: [103.1, 104.2],
  validation: { state: 'promoted', promoted: true, reasons: [] },
  forecast_status: { state: 'promoted', decision: 'model', alpha: 1, label: 'Validated against persistence.' },
};

describe('buildPriceSeries — core contract', () => {
  it('always draws Model Forecast as the primary forecast line by default, with persistence hidden', () => {
    const series = buildPriceSeries(samplePayload, 21, true, false);
    expect(series).not.toBeNull();
    const labels = series.datasets.map((d) => d.label);
    expect(labels).toEqual(['Historical Price', 'Model Forecast']);
    expect(labels).not.toContain('Persistence Benchmark');
  });

  it('anchors the forecast line continuously to the last historical close', () => {
    const series = buildPriceSeries(samplePayload, 21, true, false);
    const forecastData = series.datasets[1].data;
    expect(forecastData[2]).toBe(102); // Anchor price
    expect(forecastData[3]).toBe(104);
    expect(forecastData[4]).toBe(108.8);
    expect(series.forecastSplitIndex).toBe(2);
  });

  it('includes Persistence Benchmark as a dashed line when showBenchmark is enabled', () => {
    const series = buildPriceSeries(samplePayload, 21, true, true);
    expect(series).not.toBeNull();
    const labels = series.datasets.map((d) => d.label);
    expect(labels).toContain('Persistence Benchmark');

    const bench = series.datasets.find((d) => d.label === 'Persistence Benchmark');
    expect(bench.borderDash).toEqual([4, 4]);
    expect(bench.data[2]).toBe(102);
    expect(bench.data[3]).toBe(102);
    expect(bench.data[4]).toBe(102);
  });

  it('includes historical forecast-error band when available', () => {
    const withBand = {
      ...samplePayload,
      historical_error_band: {
        lower_prices: [102.5, 106.0],
        upper_prices: [105.5, 111.0],
      },
    };
    const series = buildPriceSeries(withBand, 21, true, false);
    const labels = series.datasets.map((d) => d.label);
    expect(labels).toContain('90% Empirical Error Range (Upper)');
    expect(labels).toContain('90% Empirical Error Range (Lower)');
  });
});

describe('buildPriceSeries — guards', () => {
  it.each([
    [null],
    [{}],
    [{ ...samplePayload, predicted_prices: undefined }],
    [{ ...samplePayload, historical_dates: [] }],
    [{ ...samplePayload, future_dates: [] }],
  ])('returns null for incomplete payloads (%#)', (payload) => {
    expect(buildPriceSeries(payload, 21, true)).toBeNull();
  });
});
