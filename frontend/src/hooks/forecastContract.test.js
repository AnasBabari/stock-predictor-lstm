import { describe, expect, it } from 'vitest';
import { describePromotionState } from '../ml/promotionPolicy';
import {
  FORECAST_TYPES,
  browserResponse,
} from '../hooks/useForecast';

const snapshot = {
  ticker: 'MSFT',
  schema_version: 4,
  window_size: 60,
  output_width: 30,
  feature_names: ['a'],
  snapshot_id: 'snap-1',
  dates: ['2025-01-02'],
  historical_prices: [100],
  future_dates: ['2025-01-03'],
  data_snapshot: {},
};

describe('describePromotionState', () => {
  it('maps a promoted model to the learned decision with alpha 1', () => {
    const status = describePromotionState({ promoted: true, applicable: true });
    expect(status).toMatchObject({ state: 'promoted', decision: 'model', alpha: 1 });
  });

  it('maps rejection to the persistence decision with alpha 0 and honest language', () => {
    const status = describePromotionState({ promoted: false, applicable: true });
    expect(status.state).toBe('experimental_no_demonstrated_edge');
    expect(status.decision).toBe('persistence');
    expect(status.alpha).toBe(0);
    expect(status.label).toMatch(/did not beat persistence/i);
  });

  it('fails closed for missing or non-applicable gates instead of assuming promotion', () => {
    const unknown = describePromotionState(null);
    expect(unknown.state).toBe('status_unknown');
    expect(unknown.decision).toBe('persistence');
    expect(unknown.alpha).toBe(0);

    const notApplicable = describePromotionState({ promoted: true, applicable: false });
    expect(notApplicable.decision).toBe('persistence');
    expect(notApplicable.state).toBe('status_unknown');
  });
});

describe('browserResponse contract — price', () => {
  it('keeps the learned path as predicted_prices when promoted and exposes persistence separately', () => {
    const response = browserResponse(
      snapshot,
      {
        predictedPrices: [103],
        learnedPrices: [103],
        persistence_forecast: [100],
        baselineFallback: false,
        promotion: { promoted: true },
        forecast_status: { state: 'promoted', decision: 'model', alpha: 1, label: 'x' },
        metrics: {},
      },
      FORECAST_TYPES.PRICE,
      1
    );
    expect(response.predicted_prices).toEqual([103]);
    expect(response.learned_prices).toEqual([103]);
    expect(response.persistence_forecast).toEqual([100]);
    expect(response.forecast_status.state).toBe('promoted');
  });

  it('never lets the fallback overwrite the learned path field', () => {
    const response = browserResponse(
      snapshot,
      {
        predictedPrices: [100], // decision = flat
        learnedPrices: [107],   // raw model path preserved
        persistence_forecast: [100],
        baselineFallback: true,
        promotion: { promoted: false, reasons: ['Relative MAE did not beat persistence.'] },
        forecast_status: { state: 'experimental_no_demonstrated_edge', decision: 'persistence', alpha: 0, label: 'y' },
        metrics: {},
      },
      FORECAST_TYPES.PRICE,
      1
    );
    expect(response.predicted_prices).toEqual([100]);
    expect(response.learned_prices).toEqual([107]);
    expect(response.persistence_forecast).toEqual([100]);
    expect(response.forecast_status.alpha).toBe(0);
    expect(response.metadata.engine.baseline_fallback).toBe(true);
  });
});

describe('browserResponse contract — direction v2', () => {
  it('exposes the three-way decision plus model and base-rate distributions', () => {
    const response = browserResponse(
      snapshot,
      {
        direction_horizon_days: 7,
        direction: 'Down',                       // argmax of the baseline distribution
        direction_probabilities: { down: 0.4, neutral: 0.25, up: 0.35 },
        model_direction_probabilities: { down: 0.1, neutral: 0.2, up: 0.7 },
        base_rate_direction_probabilities: { down: 0.5, neutral: 0.3, up: 0.2 },
        baselineFallback: true,
        forecast_status: { state: 'experimental_no_demonstrated_edge', decision: 'persistence', alpha: 0, label: 'z' },
        metrics: {},
      },
      FORECAST_TYPES.TREND,
      7
    );
    expect(response.direction_horizon_days).toBe(7);
    expect(response.direction).toBe('Down');
    expect(response.direction_probabilities).toEqual({ down: 0.4, neutral: 0.25, up: 0.35 });
    expect(response.model_direction_probabilities.up).toBe(0.7);
    expect(response.base_rate_direction_probabilities.down).toBe(0.5);
    expect(response.forecast_status.decision).toBe('persistence');
    // Legacy per-day arrays must not exist under the v2 contract.
    expect(response.directions).toBeUndefined();
    expect(response.probabilities).toBeUndefined();
  });
});
