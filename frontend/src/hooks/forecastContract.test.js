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

describe('describePromotionState v2', () => {
  it('maps a promoted model to promoted state with decision model and alpha 1', () => {
    const status = describePromotionState({ promoted: true, applicable: true });
    expect(status).toMatchObject({ state: 'promoted', decision: 'model', alpha: 1 });
  });

  it('maps rejection to experimental state with decision model and alpha 1 (model line preserved)', () => {
    const status = describePromotionState({ promoted: false, state: 'experimental', applicable: true });
    expect(status.state).toBe('experimental');
    expect(status.decision).toBe('model');
    expect(status.alpha).toBe(1);
    expect(status.label).toMatch(/validation gates were not met/i);
  });

  it('fails closed for missing or non-applicable gates to unavailable state with decision model', () => {
    const unknown = describePromotionState(null);
    expect(unknown.state).toBe('unavailable');
    expect(unknown.decision).toBe('model');
    expect(unknown.alpha).toBe(1);

    const notApplicable = describePromotionState({ promoted: true, applicable: false });
    expect(notApplicable.decision).toBe('model');
    expect(notApplicable.state).toBe('unavailable');
  });
});

describe('browserResponse contract — price', () => {
  it('exposes model forecast as predicted_prices and persistence as separate benchmark', () => {
    const response = browserResponse(
      snapshot,
      {
        predictedPrices: [104],
        learnedPrices: [104],
        persistence_forecast: [100],
        baselineFallback: false,
        promotion: { promoted: true },
        forecast_status: { state: 'promoted', decision: 'model', alpha: 1, label: 'Validated.' },
        metrics: {},
      },
      FORECAST_TYPES.PRICE,
      1
    );
    expect(response.predicted_prices).toEqual([104]);
    expect(response.model_forecast.prices).toEqual([104]);
    expect(response.benchmark.prices).toEqual([100]);
    expect(response.persistence_forecast).toEqual([100]);
    expect(response.validation.promoted).toBe(true);
  });

  it('always retains model output in predicted_prices even when unpromoted', () => {
    const response = browserResponse(
      snapshot,
      {
        predictedPrices: [107], // Model output
        learnedPrices: [107],
        persistence_forecast: [100],
        baselineFallback: true,
        promotion: { promoted: false, reasons: ['Relative MAE did not beat persistence.'] },
        forecast_status: { state: 'experimental', decision: 'model', alpha: 1, label: 'Experimental' },
        metrics: {},
      },
      FORECAST_TYPES.PRICE,
      1
    );
    expect(response.predicted_prices).toEqual([107]);
    expect(response.model_forecast.prices).toEqual([107]);
    expect(response.benchmark.prices).toEqual([100]);
    expect(response.persistence_forecast).toEqual([100]);
    expect(response.validation.promoted).toBe(false);
  });
});

describe('browserResponse contract — direction v2', () => {
  it('exposes the three-way decision plus model and base-rate distributions', () => {
    const response = browserResponse(
      snapshot,
      {
        direction_horizon_days: 7,
        direction: 'Down',
        direction_probabilities: { down: 0.4, neutral: 0.25, up: 0.35 },
        model_direction_probabilities: { down: 0.1, neutral: 0.2, up: 0.7 },
        base_rate_direction_probabilities: { down: 0.4, neutral: 0.25, up: 0.35 },
        forecast_status: { state: 'experimental', decision: 'base_rate', alpha: 0, label: 'z' },
        metrics: {},
      },
      FORECAST_TYPES.TREND,
      7
    );
    expect(response.direction).toBe('Down');
    expect(response.direction_probabilities).toEqual({ down: 0.4, neutral: 0.25, up: 0.35 });
    expect(response.model_direction_probabilities).toEqual({ down: 0.1, neutral: 0.2, up: 0.7 });
    expect(response.base_rate_direction_probabilities).toEqual({ down: 0.4, neutral: 0.25, up: 0.35 });
  });
});
