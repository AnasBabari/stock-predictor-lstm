import { describe, expect, it } from 'vitest';

import {
  FORECAST_TYPES,
  assertForecastIdentity,
  forecastIdentity,
  normalizeHorizonRequest,
  predictionErrorMessage,
} from './useForecast';
import { VolatilityApiError } from '../ml/volatilityClient';

describe('forecast horizon request contract', () => {
  it('keeps Auto distinct from numeric horizons', () => {
    expect(normalizeHorizonRequest('auto')).toEqual({
      horizon_mode: 'auto',
      requested_horizon: null,
    });
    expect(forecastIdentity('msft', 'auto', FORECAST_TYPES.PRICE))
      .toBe('MSFT::auto::price');
  });

  it('accepts a numeric Auto result only when it is explicitly labelled Auto', () => {
    const response = {
      ticker: 'MSFT',
      requested_horizon_mode: 'auto',
      forecast_days: 3,
      predicted_prices: [101, 102, 103],
      future_dates: ['2026-08-25', '2026-08-26', '2026-08-27'],
    };
    expect(assertForecastIdentity(response, 'MSFT', 'auto', FORECAST_TYPES.PRICE)).toBe(response);
    expect(() => assertForecastIdentity(
      { ...response, requested_horizon_mode: 'explicit' },
      'MSFT',
      'auto',
      FORECAST_TYPES.PRICE,
    )).toThrow(/does not match/);
  });

  it('rejects unsupported numeric horizons', () => {
    expect(() => normalizeHorizonRequest(2)).toThrow(/Forecast horizon/);
  });

  it('accepts a certified volatility distribution without a fabricated price path', () => {
    const response = {
      ticker: 'MSFT',
      forecast_days: 3,
      future_dates: ['2026-08-25', '2026-08-26', '2026-08-27'],
      predicted_prices: null,
      volatility_cone: Object.fromEntries(
        ['p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95'].map((key) => [key, [99, 100, 101]]),
      ),
      metadata: { engine: { certified_head: 'volatility' } },
    };
    expect(assertForecastIdentity(response, 'MSFT', 3, FORECAST_TYPES.PRICE)).toBe(response);
  });

  it('explains strict abstention instead of mislabelling it as capacity pressure', () => {
    const error = new VolatilityApiError('no certified release', {
      code: 'abstain_no_certified_model',
      httpStatus: 503,
    });
    expect(predictionErrorMessage(error)).toBe(
      'No certified global model is available yet. Forecasting is paused until a signed release passes validation.',
    );
  });

  it('does not mislabel a certified-service 503 as training capacity pressure', () => {
    const error = new VolatilityApiError('Volatility forecast failed (503): integrity failure', {
      code: 'artifact_integrity_failure',
      httpStatus: 503,
    });
    expect(predictionErrorMessage(error)).toBe(
      'The certified forecast service is temporarily unavailable. No baseline forecast was substituted.',
    );
  });
});
