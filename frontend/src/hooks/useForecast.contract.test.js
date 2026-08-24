import { describe, expect, it } from 'vitest';

import {
  FORECAST_TYPES,
  assertForecastIdentity,
  forecastIdentity,
  normalizeHorizonRequest,
} from './useForecast';

describe('forecast horizon request contract', () => {
  it('keeps Auto distinct from numeric horizons', () => {
    expect(normalizeHorizonRequest('auto')).toEqual({
      horizon_mode: 'auto',
      requested_horizon: null,
    });
    expect(forecastIdentity('msft', 'auto', FORECAST_TYPES.PRICE, 'balanced'))
      .toBe('MSFT::auto::price::balanced');
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
});
