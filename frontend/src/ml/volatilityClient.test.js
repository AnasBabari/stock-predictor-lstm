import { describe, expect, it } from 'vitest';
import {
  fetchVolatilityForecast,
  mapVolatilityResponse,
  validateVolatilityResponse,
  VolatilityApiError,
} from './volatilityClient';

function body(overrides = {}) {
  const days = 7;
  const dates = Array.from({ length: 6 }, (_, index) => `2026-08-${15 + index}`);
  const future = Array.from({ length: days }, (_, index) => `2026-08-${21 + index}`);
  const values = Array.from({ length: days }, (_, index) => 100 + index * 0.5);
  return {
    ticker: 'MSFT',
    as_of: '2026-08-20',
    horizon: days,
    current_price: 100,
    historical_dates: dates,
    historical_prices: [98, 99, 100, 101, 100, 100],
    forecast: {
      future_dates: future,
      price_quantiles: {
        p05: values.map((value) => value - 3),
        p10: values.map((value) => value - 2),
        p25: values.map((value) => value - 1),
        p50: Array(days).fill(100),
        p75: values.map((value) => value + 1),
        p90: values.map((value) => value + 2),
        p95: values.map((value) => value + 3),
      },
      probability_up: null,
    },
    evidence: {
      certified: true,
      certified_heads: { volatility: true, return_distribution: false, direction: false },
      metric_source: 'locked_purged_walk_forward',
      model_id: 'global-volatility-tcn-v1',
      snapshot_id: 'snapshot-1',
      horizon_certification: { '7': { decision: 'pass', relative_qlike: 0.91 } },
    },
    ...overrides,
  };
}

describe('volatility client contract', () => {
  it('maps a certified volatility response without exposing a flat price forecast', () => {
    const result = mapVolatilityResponse(body(), 'MSFT', 7);
    expect(result.predicted_prices).toBeNull();
    expect(result.persistence_forecast).toBeNull();
    expect(result.historical_error_band.lower_prices).toHaveLength(7);
    expect(result.forecast_status.decision).toBe('volatility_cone');
    expect(result.validation.promoted).toBe(true);
    expect(result.metadata.engine.role).toBe('server_artifact_loaded');
    expect(result.metrics.relative_qlike).toBe(0.91);
  });

  it('preserves the structured abstention code for safe UI mapping', async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: false,
      status: 503,
      json: async () => ({
        detail: {
          status: 'abstain_no_certified_model',
          reason: 'no certified volatility release is configured',
        },
      }),
    }));
    await expect(
      fetchVolatilityForecast('MSFT', 7, undefined, { baseUrl: 'https://api.test', fetchImpl }),
    ).rejects.toMatchObject({
      name: 'VolatilityApiError',
      code: 'abstain_no_certified_model',
      httpStatus: 503,
    });
    await fetchVolatilityForecast('MSFT', 7, undefined, { baseUrl: 'https://api.test', fetchImpl })
      .catch((error) => expect(error).toBeInstanceOf(VolatilityApiError));
  });

  it('rejects missing or non-certified volatility evidence', () => {
    expect(() => validateVolatilityResponse(body({ evidence: {} }), 'MSFT', 7)).toThrow(/certified/);
    expect(() => validateVolatilityResponse(body({ ticker: 'AAPL' }), 'MSFT', 7)).toThrow(/match/);
  });

  it('rejects a non-chronological future path', () => {
    const invalid = body();
    invalid.forecast.future_dates = ['2026-08-21', '2026-08-23', '2026-08-22', '2026-08-24', '2026-08-25', '2026-08-26', '2026-08-27'];
    expect(() => validateVolatilityResponse(invalid, 'MSFT', 7)).toThrow(/chronological/);
  });
});
