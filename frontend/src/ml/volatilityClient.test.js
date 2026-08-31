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

  it('maps the active causal baseline without claiming certification', () => {
    const result = mapVolatilityResponse(body({
      forecast: {
        ...body().forecast,
        model: 'har_rv',
        expected_annualized_volatility: 0.21,
      },
      evidence: {
        model_status: 'baseline',
        model_family: 'statistical_baseline',
        metric_source: 'baseline_definition',
        schema_version: 'deployable_v5',
        snapshot_id: 'snapshot-1',
      },
    }), 'MSFT', 7);
    expect(result.predicted_prices).toBeNull();
    expect(result.forecast_status).toMatchObject({ state: 'baseline', decision: 'baseline' });
    expect(result.validation.promoted).toBe(false);
    expect(result.metadata.engine).toMatchObject({
      role: 'baseline_forecast',
      execution_mode: 'baseline',
      baseline_fallback: true,
      volatility_forecast: true,
    });
    expect(result.metadata.engine.certified_head).toBeUndefined();
    expect(result.metrics.metric_source).toBe('baseline_definition');
  });

  it('requests the active v1 volatility endpoint', async () => {
    const fetchImpl = vi.fn(async () => ({ ok: true, json: async () => body() }));
    await fetchVolatilityForecast('MSFT', 7, undefined, { baseUrl: 'https://api.test', fetchImpl });
    expect(fetchImpl.mock.calls[0][0]).toContain('/api/v1/volatility/forecast');
  });

  it('maps a certified Student-t return distribution to a learned median path', () => {
    const location = 0.035;
    const distribution = body({
      forecast: {
        ...body().forecast,
        price_quantiles: {
          ...body().forecast.price_quantiles,
          p50: Array.from({ length: 7 }, (_, index) => 100 * Math.exp(location * (index + 1) / 7)),
        },
        expected_cumulative_return: location,
        return_distribution_variance: 0.0006,
        return_distribution_family: 'student_t',
      },
      evidence: {
        ...body().evidence,
        certified_heads: { volatility: true, return_distribution: true, direction: false },
      },
    });
    const result = mapVolatilityResponse(distribution, 'MSFT', 7);
    expect(result.predicted_prices).toEqual(result.volatility_cone.p50);
    expect(result.learned_prices).toEqual(result.volatility_cone.p50);
    expect(result.predicted_prices.at(-1)).toBeCloseTo(100 * Math.exp(location), 8);
    expect(result.persistence_forecast).toEqual(Array(7).fill(100));
    expect(result.forecast_status).toMatchObject({
      state: 'certified_return_distribution',
      decision: 'return_distribution',
    });
    expect(result.metadata.engine).toMatchObject({
      certified_head: 'return_distribution',
      location_source: 'certified_return_location',
    });
  });

  it('rejects a certified return distribution with a mismatched median path', () => {
    const invalid = body({
      forecast: {
        ...body().forecast,
        expected_cumulative_return: 0.035,
        return_distribution_variance: 0.0006,
        return_distribution_family: 'student_t',
      },
      evidence: {
        ...body().evidence,
        certified_heads: { volatility: true, return_distribution: true, direction: false },
      },
    });
    expect(() => validateVolatilityResponse(invalid, 'MSFT', 7)).toThrow(/p50 path/);
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

  it('rejects missing or unrecognised volatility evidence', () => {
    expect(() => validateVolatilityResponse(body({ evidence: {} }), 'MSFT', 7)).toThrow(/evidence/);
    expect(() => validateVolatilityResponse(body({ ticker: 'AAPL' }), 'MSFT', 7)).toThrow(/match/);
  });

  it('rejects a non-chronological future path', () => {
    const invalid = body();
    invalid.forecast.future_dates = ['2026-08-21', '2026-08-23', '2026-08-22', '2026-08-24', '2026-08-25', '2026-08-26', '2026-08-27'];
    expect(() => validateVolatilityResponse(invalid, 'MSFT', 7)).toThrow(/chronological/);
  });
});
