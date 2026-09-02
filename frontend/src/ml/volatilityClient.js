/**
 * Client for the active causal volatility contract.
 *
 * The active endpoint returns a transparent statistical baseline.  The old
 * signed global-release response is still accepted for compatibility, but it
 * is no longer required for ordinary forecasts.
 */

import { VOLATILITY_HORIZONS } from './volatilityContract';

export { VOLATILITY_HORIZONS } from './volatilityContract';
const QUANTILE_KEYS = ['p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95'];

function apiErrorBody(body) {
  const detail = body?.detail;
  if (typeof detail === 'string') {
    return { code: null, reason: detail };
  }
  if (detail && typeof detail === 'object') {
    return {
      code: detail.status || detail.code || null,
      reason: detail.reason || detail.message || 'Volatility forecast unavailable.',
    };
  }
  return { code: null, reason: 'Volatility forecast unavailable.' };
}

export class VolatilityApiError extends Error {
  constructor(message, { code = null, httpStatus = null } = {}) {
    super(message);
    this.name = 'VolatilityApiError';
    this.code = code;
    this.httpStatus = httpStatus;
  }
}

function finitePositive(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) {
    throw new Error(`Volatility response contains an invalid ${label}.`);
  }
  return number;
}

function strictDates(values, label) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error(`Volatility response contains no ${label}.`);
  }
  let previous = 0;
  values.forEach((value, index) => {
    const parsed = Date.parse(value);
    if (!Number.isFinite(parsed) || (index > 0 && parsed <= previous)) {
      throw new Error(`Volatility response contains non-chronological ${label}.`);
    }
    previous = parsed;
  });
  return values;
}

function quantileSeries(forecast, key, days) {
  const values = forecast?.price_quantiles?.[key];
  if (!Array.isArray(values) || values.length !== days) {
    throw new Error(`Volatility response is missing the ${key} uncertainty path.`);
  }
  return values.map((value) => finitePositive(value, `${key} quantile`));
}

export function validateVolatilityResponse(body, ticker, days) {
  const symbol = String(ticker).trim().toUpperCase();
  if (!body || body.ticker !== symbol || Number(body.horizon) !== days) {
    throw new Error('Volatility response does not match the selected ticker and horizon.');
  }
  const currentPrice = finitePositive(body.current_price, 'current price');
  const historicalDates = strictDates(body.historical_dates, 'historical dates');
  if (!Array.isArray(body.historical_prices) || body.historical_prices.length !== historicalDates.length) {
    throw new Error('Volatility response history is misaligned.');
  }
  const historicalPrices = body.historical_prices.map((value) => finitePositive(value, 'historical price'));
  const futureDates = strictDates(body.forecast?.future_dates, 'future dates');
  if (futureDates.length !== days || Date.parse(futureDates[0]) <= Date.parse(historicalDates.at(-1))) {
    throw new Error('Volatility response future dates do not follow the market history.');
  }
  const quantiles = Object.fromEntries(
    QUANTILE_KEYS.map((key) => [key, quantileSeries(body.forecast, key, days)])
  );
  const isBaseline = body.evidence?.model_status === 'baseline';
  const isLegacyCertified = body.evidence?.certified === true
    && body.evidence?.certified_heads?.volatility === true;
  if (!isBaseline && !isLegacyCertified) {
    throw new Error('Volatility response has no recognised baseline or learned-model evidence.');
  }
  if (
    body.evidence?.forecast_fingerprint != null
    && !/^[0-9a-f]{64}$/i.test(String(body.evidence.forecast_fingerprint))
  ) {
    throw new Error('Volatility response contains an invalid forecast fingerprint.');
  }
  const hasReturnDistribution = !isBaseline
    && body.evidence?.certified_heads?.return_distribution === true;
  if (hasReturnDistribution) {
    if (body.forecast?.return_distribution_family !== 'student_t') {
      throw new Error('Certified return distribution must declare the Student-t family.');
    }
    const expectedReturn = body.forecast?.expected_cumulative_return;
    if (
      expectedReturn == null
      || typeof expectedReturn === 'boolean'
      || !Number.isFinite(Number(expectedReturn))
    ) {
      throw new Error('Certified return distribution is missing its expected return location.');
    }
    if (!finitePositive(body.forecast?.return_distribution_variance, 'return distribution variance')) {
      throw new Error('Certified return distribution variance is invalid.');
    }
    const expectedMedian = currentPrice * Math.exp(Number(expectedReturn));
    if (
      !Number.isFinite(expectedMedian)
      || Math.abs(quantiles.p50.at(-1) - expectedMedian) > Math.max(expectedMedian * 1e-5, 1e-5)
    ) {
      throw new Error('Certified return-distribution location does not match the p50 path.');
    }
  } else if (Math.abs(quantiles.p50.at(-1) - currentPrice) > Math.max(currentPrice * 1e-6, 1e-6)) {
    throw new Error('Volatility-only response p50 path is not anchored to the unchanged close.');
  }
  return { ...body, current_price: currentPrice, historical_dates: historicalDates, historical_prices: historicalPrices, quantiles };
}

export function mapVolatilityResponse(body, ticker, days) {
  const data = validateVolatilityResponse(body, ticker, days);
  const isBaseline = data.evidence?.model_status === 'baseline';
  const summary = data.evidence?.horizon_certification?.[String(days)] || {};
  const metricSource = data.evidence?.metric_source || (isBaseline ? 'baseline_definition' : 'locked_purged_walk_forward');
  const evidence = data.evidence || {};
  const dataAsOf = evidence.data_as_of || data.as_of;
  const modelVersion = evidence.model_version || evidence.model_id || data.forecast?.model;
  const hasReturnDistribution = data.evidence?.certified_heads?.return_distribution === true;
  const learnedMedian = hasReturnDistribution ? data.quantiles.p50 : null;
  const summaryMetrics = summary?.metrics && typeof summary.metrics === 'object'
    ? summary.metrics
    : {};
  return {
    ticker: data.ticker,
    forecast_days: days,
    forecast_type: 'price',
    current_price: data.current_price,
    forecast: data.forecast,
    historical_dates: data.historical_dates,
    historical_prices: data.historical_prices,
    future_dates: data.forecast.future_dates,
    // A certified Student-t return-distribution location is a learned median
    // path. Legacy volatility-only releases deliberately expose no point path.
    predicted_prices: learnedMedian,
    persistence_forecast: hasReturnDistribution ? Array(days).fill(data.current_price) : null,
    learned_prices: learnedMedian,
    benchmark: null,
    historical_error_band: {
      lower_prices: data.quantiles.p05,
      upper_prices: data.quantiles.p95,
      source: isBaseline
        ? 'causal_statistical_baseline'
        : hasReturnDistribution ? 'certified_return_distribution' : 'certified_volatility_cone',
    },
    volatility_cone: data.quantiles,
    forecast_status: {
      state: isBaseline
        ? 'baseline'
        : hasReturnDistribution ? 'certified_return_distribution' : 'certified_volatility',
      decision: isBaseline ? 'baseline' : hasReturnDistribution ? 'return_distribution' : 'volatility_cone',
      alpha: isBaseline ? 0 : 1,
      label: isBaseline
        ? `Causal ${data.forecast?.model || 'statistical'} volatility baseline`
        : hasReturnDistribution
        ? 'Certified Student-t return-distribution forecast'
        : 'Certified conditional-volatility forecast',
    },
    validation: {
      state: isBaseline
        ? 'baseline'
        : hasReturnDistribution ? 'certified_return_distribution' : 'certified_volatility',
      promoted: !isBaseline,
      selected_horizon: days,
      best_validated_horizon: days,
      promoted_horizons: !isBaseline && data.evidence?.certified_heads?.volatility ? [days] : [],
      reasons: [isBaseline
        ? 'This forecast is a transparent causal baseline; learned-model benchmark evidence is not loaded.'
        : hasReturnDistribution
        ? 'Terminal Student-t return location and variance cleared the sealed CRPS, QLIKE, and coverage gates; direction remains uncertified.'
        : 'Conditional volatility is certified; no learned return-location or direction claim is made.'],
    },
    metrics: {
      metric_source: metricSource,
      baseline: isBaseline,
      crps: summary.crps ?? summary.crps_mean ?? summaryMetrics.crps ?? summaryMetrics.crps_mean ?? null,
      relative_crps: summary.relative_crps ?? summary.relative_student_t_crps ?? summaryMetrics.relative_crps ?? null,
      relative_qlike: summary.relative_qlike ?? null,
      qlike: summary.qlike ?? summary.qlike_mean ?? summaryMetrics.qlike ?? summaryMetrics.qlike_mean ?? null,
      ratio_upper_95: summary.ratio_upper_95 ?? null,
      dm_p_value: summary.dm_p_value ?? null,
      coverage_80: summary.coverage_80 ?? null,
      coverage_95: summary.coverage_95 ?? null,
      evaluation_rows: summary.evaluation_rows ?? null,
      model_head: isBaseline ? 'baseline' : hasReturnDistribution ? 'return_distribution' : 'volatility',
    },
    metadata: {
      schema_version: data.evidence?.schema_version || 'deployable_v5',
      feature_count: data.evidence?.feature_count ?? null,
      window_size: 60,
      snapshot_id: data.evidence?.snapshot_id,
      model_version: modelVersion,
      model_policy_version: evidence.model_policy_version,
      selected_model: data.forecast?.model,
      requested_model: data.forecast?.requested_model || evidence.requested_model,
      forecast_fingerprint: evidence.forecast_fingerprint,
      code_commit: evidence.code_commit,
      data_as_of: dataAsOf,
      data_provider: evidence.data_provider,
      metric_source: metricSource,
      browser_training: false,
      engine: {
        family: data.evidence?.model_family || 'global_volatility_tcn',
        role: isBaseline ? 'baseline_forecast' : 'server_artifact_loaded',
        execution_mode: isBaseline ? 'baseline' : 'server_artifact_loaded',
        baseline_fallback: isBaseline,
        ...(isBaseline ? {} : { certified_head: hasReturnDistribution ? 'return_distribution' : 'volatility' }),
        location_source: hasReturnDistribution ? 'certified_return_location' : 'unchanged_close_reference',
        return_distribution_family: data.forecast?.return_distribution_family || 'zero_location_normal',
        volatility_forecast: true,
      },
      data_snapshot: {
        as_of: dataAsOf,
        source: isBaseline ? 'causal_market_snapshot' : 'server_causal_market_snapshot',
        provider: evidence.data_provider,
        snapshot_id: evidence.snapshot_id,
        market_data_cache: evidence.market_data_cache,
        code_commit: evidence.code_commit,
      },
    },
    evidence,
  };
}

export async function fetchVolatilityForecast(
  symbol,
  days,
  signal,
  { baseUrl = import.meta.env.VITE_API_URL || window.STOCKLSTM_API_BASE || '', fetchImpl = (...args) => globalThis.fetch(...args) } = {},
) {
  const requestTicker = String(symbol).trim().toUpperCase();
  const horizon = Number(days);
  if (!requestTicker || !VOLATILITY_HORIZONS.includes(horizon)) {
    throw new Error('Volatility forecast requires a supported ticker and horizon.');
  }
  const response = await fetchImpl(
    `${baseUrl}/api/v1/volatility/forecast?ticker=${encodeURIComponent(requestTicker)}&horizon=${horizon}`,
    { signal, cache: 'no-cache' },
  );
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const apiError = apiErrorBody(body);
    throw new VolatilityApiError(apiError.reason, {
      code: apiError.code,
      httpStatus: response.status,
    });
  }
  return mapVolatilityResponse(body, requestTicker, horizon);
}
