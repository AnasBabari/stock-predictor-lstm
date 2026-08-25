/**
 * Client for the signed global-volatility serving contract.
 *
 * The server model certifies conditional volatility, not a directional price
 * level. This adapter deliberately keeps the p50 path at the latest close and
 * labels the uncertainty cone and its metric source so the UI cannot present
 * a zero-return baseline as a learned price claim.
 */

export const VOLATILITY_HORIZONS = [1, 3, 5, 7, 14, 30];
const QUANTILE_KEYS = ['p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95'];

function apiErrorBody(body) {
  const detail = body?.detail;
  if (typeof detail === 'string') return detail;
  if (detail && typeof detail === 'object') {
    return detail.reason || detail.message || detail.status || 'Certified volatility model unavailable.';
  }
  return 'Certified volatility model unavailable.';
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
  if (Math.abs(quantiles.p50.at(-1) - currentPrice) > Math.max(currentPrice * 1e-6, 1e-6)) {
    throw new Error('Volatility response p50 path is not anchored to the unchanged close.');
  }
  if (body.evidence?.certified !== true || body.evidence?.certified_heads?.volatility !== true) {
    throw new Error('Volatility response is not backed by a certified volatility head.');
  }
  return { ...body, current_price: currentPrice, historical_dates: historicalDates, historical_prices: historicalPrices, quantiles };
}

export function mapVolatilityResponse(body, ticker, days) {
  const data = validateVolatilityResponse(body, ticker, days);
  const summary = data.evidence?.horizon_certification?.[String(days)] || {};
  const p50 = data.quantiles.p50;
  const persistence = Array(days).fill(data.current_price);
  const metricSource = data.evidence?.metric_source || 'locked_purged_walk_forward';
  return {
    ticker: data.ticker,
    forecast_days: days,
    forecast_type: 'price',
    current_price: data.current_price,
    historical_dates: data.historical_dates,
    historical_prices: data.historical_prices,
    future_dates: data.forecast.future_dates,
    predicted_prices: p50,
    persistence_forecast: persistence,
    learned_prices: null,
    benchmark: { type: 'persistence', prices: persistence },
    historical_error_band: {
      lower_prices: data.quantiles.p05,
      upper_prices: data.quantiles.p95,
      source: 'certified_volatility_cone',
    },
    volatility_cone: data.quantiles,
    forecast_status: {
      state: 'certified_volatility',
      decision: 'persistence',
      alpha: 0,
      label: 'Certified volatility; unchanged-close location baseline',
    },
    validation: {
      state: 'volatility_certified_location_baseline',
      promoted: false,
      selected_horizon: days,
      best_validated_horizon: days,
      promoted_horizons: data.evidence?.certified_heads?.volatility ? [days] : [],
      reasons: ['Only conditional volatility is certified; return location remains the matched persistence baseline.'],
    },
    metrics: {
      metric_source: metricSource,
      relative_qlike: summary.relative_qlike ?? null,
      ratio_upper_95: summary.ratio_upper_95 ?? null,
      dm_p_value: summary.dm_p_value ?? null,
      coverage_80: summary.coverage_80 ?? null,
      coverage_95: summary.coverage_95 ?? null,
      evaluation_rows: summary.evaluation_rows ?? null,
      model_head: 'volatility',
    },
    metadata: {
      schema_version: 'deployable_v5',
      feature_count: data.evidence?.feature_count ?? null,
      window_size: 60,
      snapshot_id: data.evidence?.snapshot_id,
      model_version: data.evidence?.model_id,
      metric_source: metricSource,
      browser_training: false,
      engine: {
        family: 'global_volatility_tcn',
        role: 'server_artifact_loaded',
        execution_mode: 'server_artifact_loaded',
        baseline_fallback: false,
        certified_head: 'volatility',
        location_source: 'matched_persistence',
      },
      data_snapshot: { as_of: data.as_of, source: 'server_causal_market_snapshot' },
    },
    evidence: data.evidence,
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
    `${baseUrl}/api/v2/forecast?ticker=${encodeURIComponent(requestTicker)}&horizon=${horizon}`,
    { signal, cache: 'no-cache' },
  );
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(apiErrorBody(body));
  }
  return mapVolatilityResponse(body, requestTicker, horizon);
}
