const API_BASE = import.meta.env.VITE_API_URL || window.STOCKLSTM_API_BASE || '';

// Maps UI forecast labels to the wire `forecast_type` contract values.
// The UI says "trend"; the API/registry contract says "direction".
export const API_FORECAST_TYPES = {
  price: 'price',
  trend: 'direction',
};

const REQUIRED_FIELDS = [
  'ticker',
  'forecast_days',
  'future_dates',
  'predicted_prices',
  'metrics',
  'metadata',
];

function isBrowserFallback(z) {
  return Boolean(z) && z.available === false && z.fallback === 'browser_training';
}

function isStrictlyIncreasingIsoDates(dates) {
  if (!Array.isArray(dates) || dates.length === 0) return false;
  for (let i = 1; i < dates.length; i += 1) {
    const prev = Date.parse(dates[i - 1]);
    const next = Date.parse(dates[i]);
    if (Number.isNaN(prev) || Number.isNaN(next) || next <= prev) return false;
  }
  return true;
}

function isValidForecast(data, daysInt, apiType, requestTicker) {
  return (
    data &&
    REQUIRED_FIELDS.every((field) => data[field] !== undefined && data[field] !== null) &&
    typeof data.ticker === 'string' &&
    data.ticker === requestTicker &&
    Number.isInteger(data.forecast_days) &&
    data.forecast_days === daysInt &&
    Array.isArray(data.future_dates) &&
    data.future_dates.length === daysInt &&
    isStrictlyIncreasingIsoDates(data.future_dates) &&
    Array.isArray(data.predicted_prices) &&
    data.predicted_prices.length === daysInt &&
    data.predicted_prices.every((price) => Number.isFinite(price) && price > 0) &&
    (apiType !== 'price' ||
      (Array.isArray(data.historical_dates) &&
        data.historical_dates.length > 0 &&
        isStrictlyIncreasingIsoDates(data.historical_dates) &&
        Array.isArray(data.historical_prices) &&
        data.historical_dates.length === data.historical_prices.length &&
        data.historical_prices.every((price) => Number.isFinite(price) && price > 0)))
  );
}

function detailEnvelope(data) {
  return data && typeof data.detail === 'object' ? data.detail : data;
}

function policyLessFailure(requestTicker, status, mode) {
  // No server policy was delivered (unreadable error body, or a parseable
  // body that carries no explicit fallback decision). The deployment mode
  // decides: hybrid/browser_only keep the browser fallback, server_pretrained
  // fails visibly.
  if (mode === 'server_pretrained') {
    throw new Error(
      `Server forecast is unavailable for ${requestTicker} (${status}); no browser fallback is allowed in this deployment.`
    );
  }
  console.warn(
    `Server forecast for ${requestTicker} (${status}) did not state a fallback policy; using browser training.`
  );
  return null;
}

/**
 * Fetches a server-pretrained forecast bundle.
 *
 * The server response is already the canonical forecasting shape shared with
 * the browser path, so it is passed through unchanged (no field remapping).
 *
 * Returns null only when a browser fallback is sanctioned: the server
 * explicitly says `fallback: "browser_training"` (200 absence, or a 503 in
 * the browser training modes), or — in a `hybrid`/`browser_only` deployment —
 * a failure that delivers no server policy at all (network failure, unreadable
 * error body). In a `server_pretrained` deployment (mode option) the same
 * policy-less failures throw, so the UI fails visibly instead of silently
 * training in the browser. A 503 that explicitly forbids a fallback
 * (`fallback: null`), a 200 invalid payload, or an identity/chronology
 * violation throws in every mode.
 *
 * @param {string} symbol
 * @param {string|number} days
 * @param {string} type UI forecast type ('price' | 'trend')
 * @param {AbortSignal} [signal]
 * @param {{ mode?: 'browser_only'|'hybrid'|'server_pretrained' }} [options]
 */
export async function fetchServerPrediction(symbol, days, type, signal, { mode = 'hybrid' } = {}) {
  const apiType = API_FORECAST_TYPES[type];
  if (!apiType) return null;

  const requestTicker = String(symbol).trim().toUpperCase();
  if (!requestTicker) return null;

  let response;
  try {
    response = await fetch(
      `${API_BASE}/api/v1/server-forecasts/${encodeURIComponent(requestTicker)}?forecast_type=${apiType}&days=${days}`,
      { signal, cache: 'no-cache' }
    );
  } catch (error) {
    if (error?.name === 'AbortError') throw error;
    console.error(`Failed to fetch server prediction for ${requestTicker}:`, error);
    // Network-level failure: no server policy is available. Only a deployment
    // that requires server-pretrained forecasts may fail visibly; hybrid and
    // browser_only deployments keep the browser fallback.
    if (mode === 'server_pretrained') {
      throw new Error(
        `Server forecast is unreachable for ${requestTicker}; no browser fallback is allowed in this deployment.`
      );
    }
    return null;
  }

  let data;
  try {
    data = await response.json();
  } catch (error) {
    console.error(`Server prediction for ${requestTicker} (${response.status}) returned an unreadable body.`, error);
    // No policy could be read from the response. The deployment mode decides:
    // hybrid/browser_only fall back, server_pretrained fails visibly.
    return policyLessFailure(requestTicker, response.status, mode);
  }

  // 200 browser fallback (missing/stale/incompatible/unconfigured in hybrid).
  if (isBrowserFallback(data)) {
    console.log(
      `Server prediction unavailable for ${requestTicker}: ${data.reason}. Falling back to browser training.`
    );
    return null;
  }

  if (!response.ok) {
    // Non-200 bodies are wrapped in `{"detail": {...}}` by the FastAPI app;
    // unwrap to recover the server's explicit fallback decision.
    const envelope = detailEnvelope(data);
    if (isBrowserFallback(envelope)) {
      console.log(
        `Server prediction unavailable for ${requestTicker}: ${envelope.reason}. Falling back to browser training.`
      );
      return null;
    }
    // An explicit `fallback: null` forbids browser training in every mode.
    if (envelope && envelope.available === false && envelope.fallback === null) {
      throw new Error(
        (envelope && typeof envelope.message === 'string' && envelope.message) ||
          `Server prediction is unavailable for ${requestTicker} (${response.status}).`
      );
    }
    // No explicit policy: the deployment mode decides.
    return policyLessFailure(requestTicker, response.status, mode);
  }

  const daysInt = parseInt(days, 10);
  if (!isValidForecast(data, daysInt, apiType, requestTicker)) {
    // A structural/identity/chronology violation is a server contract breach.
    // Never present a mismatched bundle; throw so the failure is visible.
    throw new Error(
      `Server prediction for ${requestTicker} failed validation (ticker, length, or chronology).`
    );
  }
  return data;
}