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

/**
 * Fetches a server-pretrained forecast bundle.
 *
 * The server response is already the canonical forecasting shape shared with
 * the browser path, so it is passed through unchanged (no field remapping).
 *
 * Returns null only when the server explicitly directs a browser fallback (a
 * 200 ``fallback: "browser_training"``, or a 503 that carries it in the
 * browser training modes) or for a shape failure (invalid structural payload
 * or network rejection). When the server forbids a fallback — a 503 in
 * ``server_pretrained`` mode, integrity failures, an unreadable error body, or
 * an identity mismatch — this throws so the UI surfaces the failure instead of
 * silently training in the browser.
 */
export async function fetchServerPrediction(symbol, days, type, signal) {
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
    return null; // Network-level failure; no server policy is available.
  }

  let data;
  try {
    data = await response.json();
  } catch {
    if (!response.ok) {
      throw new Error(
        `Server prediction failed for ${requestTicker} (${response.status}) — response could not be read.`
      );
    }
    console.error(`Invalid server prediction payload for ${requestTicker}.`);
    return null;
  }

  // 200 browser fallback (missing/stale/incompatible/unconfigured in hybrid).
  if (isBrowserFallback(data)) {
    console.log(
      `Server prediction unavailable for ${requestTicker}: ${data.reason}. Falling back to browser training.`
    );
    return null;
  }

  // A non-200 that is not an explicit browser fallback throws (fail closed).
  if (!response.ok) {
    const envelope = detailEnvelope(data);
    if (isBrowserFallback(envelope)) {
      console.log(
        `Server prediction unavailable for ${requestTicker}: ${envelope.message || envelope.code}. Falling back to browser training.`
      );
      return null;
    }
    const message =
      (envelope && typeof envelope.message === 'string' && envelope.message) ||
      `Server prediction is unavailable for ${requestTicker} (${response.status}).`;
    throw new Error(message);
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