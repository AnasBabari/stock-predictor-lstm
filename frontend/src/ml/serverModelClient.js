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

function isFallbackPayload(data) {
  return data && data.available === false && data.fallback === 'browser_training';
}

function isValidForecast(data, daysInt, apiType) {
  return (
    data &&
    REQUIRED_FIELDS.every((field) => data[field] !== undefined && data[field] !== null) &&
    typeof data.ticker === 'string' &&
    data.ticker.length > 0 &&
    Array.isArray(data.future_dates) &&
    data.future_dates.length === daysInt &&
    Array.isArray(data.predicted_prices) &&
    data.predicted_prices.length === daysInt &&
    data.predicted_prices.every((price) => Number.isFinite(price) && price > 0) &&
    (apiType !== 'price' ||
      (Array.isArray(data.historical_dates) &&
        data.historical_dates.length > 0 &&
        Array.isArray(data.historical_prices) &&
        data.historical_prices.length > 0 &&
        data.historical_prices.every((price) => Number.isFinite(price) && price > 0)))
  );
}

/**
 * Fetches a server-pretrained forecast bundle.
 *
 * The server response is already the canonical forecasting shape shared with
 * the browser path, so it is passed through unchanged (no field remapping).
 * Returns null when the server explicitly directs a browser fallback, the
 * payload fails shape validation, or the request errors out.
 */
export async function fetchServerPrediction(symbol, days, type, signal) {
  const apiType = API_FORECAST_TYPES[type];
  if (!apiType) return null;

  try {
    const response = await fetch(
      `${API_BASE}/api/v1/server-forecasts/${encodeURIComponent(symbol)}?forecast_type=${apiType}&days=${days}`,
      { signal, cache: 'no-cache' }
    );

    if (!response.ok) {
      return null; // Let the browser fallback take over for HTTP errors
    }

    const data = await response.json();

    // The server explicitly directed us to fall back to browser training.
    if (isFallbackPayload(data)) {
      console.log(
        `Server prediction unavailable for ${symbol}: ${data.reason}. Falling back to browser training.`
      );
      return null;
    }

    const daysInt = parseInt(days, 10);
    if (!isValidForecast(data, daysInt, apiType)) {
      console.error(`Invalid server prediction payload for ${symbol}.`);
      return null;
    }
    return data;
  } catch (error) {
    if (error?.name === 'AbortError') {
      throw error; // Let AbortError propagate
    }
    console.error(`Failed to fetch server prediction for ${symbol}:`, error);
    return null;
  }
}