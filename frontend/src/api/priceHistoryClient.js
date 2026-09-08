const cache = new Map();

function getApiBase() {
  return (import.meta.env.VITE_API_URL || (typeof window !== 'undefined' ? window.STOCKLSTM_API_BASE : '') || '').replace(/\/$/, '');
}

export function clearPriceHistoryCache() {
  cache.clear();
}

/**
 * Fetch full chart history for one ticker in a single round trip.
 *
 * The backend returns downsampled daily closes plus last-session intraday
 * bars; every range tab and zoom level is derived client-side, so switching
 * ranges never hits the network. Responses are cached per ticker for the
 * lifetime of the page. A side effect worth keeping: on a cold backend this
 * request warms the shared market-data cache, so the forecast that usually
 * follows it resolves fast.
 */
export async function fetchPriceHistory(ticker, { signal, timeoutMs = 20_000 } = {}) {
  const symbol = String(ticker || '').trim().toUpperCase();
  if (!symbol) throw new Error('A ticker symbol is required for price history.');
  const cached = cache.get(symbol);
  if (cached) return { ...cached, meta: { fetchMs: 0, fromCache: true } };
  const startedAt = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  const abort = () => controller.abort();
  signal?.addEventListener('abort', abort, { once: true });
  try {
    const base = getApiBase();
    const response = await fetch(`${base}/api/v1/history?ticker=${encodeURIComponent(symbol)}`, {
      signal: controller.signal,
      cache: 'no-cache',
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload?.detail || `Price history failed (${response.status}).`);
    }
    const daily = Array.isArray(payload?.daily) ? payload.daily : [];
    if (daily.length === 0) throw new Error('No price history is available for this ticker.');
    const result = {
      ticker: payload.ticker || symbol,
      asOf: payload.as_of || null,
      provider: payload.provider || null,
      marketDataCache: payload.market_data_cache || null,
      firstDate: payload.first_date || daily[0]?.d || null,
      daily,
      intraday: Array.isArray(payload?.intraday) ? payload.intraday : null,
      intradaySession: payload.intraday_session || null,
    };
    const finishedAt = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const withMeta = { ...result, meta: { fetchMs: Math.round(finishedAt - startedAt), fromCache: false } };
    cache.set(symbol, result);
    return withMeta;
  } finally {
    window.clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}
