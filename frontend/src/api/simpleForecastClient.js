function getApiBase() {
  return (import.meta.env.VITE_API_URL || (typeof window !== 'undefined' ? window.STOCKLSTM_API_BASE : '') || '').replace(/\/$/, '');
}

const API_BASE = getApiBase();

async function getJson(path, { signal, timeoutMs = 120_000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const abort = () => controller.abort();
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) controller.abort();
  try {
    const base = getApiBase();
    const response = await fetch(`${base}${path}`, { signal: controller.signal });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(payload?.detail || payload?.message || `Request failed (${response.status}).`);
      error.status = response.status;
      throw error;
    }
    return payload;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}

export async function wakeForecastService({ signal, onAttempt } = {}) {
  let lastError;
  for (let attempt = 1; attempt <= 12; attempt += 1) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    onAttempt?.(attempt);
    try {
      const health = await getJson('/health', { signal, timeoutMs: 12_000 });
      if (health?.status === 'ok') return health;
    } catch (error) {
      if (error?.name === 'AbortError' && signal?.aborted) throw error;
      lastError = error;
    }
    await new Promise((resolve, reject) => {
      const delay = setTimeout(resolve, Math.min(3_000 + attempt * 750, 10_000));
      signal?.addEventListener('abort', () => {
        clearTimeout(delay);
        reject(new DOMException('Aborted', 'AbortError'));
      }, { once: true });
    });
  }
  throw lastError || new Error('The forecast service did not start in time.');
}

/** Request a learned forecast; unavailable service responses stay unavailable. */
export async function fetchSimpleForecast(ticker, { signal } = {}) {
  const symbol = String(ticker || '').trim().toUpperCase();
  if (!symbol) throw new Error('A stock ticker is required.');
  // Only the learned endpoint can supply a price estimate. A 404 is surfaced
  // to the UI, which retains history and offers a retry.
  return getJson(`/api/v1/forecast?ticker=${encodeURIComponent(symbol)}&days=7`, { signal });
}

export async function fetchTickerNews(ticker, { signal } = {}) {
  const symbol = String(ticker || 'MSFT').trim().toUpperCase();
  try {
    const res = await getJson(`/api/v1/news?ticker=${encodeURIComponent(symbol)}`, {
      signal,
      timeoutMs: 15_000,
    });
    if (Array.isArray(res?.items)) {
      return res;
    }
  } catch (error) {
    if (signal?.aborted) throw error;
  }
  return {
    status: 'unavailable',
    ticker: symbol,
    items: [],
    role: 'context_only',
    used_by_model: false,
    provider: null,
    as_of: null,
  };
}

export { API_BASE };
