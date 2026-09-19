const cache = new Map();
export const clearForecastLedgerCache = () => cache.clear();

export async function fetchForecastLedger(ticker, horizon, signal) {
  const runtimeBase = typeof window !== 'undefined' ? window.STOCKLSTM_API_BASE : '';
  const base = (import.meta.env.VITE_API_URL || runtimeBase || '').replace(/\/$/, '');
  const params = new URLSearchParams({ ticker });
  if (horizon != null) params.set('horizon', String(horizon));
  const url = `${base}/api/v1/volatility/ledger?${params}`;
  const saved = cache.get(url);
  if (saved && Date.now() - saved.at < 60_000) return saved.data;
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Failed to load forecast ledger (${response.status})`);
  const data = await response.json();
  if (!Array.isArray(data?.entries)) throw new Error('Invalid forecast ledger response');
  if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
  cache.set(url, { at: Date.now(), data });
  return data;
}
