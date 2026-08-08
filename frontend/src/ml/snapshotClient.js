// In-memory training-data snapshot cache with request coalescing.
//
// The `/api/v1/training-data` payload is large (up to 2,000 rows x 28
// features), static for a market-session, and shared by every forecast type
// and profile for a ticker. Repeated Predict clicks, forecast re-runs, and
// the Complete Analysis flow must never issue duplicate network requests:
// identical in-flight requests share one fetch, satisfied requests are served
// from memory for a short TTL, and failures are never cached.

export function createSnapshotClient({
  baseUrl = '',
  ttlMs = 5 * 60 * 1000,
  fetchImpl = (...args) => globalThis.fetch(...args),
} = {}) {
  const cached = new Map();
  const inFlight = new Map();

  function fetchTrainingSnapshot(ticker, signal) {
    const key = String(ticker).trim().toUpperCase();
    if (!key) return Promise.reject(new Error('Ticker is missing.'));

    const hit = cached.get(key);
    if (hit && Date.now() - hit.fetchedAt < ttlMs) return Promise.resolve(hit.snapshot);

    const pending = inFlight.get(key);
    if (pending) return pending;

    const promise = (async () => {
      const endpoint = `${baseUrl}/api/v1/training-data?ticker=${encodeURIComponent(key)}`;
      const response = await fetchImpl(endpoint, { signal });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Training data failed (${response.status})`);
      }
      const snapshot = await response.json();
      cached.set(key, { snapshot, fetchedAt: Date.now() });
      return snapshot;
    })();
    inFlight.set(key, promise);
    return promise.finally(() => {
      inFlight.delete(key);
    });
  }

  function clear() {
    cached.clear();
    for (const pending of inFlight.values()) {
      pending.catch(() => undefined);
    }
    inFlight.clear();
  }

  function activeCount() {
    return inFlight.size;
  }

  return {
    fetchTrainingSnapshot,
    clear,
    activeCount,
  };
}