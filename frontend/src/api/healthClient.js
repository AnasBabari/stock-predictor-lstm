export function getApiBase() {
  return (
    import.meta.env.VITE_API_URL ||
    (typeof window !== 'undefined' ? window.STOCKLSTM_API_BASE : '') ||
    ''
  ).replace(/\/$/, '');
}

export async function wakeForecastService({ signal, onAttempt, maxAttempts = 15, delayMs = 2000 } = {}) {
  const base = getApiBase();
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    onAttempt?.(attempt);
    try {
      const response = await fetch(`${base}/health`, {
        signal,
        cache: 'no-cache',
      });
      if (response.ok) {
        return true;
      }
    } catch (err) {
      if (err?.name === 'AbortError') throw err;
    }
    if (attempt < maxAttempts) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  throw new Error('Service did not respond within the wake window.');
}
