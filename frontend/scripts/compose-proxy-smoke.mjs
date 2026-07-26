const baseUrl = process.env.COMPOSE_FRONTEND_URL || 'http://127.0.0.1:5500';
const controller = new AbortController();
const timeout = setTimeout(() => controller.abort(), 15_000);

try {
  const response = await fetch(`${baseUrl}/api/v1/search?query=AAPL`, {
    headers: { accept: 'application/json' },
    signal: controller.signal,
  });
  if (!response.ok) {
    throw new Error(`expected HTTP 200 from the proxied search endpoint, got ${response.status}`);
  }
  const payload = await response.json();
  if (!payload || !Array.isArray(payload.results)) {
    throw new Error('proxied search response was not the expected JSON object with a results array');
  }
  console.log(`Compose proxy smoke test passed (${payload.results.length} search results).`);
} finally {
  clearTimeout(timeout);
}
