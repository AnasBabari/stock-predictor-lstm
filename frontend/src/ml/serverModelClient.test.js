import { API_FORECAST_TYPES, fetchServerPrediction } from './serverModelClient';

function canonicalPayload(overrides = {}) {
  return {
    available: true,
    ticker: 'MSFT',
    forecast_days: 7,
    future_dates: ['2026-08-01', '2026-08-02', '2026-08-03', '2026-08-04', '2026-08-05', '2026-08-06', '2026-08-07'],
    predicted_prices: [101, 102, 103, 104, 105, 106, 107],
    historical_dates: ['2026-07-01', '2026-07-02', '2026-07-03'],
    historical_prices: [90, 91, 92],
    metrics: { pooled: { relative_rmse: 0.9 } },
    metadata: {
      engine: { role: 'server_pretrained', family: 'elastic_net', version_id: 'MSFT-PRICE-v1' },
      authenticity: 'sha256_only',
    },
    ...overrides,
  };
}

function jsonResponse(body, ok = true, status = 200) {
  return {
    ok,
    status,
    json: () => Promise.resolve(body),
  };
}

describe('serverModelClient forecast contract', () => {
  let fetchMock;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  test('passes the canonical payload through unchanged', async () => {
    const payload = canonicalPayload();
    fetchMock.mockResolvedValue(jsonResponse(payload));
    const result = await fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal);
    expect(result).toEqual(payload);
  });

  test('maps a UI trend request to the API direction value', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ available: false, reason: 'unsupported_forecast_type', fallback: 'browser_training' })
    );
    await fetchServerPrediction('MSFT', 7, 'trend', new AbortController().signal);
    const url = fetchMock.mock.calls[0][0];
    expect(url).toContain('forecast_type=direction');
    expect(API_FORECAST_TYPES).toEqual({ price: 'price', trend: 'direction' });
  });

  test('unsupported direction fallback returns null (browser trend path)', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ available: false, reason: 'unsupported_forecast_type', fallback: 'browser_training' })
    );
    expect(await fetchServerPrediction('MSFT', 7, 'trend', new AbortController().signal)).toBeNull();
  });

test('missing chart arrays for a price forecast -> null', async () => {
    fetchMock.mockResolvedValue(jsonResponse(canonicalPayload({ historical_prices: [] })));
    expect(
      await fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal)
    ).toBeNull();
  });

  test('missing required field or wrong forecast length -> null', async () => {
    fetchMock.mockResolvedValue(jsonResponse(canonicalPayload({ ticker: null })));
    expect(
      await fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal)
    ).toBeNull();
    fetchMock.mockResolvedValue(jsonResponse(canonicalPayload({ predicted_prices: [100] })));
    expect(
      await fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal)
    ).toBeNull();
  });

  test('non-positive or non-finite predicted prices -> null', async () => {
    const payload = canonicalPayload({ predicted_prices: [200, NaN, 0] });
    fetchMock.mockResolvedValue(jsonResponse(payload));
    expect(await fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal)).toBeNull();
  });

  test('HTTP error or empty shape falls back to browser training (null)', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: 'nope' }, false, 503));
    expect(await fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal)).toBeNull();
    fetchMock.mockResolvedValue(jsonResponse({}, false, 500));
    expect(await fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal)).toBeNull();
  });

  test('AbortError propagates', async () => {
    const abortError = new Error('aborted');
    abortError.name = 'AbortError';
    fetchMock.mockRejectedValue(abortError);
    await expect(
      fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal)
    ).rejects.toThrow('aborted');
  });
});