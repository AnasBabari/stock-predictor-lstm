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
      authenticity: 'ed25519_verified',
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

function rejectsWith(promise, message) {
  return expect(promise).rejects.toThrow(message);
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

  test('missing chart arrays for a price forecast -> rejects', async () => {
    fetchMock.mockResolvedValue(jsonResponse(canonicalPayload({ historical_prices: [] })));
    await rejectsWith(
      fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal),
      'failed validation'
    );
  });

  test('wrong ticker in payload -> rejects (identity breach)', async () => {
    fetchMock.mockResolvedValue(jsonResponse(canonicalPayload({ ticker: 'GOOG' })));
    await rejectsWith(
      fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal),
      'failed validation'
    );
  });

  test('wrong forecast length -> rejects', async () => {
    fetchMock.mockResolvedValue(jsonResponse(canonicalPayload({ predicted_prices: [100] })));
    await rejectsWith(
      fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal),
      'failed validation'
    );
  });

  test('non-chronological dates -> rejects', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(canonicalPayload({ future_dates: ['2026-08-03', '2026-08-02', '2026-08-01', '2026-08-04', '2026-08-05', '2026-08-06', '2026-08-07'] }))
    );
    await rejectsWith(
      fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal),
      'failed validation'
    );
  });

  test('non-positive or non-finite predicted prices -> rejects', async () => {
    const payload = canonicalPayload({ predicted_prices: [200, NaN, 0, 101, 102, 103, 104] });
    fetchMock.mockResolvedValue(jsonResponse(payload));
    await rejectsWith(
      fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal),
      'failed validation'
    );
  });

  test('503 with fallback browser_training (hybrid) -> null', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(
        { detail: { available: false, code: 'registry_unavailable', message: 'Infra down', fallback: 'browser_training' } },
        false,
        503
      )
    );
    expect(await fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal)).toBeNull();
  });

  test('503 with fallback null (server_pretrained) -> throws, never silent fallback', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(
        { detail: { available: false, code: 'signature_verification_failed', message: 'Verification failed.', fallback: null } },
        false,
        503
      )
    );
    await rejectsWith(
      fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal),
      'Verification failed'
    );
  });

  test('unreadable 503 error body -> null in hybrid mode (no policy delivered)', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 503,
      json: () => Promise.reject(new Error('not json')),
    });
    expect(await fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal)).toBeNull();
  });

  test('unreadable 503 error body in server_pretrained mode -> throws', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 503,
      json: () => Promise.reject(new Error('not json')),
    });
    await rejectsWith(
      fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal, {
        mode: 'server_pretrained',
      }),
      'no browser fallback is allowed'
    );
  });

  test('network failure -> null in hybrid mode (no server policy available)', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));
    expect(await fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal)).toBeNull();
  });

  test('network failure in server_pretrained mode -> throws, never silent fallback', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));
    await rejectsWith(
      fetchServerPrediction('MSFT', 7, 'price', new AbortController().signal, {
        mode: 'server_pretrained',
      }),
      'no browser fallback is allowed'
    );
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