import { beforeEach, describe, expect, it, vi } from 'vitest';
import { clearForecastLedgerCache, fetchForecastLedger } from './forecastLedgerClient';

describe('forecast ledger client', () => {
  beforeEach(() => {
    clearForecastLedgerCache();
    vi.restoreAllMocks();
    window.STOCKLSTM_API_BASE = 'https://api.example.test/';
  });

  it('routes through the configured API base and keys cache by ticker and horizon', async () => {
    const fetchMock = vi.fn((url) => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ entries: [{ url }] }),
    }));
    vi.stubGlobal('fetch', fetchMock);

    const five = await fetchForecastLedger('MSFT', 5);
    const ten = await fetchForecastLedger('MSFT', 10);
    await fetchForecastLedger('MSFT', 5);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toBe('https://api.example.test/api/v1/volatility/ledger?ticker=MSFT&horizon=5');
    expect(fetchMock.mock.calls[1][0]).toBe('https://api.example.test/api/v1/volatility/ledger?ticker=MSFT&horizon=10');
    expect(five.entries[0].url).toContain('horizon=5');
    expect(ten.entries[0].url).toContain('horizon=10');
  });

  it('does not cache an aborted response', async () => {
    const controller = new AbortController();
    const fetchMock = vi.fn(() => new Promise((resolve) => {
      controller.signal.addEventListener('abort', () => resolve({
        ok: true,
        json: () => Promise.resolve({ entries: [] }),
      }), { once: true });
    }));
    vi.stubGlobal('fetch', fetchMock);

    const request = fetchForecastLedger('AAPL', 5, controller.signal);
    controller.abort();
    await expect(request).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
