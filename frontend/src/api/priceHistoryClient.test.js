import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { clearPriceHistoryCache, fetchPriceHistory } from './priceHistoryClient';

const payload = {
  ticker: 'MSFT',
  as_of: '2024-06-01',
  provider: 'alpaca',
  market_data_cache: 'miss',
  first_date: '2024-01-02',
  daily: [{ d: '2024-01-02', c: 100 }, { d: '2024-01-03', c: 101 }],
  intraday: null,
  intraday_session: null,
};

describe('priceHistoryClient', () => {
  beforeEach(() => {
    clearPriceHistoryCache();
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve(payload),
    })));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('attaches fetch timing metadata and serves repeats from cache', async () => {
    const first = await fetchPriceHistory('msft');
    expect(first.ticker).toBe('MSFT');
    expect(first.meta.fromCache).toBe(false);
    expect(first.meta.fetchMs).toBeGreaterThanOrEqual(0);
    expect(first.marketDataCache).toBe('miss');
    const second = await fetchPriceHistory('MSFT');
    expect(second.meta).toEqual({ fetchMs: 0, fromCache: true });
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('throws when the response carries no daily history', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ ...payload, daily: [] }),
    })));
    await expect(fetchPriceHistory('MSFT')).rejects.toThrow('No price history');
  });

  it('throws on HTTP errors', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: false,
      status: 503,
      json: () => Promise.resolve({ detail: 'down' }),
    })));
    await expect(fetchPriceHistory('MSFT')).rejects.toThrow('down');
  });
});
