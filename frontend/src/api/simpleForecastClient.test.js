import { afterEach, expect, it, vi } from 'vitest';
import { fetchSimpleForecast, fetchTickerNews } from './simpleForecastClient';

afterEach(() => vi.unstubAllGlobals());

it('surfaces a missing learned endpoint without inventing a price path or backtest', async () => {
  const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 404, json: async () => ({ detail: 'Not found' }) });
  vi.stubGlobal('fetch', fetchMock);
  await expect(fetchSimpleForecast('MSFT')).rejects.toMatchObject({ status: 404 });
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

it('preserves an empty news response instead of substituting example headlines', async () => {
  const body = { ticker: 'MSFT', status: 'available', items: [] };
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => body }));
  expect(await fetchTickerNews('MSFT')).toEqual(body);
});

it('reports news failure as unavailable with no fabricated feed or timestamp', async () => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Offline')));
  expect(await fetchTickerNews('MSFT')).toMatchObject({ status: 'unavailable', items: [], provider: null, as_of: null });
});
