import { FEATURE_NAMES, FEATURE_SCHEMA_VERSION } from './preprocessing';

function snapshot() {
  const rows = 100;
  const features = Array.from({ length: rows }, (_, row) => FEATURE_NAMES.map((_, column) => row + column + 1));
  const isoDay = (year, index) => new Date(Date.UTC(year, 0, 1 + index)).toISOString().slice(0, 10);
  return {
    ticker: 'TEST', schema_version: FEATURE_SCHEMA_VERSION, snapshot_id: 'snapshot-client', feature_names: FEATURE_NAMES,
    window_size: 60, output_width: 30,
    dates: Array.from({ length: rows }, (_, index) => isoDay(2025, index)),
    features, historical_prices: features.map((row) => row[0]),
    future_dates: Array.from({ length: 30 }, (_, index) => isoDay(2026, index)),
  };
}

describe('browser training request coordination', () => {
  let instances;

  beforeEach(() => {
    vi.resetModules();
    instances = [];
    class FakeWorker {
      constructor() { this.messages = []; instances.push(this); }
      postMessage(message) { this.messages.push(message); }
      terminate() {}
    }
    vi.stubGlobal('Worker', FakeWorker);
  });

  afterEach(() => vi.unstubAllGlobals());

  test('coalesces identical ticker, type, horizon, snapshot, and profile requests', async () => {
    const { trainBrowserForecast } = await import('./browserTrainingClient');
    const first = trainBrowserForecast({ snapshot: snapshot(), forecastType: 'price', days: 7, profile: 'balanced' });
    const second = trainBrowserForecast({ snapshot: snapshot(), forecastType: 'price', days: 7, profile: 'balanced' });
    expect(instances[0].messages.filter((message) => message.type === 'forecast')).toHaveLength(1);
    const id = instances[0].messages[0].id;
    instances[0].onmessage({ data: { id, type: 'complete', result: { ok: true } } });
    await expect(first).resolves.toEqual({ ok: true });
    await expect(second).resolves.toEqual({ ok: true });
  });

  test('rejects all subscribers when the worker fails', async () => {
    const { trainBrowserForecast } = await import('./browserTrainingClient');
    const pending = trainBrowserForecast({ snapshot: snapshot(), forecastType: 'price', days: 7, profile: 'quick' });
    instances[0].onerror(new Error('worker crashed'));
    await expect(pending).rejects.toThrow('Browser training worker failed.');
  });

  test('does not coalesce different forecast horizons', async () => {
    const { trainBrowserForecast } = await import('./browserTrainingClient');
    trainBrowserForecast({ snapshot: snapshot(), forecastType: 'price', days: 3, profile: 'quick' });
    trainBrowserForecast({ snapshot: snapshot(), forecastType: 'price', days: 7, profile: 'quick' });
    expect(instances[0].messages.filter((message) => message.type === 'forecast')).toHaveLength(2);
  });
});
