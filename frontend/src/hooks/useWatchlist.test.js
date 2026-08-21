import { describe, expect, it } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import {
  HISTORY_SCHEMA_VERSION,
  historyEntryFromPrediction,
  isValidTicker,
  normalizeHistoryEntry,
  useWatchlist,
} from './useWatchlist';

const pricePayload = {
  ticker: 'msft',
  forecast_days: 7,
  historical_prices: [90, 95, 100],
  predicted_prices: [105, 110],
  metadata: { snapshot_id: 'snap-1', engine: { role: 'browser_learned' } },
};

const trendPayload = {
  ticker: 'AAPL',
  forecast_days: 3,
  directions: [1, 0],
  probabilities: [0.6, 0.8],
  metadata: { snapshot_id: 'snap-2', engine: { role: 'baseline_fallback' } },
};

describe('historyEntryFromPrediction', () => {
  it('builds a full entry from a price prediction payload', () => {
    const entry = historyEntryFromPrediction(pricePayload);
    expect(entry.ticker).toBe('MSFT');
    expect(entry.forecastType).toBe('price');
    expect(entry.horizon).toBe(7);
    expect(entry.lastClose).toBe(100);
    expect(entry.predictedValue).toBe(110);
    expect(entry.changePercent).toBeCloseTo(10, 5);
    expect(entry.snapshotId).toBe('snap-1');
    expect(entry.modelRole).toBe('browser_learned');
    expect(typeof entry.createdAt).toBe('string');
  });

  it('builds a trend entry with mean up-probability and no change percent', () => {
    const entry = historyEntryFromPrediction(trendPayload);
    expect(entry.forecastType).toBe('trend');
    expect(entry.predictedValue).toBeCloseTo(0.7, 5);
    expect(entry.changePercent).toBeNull();
    expect(entry.lastClose).toBeNull();
  });

  it('returns null for payloads without a usable ticker', () => {
    expect(historyEntryFromPrediction(null)).toBeNull();
    expect(historyEntryFromPrediction({ ticker: '' })).toBeNull();
    expect(historyEntryFromPrediction({ ticker: '../etc/passwd' })).toBeNull();
  });
});

describe('normalizeHistoryEntry', () => {
  it('migrates legacy schema-1 string entries', () => {
    const entry = normalizeHistoryEntry(' aapl ');
    expect(entry).toMatchObject({ ticker: 'AAPL', createdAt: null, horizon: null });
  });

  it('rejects malformed entries instead of crashing', () => {
    expect(normalizeHistoryEntry(42)).toBeNull();
    expect(normalizeHistoryEntry({})).toBeNull();
    expect(normalizeHistoryEntry({ ticker: 123 })).toBeNull();
  });

  it('coerces non-finite numerics to null', () => {
    const entry = normalizeHistoryEntry({
      ticker: 'T',
      lastClose: Number.NaN,
      horizon: '5',
      changePercent: Infinity,
    });
    expect(entry.lastClose).toBeNull();
    expect(entry.horizon).toBe(5);
    expect(entry.changePercent).toBeNull();
  });
});

describe('isValidTicker', () => {
  it.each([
    ['MSFT', true],
    ['brk.b', true],
    ['', false],
    [null, false],
    [undefined, false],
    ['TOOLONGTICKERX', false],
    ['A B', false],
  ])('%s -> %s', (input, expected) => {
    expect(isValidTicker(input)).toBe(expected);
  });
});

const flush = () =>
  act(async () => {
    await Promise.resolve();
  });

describe('useWatchlist history persistence', () => {
  it('persists versioned schema and round-trips entries', async () => {
    localStorage.clear();
    const { result } = renderHook(() =>
      useWatchlist({ addToast: () => {}, forecastType: 'price', stockInfo: null })
    );
    act(() => result.current.addToHistory(pricePayload));
    await flush();

    const stored = JSON.parse(localStorage.getItem('stock_lstm_history'));
    expect(stored.schema).toBe(HISTORY_SCHEMA_VERSION);
    expect(stored.entries[0].ticker).toBe('MSFT');

    // A fresh hook instance must load the persisted entry.
    const reloaded = renderHook(() =>
      useWatchlist({ addToast: () => {}, forecastType: 'price', stockInfo: null })
    );
    expect(reloaded.result.current.history[0].ticker).toBe('MSFT');
    localStorage.clear();
  });

  it('migrates legacy string arrays on load', () => {
    localStorage.setItem('stock_lstm_history', JSON.stringify(['AAPL', 'msft']));
    const { result } = renderHook(() =>
      useWatchlist({ addToast: () => {}, forecastType: 'price', stockInfo: null })
    );
    expect(result.current.history.map((h) => h.ticker)).toEqual(['AAPL', 'MSFT']);
    localStorage.clear();
  });

  it('dedupes by ticker keeping the most recent entry', async () => {
    localStorage.clear();
    const { result } = renderHook(() =>
      useWatchlist({ addToast: () => {}, forecastType: 'price', stockInfo: null })
    );
    act(() => result.current.addToHistory(pricePayload));
    act(() => result.current.addToHistory(trendPayload));
    act(() => result.current.addToHistory({ ...pricePayload, predicted_prices: [120] }));
    await flush();
    const tickers = result.current.history.map((h) => h.ticker);
    expect(tickers.filter((t) => t === 'MSFT')).toHaveLength(1);
    expect(result.current.history[0].ticker).toBe('MSFT');
    expect(result.current.history[0].predictedValue).toBe(120);
    localStorage.clear();
  });

  it('ignores invalid prediction payloads safely', async () => {
    localStorage.clear();
    const { result } = renderHook(() =>
      useWatchlist({ addToast: () => {}, forecastType: 'price', stockInfo: null })
    );
    act(() => {
      result.current.addToHistory(undefined);
      result.current.addToHistory('PLAIN STRING');
    });
    await flush();
    expect(result.current.history).toHaveLength(0);
    localStorage.clear();
  });
});
