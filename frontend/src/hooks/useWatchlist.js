import { useCallback, useEffect, useState } from 'react';

const WL_KEY = 'stock_lstm_watchlist';
const HIST_KEY = 'stock_lstm_history';

/**
 * History entry contract (schema 2):
 *
 * {
 *   ticker: string            - uppercase symbol (required)
 *   createdAt: string | null  - ISO timestamp of the prediction
 *   horizon: number | null    - forecast days requested
 *   forecastType: string|null - 'price' | 'trend'
 *   direction: string|null    - 'Up' | 'Down' | 'Neutral' (trend entries)
 *   lastClose: number | null  - latest historical close (price forecasts)
 *   predictedValue: number|null - final-day predicted price, or mean up-probability (trend)
 *   changePercent: number|null  - predicted % change vs lastClose (price only)
 *   snapshotId: string | null
 *   modelRole: string | null    - e.g. 'browser_learned' | 'baseline_fallback'
 * }
 *
 * Schema 1 stored bare ticker strings; normalizeLegacyEntry migrates them.
 */
export const HISTORY_SCHEMA_VERSION = 2;
export const TICKER_PATTERN = /^[A-Z0-9.\-]{1,12}$/;

export function isValidTicker(value) {
  return typeof value === 'string' && TICKER_PATTERN.test(value.trim().toUpperCase());
}

function toFiniteNumberOrNull(value) {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function normalizeHistoryEntry(raw) {
  if (typeof raw === 'string') {
    const ticker = raw.trim().toUpperCase();
    if (!isValidTicker(ticker)) return null;
    return {
      ticker,
      createdAt: null,
      horizon: null,
      forecastType: null,
      direction: null,
      lastClose: null,
      predictedValue: null,
      changePercent: null,
      snapshotId: null,
      modelRole: null,
    };
  }
  if (!raw || typeof raw !== 'object') return null;
  const ticker = typeof raw.ticker === 'string' ? raw.ticker.trim().toUpperCase() : '';
  if (!isValidTicker(ticker)) return null;
  return {
    ticker,
    createdAt: typeof raw.createdAt === 'string' ? raw.createdAt : null,
    horizon: toFiniteNumberOrNull(raw.horizon),
    forecastType: raw.forecastType === 'trend' ? 'trend' : raw.forecastType === 'price' ? 'price' : null,
    direction: typeof raw.direction === 'string' && ['Up', 'Down', 'Neutral'].includes(raw.direction)
      ? raw.direction
      : null,
    lastClose: toFiniteNumberOrNull(raw.lastClose),
    predictedValue: toFiniteNumberOrNull(raw.predictedValue),
    changePercent: toFiniteNumberOrNull(raw.changePercent),
    snapshotId: typeof raw.snapshotId === 'string' ? raw.snapshotId : null,
    modelRole: typeof raw.modelRole === 'string' ? raw.modelRole : null,
  };
}

export function historyEntryFromPrediction(data) {
  const base = normalizeHistoryEntry(data?.ticker);
  if (!base) return null;
  const isTrend = data?.direction != null;
  const lastClose = toFiniteNumberOrNull(
    data?.historical_prices?.[data.historical_prices.length - 1]
  );
  let predictedValue = null;
  let changePercent = null;
  if (isTrend) {
    // v2: one three-way decision; confidence = probability of the selected
    // class (Down/Neutral/Up → down/neutral/up).
    const probs = data.direction_probabilities || {};
    const key = String(data.direction || '').toLowerCase();
    predictedValue = toFiniteNumberOrNull(probs[key]);
  } else {
    const prices = Array.isArray(data.predicted_prices) ? data.predicted_prices : [];
    predictedValue = toFiniteNumberOrNull(prices[prices.length - 1]);
    if (predictedValue !== null && lastClose !== null && lastClose > 0) {
      changePercent = ((predictedValue / lastClose - 1) * 100);
    }
  }
  return {
    ...base,
    createdAt: new Date().toISOString(),
    horizon: toFiniteNumberOrNull(data?.forecast_days),
    forecastType: isTrend ? 'trend' : 'price',
    direction: isTrend ? String(data.direction) : null,
    lastClose,
    predictedValue,
    changePercent,
    snapshotId: typeof data?.metadata?.snapshot_id === 'string' ? data.metadata.snapshot_id : null,
    modelRole: typeof data?.metadata?.engine?.role === 'string' ? data.metadata.engine.role : null,
  };
}

function loadHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HIST_KEY));
    if (Array.isArray(parsed)) {
      // Legacy schema 1: bare strings (or already-migrated objects).
      return parsed.map(normalizeHistoryEntry).filter(Boolean);
    }
    if (parsed && parsed.schema === HISTORY_SCHEMA_VERSION && Array.isArray(parsed.entries)) {
      return parsed.entries.map(normalizeHistoryEntry).filter(Boolean);
    }
    return [];
  } catch {
    return [];
  }
}

export function useWatchlist({ addToast, forecastType, stockInfo }) {
  const [watchlist, setWatchlist] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(WL_KEY)) || [];
    } catch {
      return [];
    }
  });

  const [history, setHistory] = useState(loadHistory);

  useEffect(() => {
    try {
      localStorage.setItem(WL_KEY, JSON.stringify(watchlist));
    } catch {
      // Ignore storage errors
    }
  }, [watchlist]);

  useEffect(() => {
    try {
      localStorage.setItem(
        HIST_KEY,
        JSON.stringify({ schema: HISTORY_SCHEMA_VERSION, entries: history })
      );
    } catch {
      // Ignore storage errors
    }
  }, [history]);

  const addToHistory = useCallback((predictionResult) => {
    const entry = historyEntryFromPrediction(predictionResult);
    if (!entry) return;
    setHistory((prev) => [
      entry,
      ...prev.filter((item) => item.ticker !== entry.ticker),
    ].slice(0, 10));
  }, []);

  const handleAddWatchlist = useCallback(
    (data) => {
      if (!data || forecastType !== 'price' || !data.historical_prices?.length) {
        addToast('info', 'Watchlist requires a price forecast result');
        return;
      }
      if (watchlist.some((w) => w.ticker === data.ticker)) {
        addToast('info', `${data.ticker} is already in your watchlist`);
        return;
      }

      const safeName = stockInfo && stockInfo.ticker === data.ticker ? stockInfo.name : '';
      const newWatchItem = {
        ticker: data.ticker,
        name: safeName,
        lastPrice: data.historical_prices[data.historical_prices.length - 1],
      };

      setWatchlist((prev) => [newWatchItem, ...prev]);
      addToast('success', `${data.ticker} added to watchlist`);
    },
    [watchlist, stockInfo, addToast, forecastType]
  );

  const handleRemoveWatchlist = useCallback(
    (index) => {
      setWatchlist((prev) => {
        const updated = [...prev];
        updated.splice(index, 1);
        return updated;
      });
      addToast('info', 'Removed from watchlist');
    },
    [addToast]
  );

  const handleClearWatchlist = useCallback(() => {
    setWatchlist([]);
    addToast('info', 'Watchlist cleared');
  }, [addToast]);

  const handleClearHistory = useCallback(() => {
    setHistory([]);
    addToast('info', 'History cleared');
  }, [addToast]);

  return {
    watchlist,
    history,
    addToHistory,
    handleAddWatchlist,
    handleRemoveWatchlist,
    handleClearWatchlist,
    handleClearHistory,
  };
}
