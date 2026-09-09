import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchVolatilityForecast } from '../ml/volatilityClient';

export const OUTLOOK_HORIZONS = [5, 10, 20];
const outlookCache = new Map();

export function clearVolatilityOutlookCache() {
  outlookCache.clear();
}

/**
 * Enhanced-volatility outlook for one ticker (model=gpu_g3, horizons
 * 5/10/20 fetched in parallel). Cached per ticker for the page lifetime.
 * A failed horizon resolves to null so one bad response never blocks the
 * other two; the card renders whatever arrived.
 */
export function useVolatilityOutlook(ticker) {
  const [outlook, setOutlook] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [retryTick, setRetryTick] = useState(0);
  const seqRef = useRef(0);

  useEffect(() => {
    const symbol = String(ticker || '').trim().toUpperCase();
    if (!symbol) {
      setOutlook(null);
      setLoading(false);
      setError('');
      return undefined;
    }
    const id = seqRef.current + 1;
    seqRef.current = id;
    const cached = outlookCache.get(symbol);
    if (cached) {
      setOutlook(cached);
      setLoading(false);
      setError('');
      return undefined;
    }
    const controller = new AbortController();
    setOutlook(null);
    setLoading(true);
    setError('');
    Promise.all(OUTLOOK_HORIZONS.map(async (horizon) => {
      try {
        const result = await fetchVolatilityForecast(symbol, horizon, controller.signal, { model: 'gpu_g3' });
        return [horizon, result];
      } catch (err) {
        if (err?.name === 'AbortError') throw err;
        return [horizon, null];
      }
    }))
      .then((entries) => {
        if (seqRef.current !== id) return;
        const byHorizon = Object.fromEntries(entries);
        const loaded = OUTLOOK_HORIZONS.filter((horizon) => byHorizon[horizon]);
        if (loaded.length === 0) {
          setOutlook(null);
          setLoading(false);
          setError('Volatility outlook is unavailable right now.');
          return;
        }
        const value = { ticker: symbol, byHorizon };
        outlookCache.set(symbol, value);
        setOutlook(value);
        setLoading(false);
      })
      .catch((err) => {
        if (seqRef.current === id && err?.name !== 'AbortError') {
          setOutlook(null);
          setLoading(false);
          setError(err?.message || 'Volatility outlook is unavailable right now.');
        }
      });
    return () => controller.abort();
  }, [ticker, retryTick]);

  const retry = useCallback(() => {
    setRetryTick((tick) => tick + 1);
  }, []);

  return { outlook, loading, error, retry };
}
