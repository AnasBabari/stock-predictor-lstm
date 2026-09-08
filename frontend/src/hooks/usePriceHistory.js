import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchPriceHistory } from '../api/priceHistoryClient';

/**
 * Load full chart history for one ticker. Responses are cached per ticker
 * inside the client, so remounts and tab switches never refetch.
 */
export function usePriceHistory(ticker) {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [meta, setMeta] = useState(null);
  const [retryTick, setRetryTick] = useState(0);
  const seqRef = useRef(0);

  useEffect(() => {
    const symbol = String(ticker || '').trim().toUpperCase();
    if (!symbol) {
      setHistory(null);
      setLoading(false);
      setError('');
      setMeta(null);
      return undefined;
    }
    const id = seqRef.current + 1;
    seqRef.current = id;
    const controller = new AbortController();
    setHistory(null);
    setLoading(true);
    setError('');
    setMeta(null);
    fetchPriceHistory(symbol, { signal: controller.signal })
      .then((result) => {
        if (seqRef.current === id) {
          setHistory(result);
          setMeta(result?.meta || null);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (seqRef.current === id && err?.name !== 'AbortError') {
          setHistory(null);
          setLoading(false);
          setError(err?.message || 'Price history is unavailable right now.');
        }
      });
    return () => controller.abort();
  }, [ticker, retryTick]);

  const retry = useCallback(() => {
    setRetryTick((tick) => tick + 1);
  }, []);

  return { history, loading, error, meta, retry };
}
