import { useCallback, useEffect, useState } from 'react';

const WL_KEY = 'stock_lstm_watchlist';
const HIST_KEY = 'stock_lstm_history';

export function useWatchlist({ addToast, forecastType, stockInfo }) {
  const [watchlist, setWatchlist] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(WL_KEY)) || [];
    } catch {
      return [];
    }
  });

  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(HIST_KEY)) || [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(WL_KEY, JSON.stringify(watchlist));
    } catch {
      // Ignore storage errors
    }
  }, [watchlist]);

  useEffect(() => {
    try {
      localStorage.setItem(HIST_KEY, JSON.stringify(history));
    } catch {
      // Ignore storage errors
    }
  }, [history]);

  const addToHistory = useCallback((symbol) => {
    const clean = symbol.toUpperCase().trim();
    if (!clean) return;
    setHistory((prev) => [clean, ...prev.filter((item) => item !== clean)].slice(0, 10));
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
