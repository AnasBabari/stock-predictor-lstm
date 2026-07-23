import React, { useState, useEffect, useCallback, useRef } from 'react';
import SplashScreen from './components/SplashScreen';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import SearchCard from './components/SearchCard';
import LoadingIndicator from './components/LoadingIndicator';
import StockInfoGrid from './components/StockInfoGrid';
import StatsBar from './components/StatsBar';
import StockChart from './components/StockChart';
import MetricsCard from './components/MetricsCard';
import Watchlist from './components/Watchlist';
import PredictionHistory from './components/PredictionHistory';
import ToastContainer from './components/ToastContainer';

const API_BASE = import.meta.env.VITE_API_URL || window.STOCKLSTM_API_BASE || '';
const THEME_KEY = 'stocklstm-theme:v1';
const WL_KEY = 'stocklstm-watchlist:v1';
const HIST_KEY = 'stocklstm-history:v1';
const MAX_HISTORY = 15;

export default function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem(THEME_KEY) || 'dark';
  });

  const [ticker, setTicker] = useState('');
  const [forecastDays, setForecastDays] = useState(7);
  const [daysView, setDaysView] = useState(21);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [stockData, setStockData] = useState(null);
  const [stockInfo, setStockInfo] = useState(null);
  const [toasts, setToasts] = useState([]);

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

  const abortControllerRef = useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // Ignore storage errors
    }
  }, [theme]);

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

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  const addToast = useCallback((type, message) => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { id, type, message }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3000);
  }, []);

  const handlePredict = useCallback(
    async (tickerToPredict) => {
      const symbol = (tickerToPredict || '').trim().toUpperCase();
      if (!symbol) {
        setErrorMsg('Please enter a ticker symbol.');
        addToast('error', 'Please enter a ticker symbol.');
        return;
      }

      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();
      const { signal } = abortControllerRef.current;

      setErrorMsg('');
      setIsLoading(true);

      try {
        const [predRes, infoRes] = await Promise.allSettled([
          fetch(`${API_BASE}/api/v1/predict?ticker=${symbol}&days=${forecastDays}`, { signal }),
          fetch(`${API_BASE}/api/v1/info?ticker=${symbol}`, { signal }),
        ]);

        if (predRes.status === 'fulfilled' && predRes.value.ok) {
          const fetchedData = await predRes.value.json();
          setStockData(fetchedData);

          const lastClose = fetchedData.historical_prices[fetchedData.historical_prices.length - 1];
          const forecast = fetchedData.predicted_prices[fetchedData.predicted_prices.length - 1];
          const changePct = (((forecast - lastClose) / lastClose) * 100).toFixed(2);

          const newHistoryItem = {
            ticker: fetchedData.ticker,
            lastClose,
            forecast,
            change: changePct,
            days: fetchedData.forecast_days,
            date: new Date().toISOString(),
          };

          setHistory((prev) => {
            const nowIso = newHistoryItem.date.slice(0, 16);
            const filtered = prev.filter(
              (h) => !(h.ticker === fetchedData.ticker && h.date?.startsWith(nowIso))
            );
            return [newHistoryItem, ...filtered].slice(0, MAX_HISTORY);
          });

          addToast('success', `Forecast ready for ${fetchedData.ticker}`);
        } else if (predRes.status === 'fulfilled') {
          const errData = await predRes.value.json().catch(() => ({}));
          throw new Error(errData.detail || `Prediction failed (${predRes.value.status})`);
        } else if (predRes.reason.name === 'AbortError') {
          return;
        } else {
          throw new Error('Network error. Failed to fetch prediction.');
        }

        if (infoRes.status === 'fulfilled' && infoRes.value.ok) {
          const fetchedInfo = await infoRes.value.json();
          setStockInfo(fetchedInfo);
        } else {
          setStockInfo(null);
        }
      } catch (err) {
        if (err.name === 'AbortError') return;
        const msg = err.message.includes('Failed to fetch')
          ? 'Could not connect to the backend. Make sure the server is running.'
          : err.message.includes('400')
          ? 'Invalid ticker or not enough data. Try a different symbol.'
          : err.message;
        setErrorMsg(msg);
        addToast('error', msg);
      } finally {
        setIsLoading(false);
      }
    },
    [forecastDays, addToast]
  );

  const handleAddWatchlist = useCallback(
    (data) => {
      if (!data) return;
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
    [watchlist, stockInfo, addToast]
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

  const handleSelectTicker = useCallback(
    (selectedSymbol) => {
      setTicker(selectedSymbol);
      handlePredict(selectedSymbol);
    },
    [handlePredict]
  );

  return (
    <>
      <SplashScreen />

      {/* Ambient Orbs */}
      <div className="ambient-orbs" aria-hidden="true">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
        <div className="orb orb-3"></div>
      </div>

      <Navbar theme={theme} onToggleTheme={toggleTheme} />

      <main className="container">
        <HeroSection />

        <SearchCard
          ticker={ticker}
          setTicker={setTicker}
          forecastDays={forecastDays}
          setForecastDays={setForecastDays}
          onPredict={handlePredict}
          isLoading={isLoading}
          apiBase={API_BASE}
        />

        {errorMsg && <div className="error">{errorMsg}</div>}

        <LoadingIndicator isLoading={isLoading} />

        <StockInfoGrid info={stockInfo} />

        <StatsBar stockData={stockData} />

        <StockChart
          stockData={stockData}
          daysView={daysView}
          setDaysView={setDaysView}
          theme={theme}
          onAddWatchlist={handleAddWatchlist}
          onToast={addToast}
        />

        <MetricsCard stockData={stockData} />

        <div className="bottom-panels">
          <Watchlist
            items={watchlist}
            onSelectTicker={handleSelectTicker}
            onRemoveItem={handleRemoveWatchlist}
            onClearAll={handleClearWatchlist}
          />

          <PredictionHistory
            items={history}
            onSelectTicker={handleSelectTicker}
            onClearAll={handleClearHistory}
          />
        </div>
      </main>

      <footer className="footer">
        <p>Built for educational purposes only · Forecasts are not financial advice</p>
      </footer>

      <ToastContainer toasts={toasts} />
    </>
  );
}
