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
import ForecastChartActions from './components/ForecastChartActions';
import Watchlist from './components/Watchlist';
import PredictionHistory from './components/PredictionHistory';
import ToastContainer from './components/ToastContainer';

const API_BASE = import.meta.env.VITE_API_URL || window.STOCKLSTM_API_BASE || '';
const THEME_KEY = 'stocklstm-theme:v1';
const WL_KEY = 'stocklstm-watchlist:v1';
const HIST_KEY = 'stocklstm-history:v1';
const MAX_HISTORY = 15;
const FORECAST_TYPES = {
  PRICE: 'price',
  TREND: 'trend',
};

export default function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem(THEME_KEY) || 'dark';
  });

  const [ticker, setTicker] = useState('');
  const [forecastDays, setForecastDays] = useState(7);
  const [daysView, setDaysView] = useState(21);
  const [forecastType, setForecastType] = useState(FORECAST_TYPES.PRICE);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [predictionData, setPredictionData] = useState(null);
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
  const requestIdRef = useRef(0);
  const forecastCacheRef = useRef({
    [FORECAST_TYPES.PRICE]: null,
    [FORECAST_TYPES.TREND]: null,
  });
  const chartRef = useRef(null);

  const abortActiveRequest = useCallback(() => {
    requestIdRef.current += 1;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

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

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

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

  const fetchPredictionData = useCallback(
    async (symbol, days, type, signal) => {
      const endpoint =
        type === FORECAST_TYPES.TREND ? '/api/v1/predict/direction' : '/api/v1/predict';
      const res = await fetch(`${API_BASE}${endpoint}?ticker=${symbol}&days=${days}`, { signal });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Prediction failed (${res.status})`);
      }

      return res.json();
    },
    []
  );

  const fetchStockInfo = useCallback(
    async (symbol, signal) => {
      const res = await fetch(`${API_BASE}/api/v1/info?ticker=${symbol}`, { signal });
      if (!res.ok) return null;
      return res.json();
    },
    []
  );

  const handlePredict = useCallback(
    async (tickerToPredict, requestedType = forecastType) => {
      const symbol = (tickerToPredict || '').trim().toUpperCase();
      if (!symbol) {
        setErrorMsg('Please enter a ticker symbol.');
        addToast('error', 'Please enter a ticker symbol.');
        return;
      }

      abortActiveRequest();
      const requestId = requestIdRef.current;
      abortControllerRef.current = new AbortController();
      const { signal } = abortControllerRef.current;

      setErrorMsg('');
      setIsLoading(true);
      setPredictionData(null);

      try {
        const [predRes, infoRes] = await Promise.allSettled([
          fetchPredictionData(symbol, forecastDays, requestedType, signal),
          fetchStockInfo(symbol, signal),
        ]);

        if (requestIdRef.current !== requestId) {
          return;
        }

        if (predRes.status === 'fulfilled') {
          const fetchedData = predRes.value;
          forecastCacheRef.current[requestedType] = fetchedData;
          setPredictionData(fetchedData);

          if (requestedType === FORECAST_TYPES.PRICE && fetchedData.historical_prices?.length) {
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
          }

          addToast(
            'success',
            `${requestedType === FORECAST_TYPES.TREND ? 'Trend' : 'Price'} forecast ready for ${fetchedData.ticker}`
          );
        } else {
          throw new Error('Network error. Failed to fetch prediction.');
        }

        if (infoRes.status === 'fulfilled' && infoRes.value) {
          setStockInfo(infoRes.value);
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
        if (requestIdRef.current === requestId) {
          setIsLoading(false);
        }
      }
    },
    [abortActiveRequest, addToast, fetchPredictionData, fetchStockInfo, forecastDays, forecastType]
  );

  const handleForecastTypeChange = useCallback(
    (nextType) => {
      if (nextType === forecastType) return;
      abortActiveRequest();
      setForecastType(nextType);
      setPredictionData(null);
      setErrorMsg('');
      setIsLoading(false);
    },
    [abortActiveRequest, forecastType]
  );

  const handleExportCompleteAnalysis = useCallback(async () => {
    const tickerSymbol = (ticker || '').trim().toUpperCase();
    if (!tickerSymbol) {
      setErrorMsg('Please enter a ticker symbol.');
      addToast('error', 'Please enter a ticker symbol.');
      return;
    }

    const cachedPrice = forecastCacheRef.current[FORECAST_TYPES.PRICE];
    const cachedTrend = forecastCacheRef.current[FORECAST_TYPES.TREND];

    const ensureForecast = async (type) => {
      if (forecastCacheRef.current[type]) {
        return forecastCacheRef.current[type];
      }

      const controller = new AbortController();
      const data = await fetchPredictionData(tickerSymbol, forecastDays, type, controller.signal);
      forecastCacheRef.current[type] = data;
      return data;
    };

    try {
      setIsLoading(true);
      const [priceData, trendData] = await Promise.all([
        cachedPrice ? Promise.resolve(cachedPrice) : ensureForecast(FORECAST_TYPES.PRICE),
        cachedTrend ? Promise.resolve(cachedTrend) : ensureForecast(FORECAST_TYPES.TREND),
      ]);

      if (!priceData || !trendData) {
        throw new Error('Both forecast types are required to export the complete analysis.');
      }

      const metadata = {
        ticker: tickerSymbol,
        generated_at: new Date().toISOString(),
        forecast_days: forecastDays,
        window_size: 60,
        price_model: 'LSTM',
        price_model_version: 'v1',
        direction_model: 'Attention-LSTM',
        direction_model_version: 'v1',
        backend_api_version: '3.0',
      };

      const { exportCompleteAnalysis } = await import('./utils/exportService');
      await exportCompleteAnalysis({
        priceData,
        directionData: trendData,
        metadata,
      });
      addToast('success', 'Complete analysis exported as ZIP');
    } catch (err) {
      const msg = err.message || 'Failed to export complete analysis.';
      setErrorMsg(msg);
      addToast('error', msg);
    } finally {
      setIsLoading(false);
    }
  }, [addToast, fetchPredictionData, forecastDays, ticker]);

  const handleAddWatchlist = useCallback(
    (data) => {
      if (!data || forecastType !== FORECAST_TYPES.PRICE || !data.historical_prices?.length) {
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
          forecastType={forecastType}
          onForecastTypeChange={handleForecastTypeChange}
          onPredict={handlePredict}
          isLoading={isLoading}
          apiBase={API_BASE}
        />

        {errorMsg && <div className="error">{errorMsg}</div>}

        <LoadingIndicator isLoading={isLoading} />

        <StockInfoGrid info={stockInfo} />

        <StatsBar stockData={predictionData} forecastType={forecastType} />

        <div className="chart-panel">
          <StockChart
            ref={chartRef}
            stockData={predictionData}
            forecastType={forecastType}
            daysView={daysView}
            setDaysView={setDaysView}
            theme={theme}
          />
          <ForecastChartActions
            chartRef={chartRef}
            stockData={predictionData}
            forecastType={forecastType}
            onAddWatchlist={handleAddWatchlist}
            onToast={addToast}
            onExportCompleteAnalysis={handleExportCompleteAnalysis}
          />
        </div>

        <MetricsCard stockData={predictionData} forecastType={forecastType} />

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
