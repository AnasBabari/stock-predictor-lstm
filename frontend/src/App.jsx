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
import { exportCompleteAnalysis } from './utils/exportService';

const API_BASE = import.meta.env.VITE_API_URL || window.STOCKLSTM_API_BASE || '';
const THEME_KEY = 'stocklstm-theme:v1';
const WL_KEY = 'stocklstm-watchlist:v1';
const HIST_KEY = 'stocklstm-history:v1';
const MAX_HISTORY = 15;
const FORECAST_TYPES = {
  PRICE: 'price',
  TREND: 'trend',
};
const STATUS_POLL_INTERVAL_MS = 1750;

const stageLabels = {
  queued: 'Waiting for prediction capacity…',
  downloading_market_data: 'Downloading market data…',
  preparing_features: 'Preparing market features…',
  checking_artifact: 'Checking for a compatible model…',
  training: 'Training a new model for this ticker…',
  generating_forecast: 'Generating forecast…',
  completed: 'Forecast ready.',
  failed: 'Forecast could not be completed.',
};

function createRequestId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  if (!globalThis.crypto?.getRandomValues) {
    throw new Error('Secure request identifiers are unavailable in this browser.');
  }
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

const forecastIdentity = (ticker, days, type) =>
  `${ticker.trim().toUpperCase()}::${Number(days)}::${type}`;

function assertForecastIdentity(data, ticker, days, type) {
  const symbol = ticker.trim().toUpperCase();
  if (!data || data.ticker !== symbol || Number(data.forecast_days) !== Number(days)) {
    throw new Error('The forecast response does not match the selected ticker and horizon.');
  }
  const hasExpectedPayload =
    type === FORECAST_TYPES.PRICE
      ? data.predicted_prices?.length === Number(days)
      : data.directions?.length === Number(days) && data.probabilities?.length === Number(days);
  if (!hasExpectedPayload || data.future_dates?.length !== Number(days)) {
    throw new Error('The forecast response is incomplete for the selected forecast type.');
  }
  return data;
}

export default function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem(THEME_KEY) || 'dark';
  });

  const [ticker, setTicker] = useState('');
  const [forecastDays, setForecastDays] = useState(7);
  const [daysView, setDaysView] = useState(21);
  const [forecastType, setForecastType] = useState(FORECAST_TYPES.PRICE);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('');
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
  const statusPollRef = useRef(null);
  const requestIdRef = useRef(0);
  const forecastCacheRef = useRef(new Map());
  const chartRef = useRef(null);

  const abortActiveRequest = useCallback(() => {
    requestIdRef.current += 1;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (statusPollRef.current) {
      clearInterval(statusPollRef.current);
      statusPollRef.current = null;
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
    return abortActiveRequest;
  }, [abortActiveRequest]);

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
    async (symbol, days, type, signal, requestId) => {
      const endpoint =
        type === FORECAST_TYPES.TREND ? '/api/v1/predict/direction' : '/api/v1/predict';
      const headers = requestId ? { 'X-Prediction-Request-ID': requestId } : undefined;
      const res = await fetch(`${API_BASE}${endpoint}?ticker=${symbol}&days=${days}`, { signal, headers });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Prediction failed (${res.status})`);
      }

      return res.json();
    },
    []
  );

  const startStatusPolling = useCallback((requestId, requestIdNumber, signal) => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/prediction-status/${requestId}`, { signal });
        if (!res.ok) return;
        const status = await res.json();
        const stageLabel = stageLabels[status.stage];
        if (requestIdRef.current === requestIdNumber && stageLabel) {
          setLoadingStage(stageLabel);
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          // The prediction request remains the source of truth for user-facing errors.
        }
      }
    };
    void poll();
    statusPollRef.current = setInterval(() => void poll(), STATUS_POLL_INTERVAL_MS);
  }, []);

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
      setLoadingStage('');
      setPredictionData(null);
      setStockInfo(null);

      try {
        const requestToken = createRequestId();
        startStatusPolling(requestToken, requestId, signal);
        const [predRes, infoRes] = await Promise.allSettled([
          fetchPredictionData(symbol, forecastDays, requestedType, signal, requestToken),
          fetchStockInfo(symbol, signal),
        ]);

        if (requestIdRef.current !== requestId) {
          return;
        }

        if (predRes.status === 'fulfilled') {
          const fetchedData = assertForecastIdentity(
            predRes.value,
            symbol,
            forecastDays,
            requestedType
          );
          forecastCacheRef.current.set(
            forecastIdentity(symbol, forecastDays, requestedType),
            fetchedData
          );
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
          : err.message.includes('capacity')
          ? 'Prediction capacity is currently full. Please try again shortly.'
          : err.message.includes('timed out')
          ? 'Prediction timed out. The shared work may still finish; try again shortly.'
          : err.message.includes('Market data')
          ? 'Market data is temporarily unavailable. Please try again later.'
          : err.message.includes('400')
          ? 'Invalid ticker or not enough data. Try a different symbol.'
          : err.message;
        setErrorMsg(msg);
        addToast('error', msg);
      } finally {
        if (requestIdRef.current === requestId) {
          if (statusPollRef.current) {
            clearInterval(statusPollRef.current);
            statusPollRef.current = null;
          }
          setIsLoading(false);
          setLoadingStage('');
        }
      }
    },
    [
      abortActiveRequest,
      addToast,
      fetchPredictionData,
      fetchStockInfo,
      forecastDays,
      forecastType,
      startStatusPolling,
    ]
  );

  const handleForecastTypeChange = useCallback(
    (nextType) => {
      if (nextType === forecastType) return;
      abortActiveRequest();
      setForecastType(nextType);
      setPredictionData(null);
      setErrorMsg('');
      setIsLoading(false);
      setLoadingStage('');
    },
    [abortActiveRequest, forecastType]
  );

  const handleTickerChange = useCallback(
    (nextTicker) => {
      abortActiveRequest();
      setTicker(nextTicker);
      setPredictionData(null);
      setStockInfo(null);
      setErrorMsg('');
      setIsLoading(false);
      setLoadingStage('');
    },
    [abortActiveRequest]
  );

  const handleForecastDaysChange = useCallback(
    (nextDays) => {
      abortActiveRequest();
      setForecastDays(nextDays);
      setPredictionData(null);
      setStockInfo(null);
      setErrorMsg('');
      setIsLoading(false);
      setLoadingStage('');
    },
    [abortActiveRequest]
  );

  const handleExportCompleteAnalysis = useCallback(async () => {
    const tickerSymbol = (ticker || '').trim().toUpperCase();
    if (!tickerSymbol) {
      setErrorMsg('Please enter a ticker symbol.');
      addToast('error', 'Please enter a ticker symbol.');
      return;
    }

    const priceKey = forecastIdentity(tickerSymbol, forecastDays, FORECAST_TYPES.PRICE);
    const trendKey = forecastIdentity(tickerSymbol, forecastDays, FORECAST_TYPES.TREND);
    const cachedPrice = forecastCacheRef.current.get(priceKey);
    const cachedTrend = forecastCacheRef.current.get(trendKey);

    const ensureForecast = async (type) => {
      const key = forecastIdentity(tickerSymbol, forecastDays, type);
      if (forecastCacheRef.current.has(key)) {
        return forecastCacheRef.current.get(key);
      }

      const controller = new AbortController();
      const data = await fetchPredictionData(tickerSymbol, forecastDays, type, controller.signal);
      const validated = assertForecastIdentity(data, tickerSymbol, forecastDays, type);
      forecastCacheRef.current.set(key, validated);
      return validated;
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
      assertForecastIdentity(priceData, tickerSymbol, forecastDays, FORECAST_TYPES.PRICE);
      assertForecastIdentity(trendData, tickerSymbol, forecastDays, FORECAST_TYPES.TREND);

      const metadata = {
        ticker: tickerSymbol,
        generated_at: new Date().toISOString(),
        forecast_days: forecastDays,
        window_size: priceData.metadata?.window_size,
        price_model: priceData.metadata?.architecture,
        price_model_version: priceData.metadata?.model_version,
        direction_model: trendData.metadata?.architecture,
        direction_model_version: trendData.metadata?.model_version,
        backend_api_version: priceData.metadata?.model_version,
        price_metric_source: priceData.metrics?.metric_source,
        direction_metric_source: trendData.metrics?.metric_source,
      };

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
          setTicker={handleTickerChange}
          forecastDays={forecastDays}
          setForecastDays={handleForecastDaysChange}
          forecastType={forecastType}
          onForecastTypeChange={handleForecastTypeChange}
          onPredict={handlePredict}
          isLoading={isLoading}
          apiBase={API_BASE}
        />

        {errorMsg && <div className="error">{errorMsg}</div>}

        <LoadingIndicator isLoading={isLoading} stage={loadingStage} />

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
