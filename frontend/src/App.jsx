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
import { browserTrainingSupported, clearBrowserModelCache, trainBrowserForecast } from './ml/browserTrainingClient';
import { fetchServerPrediction } from './ml/serverModelClient';
import { defaultTrainingProfile, expectedDurationLabel } from './ml/trainingProfiles';

const API_BASE = import.meta.env.VITE_API_URL || window.STOCKLSTM_API_BASE || '';
const BROWSER_TRAINING_ENABLED = import.meta.env.VITE_BROWSER_TRAINING_ENABLED !== 'false';
const THEME_KEY = 'stocklstm-theme:v1';
const WL_KEY = 'stocklstm-watchlist:v1';
const HIST_KEY = 'stocklstm-history:v1';
const PROFILE_KEY = 'stocklstm-training-profile:v1';
const MAX_HISTORY = 15;
const FORECAST_TYPES = {
  PRICE: 'price',
  TREND: 'trend',
};

const stageLabels = {
  queued: 'Preparing browser training data…',
  downloading_market_data: 'Downloading market data…',
  preparing_features: 'Preparing browser features…',
  checking_artifact: 'Checking your local model cache…',
  cache_hit: 'Loading your cached browser model…',
  capability_check: 'Checking this device’s training speed…',
  checkpoint_loaded: 'Resuming your local research benchmark…',
  evaluating_fold: 'Running walk-forward evaluation…',
  final_fit: 'Fitting the final local model…',
  training: 'Training your local model…',
  generating_forecast: 'Generating browser forecast…',
  completed: 'Forecast ready.',
  failed: 'Forecast could not be completed.',
};

function normalizePredictionError(reason) {
  if (reason instanceof Error) return reason;
  return new Error('Prediction request failed.');
}

function predictionErrorMessage(error) {
  const message = typeof error?.message === 'string' ? error.message.toLowerCase() : '';
  if (
    error instanceof TypeError ||
    message.includes('failed to fetch') ||
    message.includes('networkerror') ||
    message.includes('network error')
  ) {
    return 'Could not connect to the backend. Make sure the server is running.';
  }
  if (message.includes('timed out') || message.includes('timeout')) {
    return 'Prediction timed out. The shared work may still finish; try again shortly.';
  }
  if (
    message.includes('market data') ||
    message.includes('upstream') ||
    message.includes('data source')
  ) {
    return 'Market data is temporarily unavailable. Please try again later.';
  }
  if (message.includes('forecast model') || message.includes('prepared model')) {
    return 'No prepared forecast model is available for this ticker. Try an approved symbol.';
  }
  if (
    message.includes('capacity') ||
    message.includes('queue is full') ||
    message.includes('(503)') ||
    message.includes(' 503')
  ) {
    return 'Prediction capacity is currently full. Please try again shortly.';
  }
  if (message.includes('rate limit') || message.includes('(429)') || message.includes(' 429')) {
    return 'Too many prediction requests. Please wait before trying again.';
  }
  if (message.includes('invalid ticker') || message.includes('not enough data') || message.includes('(400)')) {
    return 'Invalid ticker or not enough data. Try a different symbol.';
  }
  return 'Prediction could not be completed. Please try again.';
}

const forecastIdentity = (ticker, days, type, profile) =>
  `${ticker.trim().toUpperCase()}::${Number(days)}::${type}::${profile}`;

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

function browserResponse(snapshot, result, forecastType, days) {
  const baselineFallback = result.baselineFallback === true || result.promotion?.promoted === false;
  const engine = {
    family: `${result.trainingProfile || 'balanced'}_tfjs_lstm`,
    role: baselineFallback ? 'baseline_fallback' : 'browser_learned',
    baseline_fallback: baselineFallback,
    cache_status: result.cacheStatus,
    backend: result.backend,
    execution_mode: result.executionMode,
  };
  const metadata = {
    model_version: result.modelVersion || 'tfjs-return-lstm-v4',
    architecture_version: result.architectureVersion,
    target_mode: result.targetMode,
    training_profile: result.trainingProfile,
    training_duration_ms: result.trainingDurationMs,
    selected_epochs: result.selectedEpochs,
    completed_epochs: result.completedEpochs,
    tfjs_version: result.tfjsVersion,
    storage_status: result.storageStatus,
    evaluation: result.evaluation,
    promotion: result.promotion,
    schema_version: snapshot.schema_version,
    window_size: snapshot.window_size,
    feature_count: snapshot.feature_names.length,
    output_width: result.horizon || snapshot.output_width,
    architecture: `${result.trainingProfile || 'balanced'}_lstm_in_browser`,
    metric_source: result.metrics?.metric_source || 'browser_purged_holdout',
    data_snapshot: snapshot.data_snapshot,
    snapshot_id: snapshot.snapshot_id,
    browser_training: true,
    engine,
    execution: {
      mode: result.executionMode,
      coalesced: false,
    },
    artifact_state_before: result.cacheStatus === 'hit' ? 'fresh' : 'missing',
    artifact_action: result.cacheStatus === 'hit' ? 'loaded' : 'trained',
  };
  if (forecastType === FORECAST_TYPES.TREND) {
    return {
      ticker: snapshot.ticker,
      forecast_days: days,
      future_dates: snapshot.future_dates.slice(0, days),
      directions: result.directions,
      probabilities: result.probabilities,
      attention_weights: [],
      metrics: result.metrics,
      sentiment: { status: 'context_only', detail: 'News is not used as a browser model feature.' },
      metadata,
    };
  }
  return {
    ticker: snapshot.ticker,
    forecast_days: days,
    historical_dates: snapshot.dates,
    historical_prices: snapshot.historical_prices,
    future_dates: snapshot.future_dates.slice(0, days),
    predicted_prices: result.predictedPrices,
    learned_prices: result.learnedPrices,
    metrics: result.metrics,
    metadata,
  };
}
export default function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem(THEME_KEY) || 'dark';
  });

  const [ticker, setTicker] = useState('');
  const [forecastDays, setForecastDays] = useState(7);
  const [daysView, setDaysView] = useState(21);
  const [forecastType, setForecastType] = useState(FORECAST_TYPES.PRICE);
  const [trainingProfile, setTrainingProfile] = useState(() => {
    const stored = localStorage.getItem(PROFILE_KEY);
    return ['quick', 'balanced', 'research'].includes(stored) ? stored : defaultTrainingProfile();
  });
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('');
  const [trainingProgress, setTrainingProgress] = useState(null);
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
  const forecastCacheRef = useRef(new Map());
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
      localStorage.setItem(PROFILE_KEY, trainingProfile);
    } catch {
      // The selected profile remains active for this session.
    }
  }, [trainingProfile]);

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

  const fetchServerBaseline = useCallback(async (symbol, days, type, signal, cause) => {
    const endpoint = type === FORECAST_TYPES.TREND ? '/api/v1/predict/direction' : '/api/v1/predict';
    const response = await fetch(`${API_BASE}${endpoint}?ticker=${encodeURIComponent(symbol)}&days=${days}`, { signal });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Prediction failed (${response.status})`);
    }
    const data = await response.json();
    data.metadata = {
      ...(data.metadata || {}),
      browser_training: false,
      browser_training_error: cause instanceof Error ? cause.message : 'unavailable',
      engine: {
        ...(data.metadata?.engine || {}),
        role: 'baseline_fallback',
        baseline_fallback: true,
      },
    };
    return data;
  }, []);
  const fetchPredictionData = useCallback(
    async (symbol, days, type, signal, onProgress) => {
      // Try fetching a high-quality server-pretrained model first
      onProgress?.({ stage: 'checking_server', message: 'Checking for server-pretrained models...' });
      const serverPrediction = await fetchServerPrediction(symbol, days, type, signal);
      if (serverPrediction) {
        onProgress?.({ stage: 'server_model_loaded', message: 'Loaded server-pretrained model.' });
        return serverPrediction;
      }

      if (!BROWSER_TRAINING_ENABLED) {
        return fetchServerBaseline(symbol, days, type, signal, new Error('Browser training is disabled by the deployment flag.'));
      }
      
      const trainingEndpoint = `${API_BASE}/api/v1/training-data?ticker=${encodeURIComponent(symbol)}`;
      let snapshot;
      try {
        const dataResponse = await fetch(trainingEndpoint, { signal });
        if (!dataResponse.ok) {
          const errorData = await dataResponse.json().catch(() => ({}));
          throw new Error(errorData.detail || `Training data failed (${dataResponse.status})`);
        }
        snapshot = await dataResponse.json();
      } catch (error) {
        if (error?.name === 'AbortError') throw error;
        return fetchServerBaseline(symbol, days, type, signal, error);
      }

      if (!browserTrainingSupported()) {
        return fetchServerBaseline(symbol, days, type, signal, new Error('Browser training is unavailable on this device.'));
      }

      try {
        onProgress?.({ stage: 'checking_artifact', message: 'Checking your local model cache…' });
        const result = await trainBrowserForecast({
          snapshot,
          forecastType: type,
          days,
          profile: trainingProfile,
          signal,
          onProgress,
        });
        onProgress?.({ stage: 'generating_forecast', message: 'Generating browser forecast…' });
        return browserResponse(snapshot, result, type, days);
      } catch (error) {
        if (error?.name === 'AbortError') throw error;
        return fetchServerBaseline(symbol, days, type, signal, error);
      }
    },
    [fetchServerBaseline, trainingProfile]
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

      if (trainingProfile === 'research') {
        const accepted = window.confirm(
          `Research mode runs a five-fold local benchmark and may take ${expectedDurationLabel('research')}. Continue?`
        );
        if (!accepted) return;
      }

      abortActiveRequest();
      const requestId = requestIdRef.current;
      abortControllerRef.current = new AbortController();
      const { signal } = abortControllerRef.current;

      setErrorMsg('');
      setIsLoading(true);
      setLoadingStage('');
      setTrainingProgress(null);
      setPredictionData(null);
      setStockInfo(null);

      try {
        const onProgress = (progress) => {
          if (requestIdRef.current !== requestId) return;
          setLoadingStage(progress.message || stageLabels[progress.stage] || 'Training your local model…');
          setTrainingProgress(progress);
        };
        const [predRes, infoRes] = await Promise.allSettled([
          fetchPredictionData(symbol, forecastDays, requestedType, signal, onProgress),
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
            forecastIdentity(symbol, forecastDays, requestedType, trainingProfile),
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
          throw normalizePredictionError(predRes.reason);
        }

        if (infoRes.status === 'fulfilled' && infoRes.value) {
          setStockInfo(infoRes.value);
        }
      } catch (err) {
        if (err.name === 'AbortError') return;
        const msg = predictionErrorMessage(err);
        setErrorMsg(msg);
        addToast('error', msg);
      } finally {
        if (requestIdRef.current === requestId) {
          setIsLoading(false);
          setLoadingStage('');
          setTrainingProgress(null);
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
      trainingProfile,
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

  const handleTrainingProfileChange = useCallback((nextProfile) => {
    abortActiveRequest();
    setTrainingProfile(nextProfile);
    setPredictionData(null);
    setStockInfo(null);
    setErrorMsg('');
    setIsLoading(false);
    setLoadingStage('');
    setTrainingProgress(null);
  }, [abortActiveRequest]);

  const handleExportCompleteAnalysis = useCallback(async () => {
    const tickerSymbol = (ticker || '').trim().toUpperCase();
    if (!tickerSymbol) {
      setErrorMsg('Please enter a ticker symbol.');
      addToast('error', 'Please enter a ticker symbol.');
      return;
    }

    const priceKey = forecastIdentity(tickerSymbol, forecastDays, FORECAST_TYPES.PRICE, trainingProfile);
    const trendKey = forecastIdentity(tickerSymbol, forecastDays, FORECAST_TYPES.TREND, trainingProfile);
    const cachedPrice = forecastCacheRef.current.get(priceKey);
    const cachedTrend = forecastCacheRef.current.get(trendKey);

    const ensureForecast = async (type) => {
      const key = forecastIdentity(tickerSymbol, forecastDays, type, trainingProfile);
      if (forecastCacheRef.current.has(key)) {
        return forecastCacheRef.current.get(key);
      }

      const signal = abortControllerRef.current?.signal;
      const onProgress = (progress) => {
        setLoadingStage(progress.message || stageLabels[progress.stage] || 'Training your local model…');
        setTrainingProgress(progress);
      };
      const data = await fetchPredictionData(tickerSymbol, forecastDays, type, signal, onProgress);
      const validated = assertForecastIdentity(data, tickerSymbol, forecastDays, type);
      forecastCacheRef.current.set(key, validated);
      return validated;
    };

    if (trainingProfile === 'research') {
      const accepted = window.confirm(
        `Complete Analysis may run two five-fold local benchmarks and take longer than ${expectedDurationLabel('research')}. Continue?`
      );
      if (!accepted) return;
    }

    abortActiveRequest();
    abortControllerRef.current = new AbortController();
    try {
      setIsLoading(true);
      setLoadingStage('');
      setTrainingProgress(null);
      const priceData = cachedPrice || await ensureForecast(FORECAST_TYPES.PRICE);
      const trendData = cachedTrend || await ensureForecast(FORECAST_TYPES.TREND);

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
      if (err?.name === 'AbortError') return;
      const msg = err.message || 'Failed to export complete analysis.';
      setErrorMsg(msg);
      addToast('error', msg);
    } finally {
      setIsLoading(false);
      setLoadingStage('');
      setTrainingProgress(null);
      abortControllerRef.current = null;
    }
  }, [abortActiveRequest, addToast, fetchPredictionData, forecastDays, ticker, trainingProfile]);

  const handleClearBrowserModels = useCallback(async () => {
    try {
      await clearBrowserModelCache();
      forecastCacheRef.current.clear();
      setPredictionData(null);
      addToast('success', 'Locally trained browser models cleared');
    } catch (error) {
      addToast('error', error?.message || 'Could not clear browser models');
    }
  }, [addToast]);
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
          trainingProfile={trainingProfile}
          setTrainingProfile={handleTrainingProfileChange}
          onForecastTypeChange={handleForecastTypeChange}
          onPredict={handlePredict}
          isLoading={isLoading}
          apiBase={API_BASE}
        />

        {errorMsg && <div className="error">{errorMsg}</div>}

        <LoadingIndicator
          isLoading={isLoading}
          stage={loadingStage}
          progress={trainingProgress}
          profile={trainingProfile}
          onCancel={abortActiveRequest}
        />

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

        {browserTrainingSupported() && (
          <button type="button" className="cache-clear-button" onClick={handleClearBrowserModels}>
            Clear locally trained models
          </button>
        )}

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
