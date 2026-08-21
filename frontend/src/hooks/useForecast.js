import { useCallback, useEffect, useRef, useState } from 'react';
import { clearBrowserModelCache, trainBrowserForecast } from '../ml/browserTrainingClient';
import { createSnapshotClient } from '../ml/snapshotClient';
import { fetchServerPrediction } from '../ml/serverModelClient';
import { defaultTrainingProfile } from '../ml/trainingProfiles';
import { safeGet, safeSet } from '../utils/safeStorage';

const API_BASE = import.meta.env.VITE_API_URL || window.STOCKLSTM_API_BASE || '';
const BROWSER_TRAINING_ENABLED = import.meta.env.VITE_BROWSER_TRAINING_ENABLED !== 'false';
const DEPLOYMENT_TRAINING_MODE = (
  window.STOCKLSTM_TRAINING_MODE ||
  import.meta.env.VITE_TRAINING_MODE ||
  'browser_only'
).toLowerCase();

const PROFILE_KEY = 'stocklstm-training-profile:v1';

export const FORECAST_TYPES = {
  PRICE: 'price',
  TREND: 'trend',
};

export const stageLabels = {
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

export function predictionErrorMessage(error) {
  const message = typeof error?.message === 'string' ? error.message.toLowerCase() : '';
  if (
    error instanceof TypeError ||
    message.includes('failed to fetch') ||
    message.includes('networkerror') ||
    message.includes('network error')
  ) {
    return 'Could not connect to the backend. Make sure the server is running.';
  }
  if (
    message.includes('server forecast') ||
    message.includes('server prediction') ||
    message.includes('failed validation')
  ) {
    return error.message;
  }
  if (message.includes('timed out') || message.includes('timeout')) {
    return 'Prediction timed out. The shared work may still finish; try again shortly.';
  }
  if (message.includes('no market data')) {
    return 'Unknown or untracked symbol. Try a different ticker.';
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

export const forecastIdentity = (ticker, days, type, profile) =>
  `${ticker.trim().toUpperCase()}::${Number(days)}::${type}::${profile}`;

export function assertForecastIdentity(data, ticker, days, type) {
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

export function browserResponse(snapshot, result, forecastType, days) {
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
      model_directions: result.model_directions,
      model_probabilities: result.model_probabilities,
      persistence_directions: result.persistence_directions,
      persistence_probabilities: result.persistence_probabilities,
      forecast_status: result.forecast_status,
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
    persistence_forecast: result.persistence_forecast,
    forecast_status: result.forecast_status,
    metrics: result.metrics,
    ...(result.evaluation_series ? { evaluation_series: result.evaluation_series } : {}),
    metadata,
  };
}

export function useForecast({ addToast, onNewTickerSearched }) {
  const [ticker, setTicker] = useState('');
  const [forecastDays, setForecastDays] = useState(7);
  const [daysView, setDaysView] = useState(21);
  const [forecastType, setForecastType] = useState(FORECAST_TYPES.PRICE);
  const [trainingProfile, setTrainingProfile] = useState(() => {
    const stored = safeGet(PROFILE_KEY);
    return ['quick', 'balanced', 'research'].includes(stored) ? stored : defaultTrainingProfile();
  });
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('');
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [predictionData, setPredictionData] = useState(null);
  const [stockInfo, setStockInfo] = useState(null);

  const abortControllerRef = useRef(null);
  const requestIdRef = useRef(0);
  const forecastCacheRef = useRef(new Map());
  const chartRef = useRef(null);
  const snapshotClientRef = useRef(null);
  if (!snapshotClientRef.current) {
    snapshotClientRef.current = createSnapshotClient({ baseUrl: API_BASE });
  }

  useEffect(() => {
    safeSet(PROFILE_KEY, trainingProfile);
  }, [trainingProfile]);

  const abortActiveRequest = useCallback(() => {
    requestIdRef.current += 1;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return abortActiveRequest;
  }, [abortActiveRequest]);

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
      let serverPrediction = null;
      if (DEPLOYMENT_TRAINING_MODE !== 'browser_only') {
        onProgress?.({ stage: 'checking_server', message: 'Checking for server-pretrained models...' });
        serverPrediction = await fetchServerPrediction(symbol, days, type, signal, {
          mode: DEPLOYMENT_TRAINING_MODE,
        });
      }
      if (serverPrediction) {
        onProgress?.({ stage: 'server_model_loaded', message: 'Loaded server-pretrained model.' });
        return serverPrediction;
      }

      if (!BROWSER_TRAINING_ENABLED) {
        return fetchServerBaseline(symbol, days, type, signal, new Error('Browser training is disabled by the deployment flag.'));
      }
      
      let snapshot;
      try {
        snapshot = await snapshotClientRef.current.fetchTrainingSnapshot(symbol, signal);
      } catch (error) {
        if (error?.name === 'AbortError') throw error;
        return fetchServerBaseline(symbol, days, type, signal, error);
      }

      try {
        const result = await trainBrowserForecast({
          snapshot,
          forecastType: type,
          days,
          profile: trainingProfile,
          signal,
          onProgress,
        });
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
      try {
        const res = await fetch(`${API_BASE}/api/v1/info?ticker=${encodeURIComponent(symbol)}`, { signal });
        if (res.ok) {
          const info = await res.json();
          setStockInfo(info);
        }
      } catch (err) {
        if (err?.name !== 'AbortError') {
          setStockInfo(null);
        }
      }
    },
    []
  );

  const handlePredict = useCallback(
    async (overrideTicker, overrideDays, overrideType) => {
      if (
        overrideTicker !== undefined &&
        !(typeof overrideTicker === 'string' && overrideTicker.trim())
      ) {
        // An explicitly provided but unusable ticker must never silently fall
        // back to the currently active ticker.
        const msg = 'Please enter a valid stock ticker symbol.';
        setErrorMsg(msg);
        addToast('error', msg);
        return;
      }
      const activeTicker = (typeof overrideTicker === 'string' && overrideTicker.trim() ? overrideTicker : ticker).toUpperCase().trim();
      let activeDays = forecastDays;
      let activeType = forecastType;

      if (typeof overrideDays === 'number') {
        activeDays = overrideDays;
      } else if (typeof overrideDays === 'string' && (overrideDays === 'price' || overrideDays === 'trend')) {
        activeType = overrideDays;
      }

      if (typeof overrideType === 'string') {
        activeType = overrideType;
      }

      if (!activeTicker) {
        setErrorMsg('Please enter a stock ticker symbol.');
        addToast('error', 'Please enter a stock ticker symbol.');
        return;
      }


      abortActiveRequest();
      const currentRequestId = requestIdRef.current;
      const controller = new AbortController();
      abortControllerRef.current = controller;

      const cacheKey = forecastIdentity(activeTicker, activeDays, activeType, trainingProfile);
      const cached = forecastCacheRef.current.get(cacheKey);
      if (cached) {
        setPredictionData(cached);
        setErrorMsg('');
        fetchStockInfo(activeTicker, controller.signal);
        onNewTickerSearched?.(cached);
        return;
      }

      setIsLoading(true);
      setLoadingStage('queued');
      setTrainingProgress(null);
      setErrorMsg('');

      try {
        fetchStockInfo(activeTicker, controller.signal);
        const data = await fetchPredictionData(
          activeTicker,
          activeDays,
          activeType,
          controller.signal,
          (progress) => {
            if (requestIdRef.current === currentRequestId) {
              setLoadingStage(progress.message || stageLabels[progress.stage] || 'Training your local model…');
              setTrainingProgress(progress);
            }
          }

        );

        if (requestIdRef.current === currentRequestId) {
          assertForecastIdentity(data, activeTicker, activeDays, activeType);
          forecastCacheRef.current.set(cacheKey, data);
          setPredictionData(data);
          onNewTickerSearched?.(data);
          addToast('success', `${activeTicker} ${activeType} forecast ready`);
        }
      } catch (err) {
        if (err?.name === 'AbortError') return;
        if (requestIdRef.current === currentRequestId) {
          const msg = predictionErrorMessage(err);
          setErrorMsg(msg);
          addToast('error', msg);
        }
      } finally {
        if (requestIdRef.current === currentRequestId) {
          setIsLoading(false);
          setLoadingStage('');
          setTrainingProgress(null);
          abortControllerRef.current = null;
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
      onNewTickerSearched,
      ticker,
      trainingProfile,
    ]
  );

  const handleCancelRequest = useCallback(() => {
    abortActiveRequest();
    setIsLoading(false);
    setLoadingStage('');
    setTrainingProgress(null);
    addToast('info', 'Forecast request cancelled.');
  }, [abortActiveRequest, addToast]);

  const handleClearBrowserModels = useCallback(async () => {
    try {
      await clearBrowserModelCache();
      forecastCacheRef.current.clear();
      snapshotClientRef.current?.clear();
      setPredictionData(null);
      addToast('success', 'Locally trained browser models cleared');
    } catch (error) {
      addToast('error', error?.message || 'Could not clear browser models');
    }
  }, [addToast]);

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

  const handleTrainingProfileChange = useCallback(
    (nextProfile) => {
      abortActiveRequest();
      setTrainingProfile(nextProfile);
      setPredictionData(null);
      setStockInfo(null);
      setErrorMsg('');
      setIsLoading(false);
      setLoadingStage('');
      setTrainingProgress(null);
    },
    [abortActiveRequest]
  );

  return {
    ticker,
    setTicker: handleTickerChange,
    forecastDays,
    setForecastDays: handleForecastDaysChange,
    daysView,
    setDaysView,
    forecastType,
    setForecastType: handleForecastTypeChange,
    trainingProfile,
    setTrainingProfile: handleTrainingProfileChange,
    handleForecastTypeChange,
    handleTickerChange,
    handleForecastDaysChange,
    handleTrainingProfileChange,
    isLoading,
    loadingStage,
    trainingProgress,
    errorMsg,
    setErrorMsg,
    predictionData,
    setPredictionData,
    stockInfo,
    setStockInfo,
    chartRef,
    forecastCacheRef,
    fetchPredictionData,
    handlePredict,
    handleCancelRequest,
    handleClearBrowserModels,
    apiBase: API_BASE,
  };
}

