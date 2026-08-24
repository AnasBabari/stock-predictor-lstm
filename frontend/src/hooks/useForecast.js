import { useCallback, useEffect, useRef, useState } from 'react';
import { clearBrowserModelCache, trainBrowserForecast } from '../ml/browserTrainingClient';
import { createSnapshotClient } from '../ml/snapshotClient';
import { fetchServerPrediction } from '../ml/serverModelClient';
import { isGlobalModelEnabled, loadGlobalModel } from '../ml/globalModelClient';
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

export function normalizeHorizonRequest(value) {
  if (value === 'auto' || value?.horizon_mode === 'auto') {
    return { horizon_mode: 'auto', requested_horizon: null };
  }
  const numeric = Number(value?.requested_horizon ?? value);
  if (![1, 3, 5, 7, 14, 30].includes(numeric)) {
    throw new Error('Forecast horizon must be Auto or one of 1, 3, 5, 7, 14, or 30 days.');
  }
  return { horizon_mode: 'explicit', requested_horizon: numeric };
}

export const forecastIdentity = (ticker, days, type, profile) => {
  const request = normalizeHorizonRequest(days);
  const horizon = request.horizon_mode === 'auto' ? 'auto' : request.requested_horizon;
  return `${ticker.trim().toUpperCase()}::${horizon}::${type}::${profile}`;
};

export function assertForecastIdentity(data, ticker, days, type) {
  const symbol = ticker.trim().toUpperCase();
  const request = normalizeHorizonRequest(days);
  const selectedDays = Number(data?.forecast_days);
  const horizonMatches = request.horizon_mode === 'auto'
    ? data?.requested_horizon_mode === 'auto' && Number.isFinite(selectedDays)
    : selectedDays === request.requested_horizon;
  if (!data || data.ticker !== symbol || !horizonMatches) {
    throw new Error('The forecast response does not match the selected ticker and horizon.');
  }
  const hasExpectedPayload =
    type === FORECAST_TYPES.PRICE
      ? data.predicted_prices?.length === selectedDays
      : data.direction_probabilities != null &&
        Number(data.direction_horizon_days) === selectedDays;
  if (!hasExpectedPayload || data.future_dates?.length !== selectedDays) {
    throw new Error('The forecast response is incomplete for the selected forecast type.');
  }
  return data;
}

export function browserResponse(snapshot, result, forecastType, days) {
  const resolvedDays = Number(result.selectedHorizon || result.days || days);
  const promotion = result.promotion || {};
  const validation = result.validation || {
    state: promotion.state || (promotion.promoted ? 'promoted' : 'experimental'),
    promoted: promotion.promoted === true,
    reasons: promotion.reasons || [],
    selected_horizon: resolvedDays,
    promoted_horizons: promotion.promoted_horizons || [],
    best_validated_horizon: promotion.best_validated_horizon || null,
  };
  const engine = {
    family: `${result.trainingProfile || 'balanced'}_tfjs_lstm`,
    role: validation.promoted ? 'browser_promoted' : 'browser_experimental',
    validation_state: validation.state,
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
    validation,
    schema_version: snapshot.schema_version,
    window_size: snapshot.window_size,
    feature_count: snapshot.feature_names.length,
    output_width: result.horizon || snapshot.output_width,
    architecture: `${result.trainingProfile || 'balanced'}_lstm_in_browser`,
    metric_source: result.metrics?.metric_source || 'browser_purged_holdout',
    data_snapshot: snapshot.data_snapshot,
    snapshot_id: snapshot.snapshot_id,
    requested_horizon_mode: result.requestedHorizonMode || 'explicit',
    development_selection: result.developmentSelection || null,
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
      forecast_days: resolvedDays,
      requested_horizon_mode: result.requestedHorizonMode || 'explicit',
      selected_horizon: resolvedDays,
      development_selection: result.developmentSelection || null,
      future_dates: snapshot.future_dates.slice(0, resolvedDays),
      // Direction v2 contract: one three-way call per origin.
      direction_horizon_days: result.direction_horizon_days,
      direction: result.direction,
      direction_probabilities: result.direction_probabilities,
      model_direction_probabilities: result.model_direction_probabilities,
      base_rate_direction_probabilities: result.base_rate_direction_probabilities,
      forecast_status: result.forecast_status,
      attention_weights: [],
      metrics: result.metrics,
      validation,
      sentiment: { status: 'context_only', detail: 'News is not used as a browser model feature.' },
      metadata,
    };
  }
  return {
    ticker: snapshot.ticker,
    forecast_days: resolvedDays,
    requested_horizon_mode: result.requestedHorizonMode || 'explicit',
    selected_horizon: resolvedDays,
    development_selection: result.developmentSelection || null,
    historical_dates: snapshot.dates,
    historical_prices: snapshot.historical_prices,
    future_dates: snapshot.future_dates.slice(0, resolvedDays),
    predicted_prices: result.predictedPrices,
    learned_prices: result.learnedPrices,
    model_forecast: result.model_forecast || {
      prices: result.predictedPrices,
      source: `browser_${result.trainingProfile || 'balanced'}_lstm`,
      candidate: 'balanced_tfjs_lstm',
      horizon: result.horizon || days,
    },
    benchmark: result.benchmark || {
      type: 'persistence',
      prices: result.persistence_forecast,
    },
    validation,
    persistence_forecast: result.persistence_forecast,
    ...(result.historical_error_band ? { historical_error_band: result.historical_error_band } : {}),
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
      const horizonRequest = normalizeHorizonRequest(days);
      const explicitDays = horizonRequest.requested_horizon;
      let serverPrediction = null;
      if (horizonRequest.horizon_mode === 'explicit' && DEPLOYMENT_TRAINING_MODE !== 'browser_only') {
        onProgress?.({ stage: 'checking_server', message: 'Checking for server-pretrained models...' });
        serverPrediction = await fetchServerPrediction(symbol, explicitDays, type, signal, {
          mode: DEPLOYMENT_TRAINING_MODE,
        });
      }
      if (serverPrediction) {
        onProgress?.({ stage: 'server_model_loaded', message: 'Loaded server-pretrained model.' });
        return serverPrediction;
      }

      if (horizonRequest.horizon_mode === 'explicit' && isGlobalModelEnabled()) {
        try {
          onProgress?.({ stage: 'checking_global', message: 'Checking verified global models…' });
          const tf = await import('@tensorflow/tfjs').catch(() => null);
          if (tf) {
            const globalResult = await loadGlobalModel(explicitDays, tf);
            if (globalResult) {
              onProgress?.({ stage: 'global_model_loaded', message: 'Executing certified global model…' });
              const snapshot = await snapshotClientRef.current.fetchTrainingSnapshot(symbol, signal);
              const features = snapshot.features;
              const prices = snapshot.historical_prices;
              const lastPrice = prices[prices.length - 1];
              if (features && features.length >= 60 && lastPrice > 0) {
                const windowFeatures = features.slice(features.length - 60);
                const inputTensor = tf.tensor3d([windowFeatures], [1, 60, windowFeatures[0].length]);
                const rawOutput = globalResult.model.predict(inputTensor);
                const outputArray = Array.isArray(rawOutput) ? await rawOutput[0].data() : await rawOutput.data();
                inputTensor.dispose();
                if (Array.isArray(rawOutput)) rawOutput.forEach((t) => t.dispose());
                else rawOutput.dispose();

                const h = Math.min(explicitDays, outputArray.length || explicitDays);
                const predictedPrices = [];
                const futureDates = snapshot.future_dates.slice(0, h);
                for (let i = 0; i < h; i += 1) {
                  const ret = Number(outputArray[i]) || 0;
                  predictedPrices.push(lastPrice * Math.exp(ret));
                }
                const alpha = Number(globalResult.artifact?.alpha ?? 1.0);
                return {
                  ticker: symbol,
                  forecast_days: explicitDays,
                  forecast_type: type,
                  current_price: lastPrice,
                  predicted_prices: predictedPrices,
                  learned_prices: predictedPrices,
                  persistence_forecast: Array(h).fill(lastPrice),
                  future_dates: futureDates,
                  historical_dates: snapshot.dates,
                  historical_prices: snapshot.historical_prices,
                  forecast_status: {
                    state: alpha > 0 ? 'promoted' : 'experimental_no_demonstrated_edge',
                    decision: alpha > 0 ? 'model' : 'persistence',
                    alpha,
                    label: `Certified Global Model (${globalResult.artifact?.name || 'global_v1'})`,
                  },
                  metadata: {
                    model_source: 'global_model',
                    artifact_name: globalResult.artifact?.name,
                    catalog_recorded_sha: globalResult.catalog?.recorded_sha,
                    alpha,
                    browser_training: false,
                    engine: {
                      family: 'global_model',
                      role: 'global_champion',
                      baseline_fallback: false,
                    },
                  },
                };
              }
            }
          }
        } catch {
          // Global model execution failed -> fail closed to browser training
        }
      }

      if (!BROWSER_TRAINING_ENABLED) {
        return fetchServerBaseline(symbol, explicitDays || 7, type, signal, new Error('Browser training is disabled by the deployment flag.'));
      }
      
      let snapshot;
      try {
        snapshot = await snapshotClientRef.current.fetchTrainingSnapshot(symbol, signal);
      } catch (error) {
        if (error?.name === 'AbortError') throw error;
        return fetchServerBaseline(symbol, explicitDays || 7, type, signal, error);
      }

      try {
        const result = await trainBrowserForecast({
          snapshot,
          forecastType: type,
          days,
          horizonMode: horizonRequest.horizon_mode,
          profile: trainingProfile,
          signal,
          onProgress,
        });
        return browserResponse(snapshot, result, type, days);
      } catch (error) {
        if (error?.name === 'AbortError') throw error;
        return fetchServerBaseline(symbol, explicitDays || 7, type, signal, error);
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
    async (request = {}) => {
      const overrideTicker = typeof request === 'string' ? request : request.ticker;
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
      const activeDays = request && typeof request === 'object' && request.days !== undefined
        ? request.days
        : forecastDays;
      const activeType = request && typeof request === 'object' && request.type
        ? request.type
        : forecastType;

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

