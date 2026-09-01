import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchVolatilityForecast } from '../ml/volatilityClient';

const API_BASE = import.meta.env.VITE_API_URL || window.STOCKLSTM_API_BASE || '';

export const VOLATILITY_SERVING_ENABLED = true;

export const FORECAST_TYPES = {
  PRICE: 'price',
  TREND: 'trend',
};

export const stageLabels = {
  queued: 'Preparing request…',
  volatility_snapshot: 'Building causal market snapshot…',
  volatility_inference: 'Calculating volatility baseline…',
  completed: 'Forecast ready.',
  failed: 'Forecast could not be completed.',
};

export function predictionErrorMessage(error) {
  const message = typeof error?.message === 'string' ? error.message.toLowerCase() : '';
  if (error?.code === 'abstain_no_certified_model') {
    return 'The legacy global model is unavailable. Try the active volatility forecast again shortly.';
  }
  if (error?.code === 'certified_horizon_unavailable') {
    return 'The selected volatility horizon is not available from the active data service.';
  }
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
  if (error?.name === 'VolatilityApiError' && error?.httpStatus === 503) {
    return 'The volatility forecast service is temporarily unavailable. Please retry shortly.';
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
  return error?.message || 'Prediction could not be completed. Please try again.';
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

export const forecastIdentity = (ticker, days, type) => {
  const request = normalizeHorizonRequest(days);
  const horizon = request.horizon_mode === 'auto' ? 'auto' : request.requested_horizon;
  return `${ticker.trim().toUpperCase()}::${horizon}::${type}`;
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
  const certifiedHead = data?.metadata?.engine?.certified_head;
  const isCertifiedDistribution =
    (certifiedHead === 'volatility' || certifiedHead === 'return_distribution')
    && data?.volatility_cone != null;
  const isVolatilityForecast = data?.metadata?.engine?.volatility_forecast === true
    && data?.volatility_cone != null;
  const hasExpectedPayload =
    type === FORECAST_TYPES.PRICE
      ? (isCertifiedDistribution || isVolatilityForecast)
        ? ['p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95'].every(
          (key) => data.volatility_cone?.[key]?.length === selectedDays,
        )
        : data.predicted_prices?.length === selectedDays
      : data.direction_probabilities != null &&
        Number(data.direction_horizon_days) === selectedDays;
  if (!hasExpectedPayload || data.future_dates?.length !== selectedDays) {
    throw new Error('The forecast response is incomplete for the selected forecast type.');
  }
  return data;
}

export function useForecast({ addToast, onNewTickerSearched }) {
  const [ticker, setTicker] = useState('');
  const [forecastDays, setForecastDays] = useState(7);
  const [daysView, setDaysView] = useState(21);
  const [forecastType, setForecastType] = useState(FORECAST_TYPES.PRICE);
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

  const fetchPredictionData = useCallback(
    async (symbol, days, type, signal, onProgress) => {
      const horizonRequest = normalizeHorizonRequest(days);
      const explicitDays = horizonRequest.requested_horizon;
      if (type !== FORECAST_TYPES.PRICE) {
        throw new Error(
          'The active volatility service provides price uncertainty only; direction forecasts are not available.',
        );
      }
      if (explicitDays == null) {
        throw new Error(
          'Volatility forecasting requires an explicit forecast horizon.',
        );
      }
      onProgress?.({ stage: 'volatility_snapshot' });
      const volatilityResult = await fetchVolatilityForecast(symbol, explicitDays, signal, {
        baseUrl: API_BASE,
      });
      onProgress?.({ stage: 'volatility_inference' });
      return volatilityResult;
    },
    []
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

      const cacheKey = forecastIdentity(activeTicker, activeDays, activeType);
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
              setLoadingStage(progress.message || stageLabels[progress.stage] || 'Evaluating forecast…');
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
    ]
  );

  const handleCancelRequest = useCallback(() => {
    abortActiveRequest();
    setIsLoading(false);
    setLoadingStage('');
    setTrainingProgress(null);
    addToast('info', 'Forecast request cancelled.');
  }, [abortActiveRequest, addToast]);

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

  return {
    ticker,
    setTicker: handleTickerChange,
    forecastDays,
    setForecastDays: handleForecastDaysChange,
    daysView,
    setDaysView,
    forecastType,
    setForecastType: handleForecastTypeChange,
    handleForecastTypeChange,
    handleTickerChange,
    handleForecastDaysChange,
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
    apiBase: API_BASE,
  };
}
