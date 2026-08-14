import { useCallback, useEffect, useRef, useState } from 'react';
import { exportCompleteAnalysis } from '../utils/exportService';
import {
  FORECAST_TYPES,
  assertForecastIdentity,
  forecastIdentity,
} from './useForecast';

export function useCompleteAnalysisExport({
  addToast,
  setErrorMsg,
  fetchPredictionData,
  forecastCacheRef,
  ticker,
  forecastDays,
  trainingProfile,
}) {
  const [isExportLoading, setIsExportLoading] = useState(false);
  const exportAbortControllerRef = useRef(null);
  const exportRequestIdRef = useRef(0);

  const abortExport = useCallback(() => {
    exportRequestIdRef.current += 1;
    if (exportAbortControllerRef.current) {
      exportAbortControllerRef.current.abort();
      exportAbortControllerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return abortExport;
  }, [abortExport]);

  const handleExportCompleteAnalysis = useCallback(async () => {
    const tickerSymbol = ticker.trim().toUpperCase();
    if (!tickerSymbol) {
      addToast('error', 'Please select or enter a ticker symbol first.');
      return;
    }

    abortExport();
    const exportRequestId = exportRequestIdRef.current;
    const controller = new AbortController();
    exportAbortControllerRef.current = controller;

    const priceKey = forecastIdentity(tickerSymbol, forecastDays, FORECAST_TYPES.PRICE, trainingProfile);
    const trendKey = forecastIdentity(tickerSymbol, forecastDays, FORECAST_TYPES.TREND, trainingProfile);
    const cachedPrice = forecastCacheRef.current.get(priceKey);
    const cachedTrend = forecastCacheRef.current.get(trendKey);

    const ensureForecast = async (type) => {
      const data = await fetchPredictionData(
        tickerSymbol,
        forecastDays,
        type,
        controller.signal,
        () => {}
      );
      assertForecastIdentity(data, tickerSymbol, forecastDays, type);
      const key = forecastIdentity(tickerSymbol, forecastDays, type, trainingProfile);
      forecastCacheRef.current.set(key, data);
      return data;
    };

    try {
      setIsExportLoading(true);
      const priceData = cachedPrice || (await ensureForecast(FORECAST_TYPES.PRICE));
      const trendData = cachedTrend || (await ensureForecast(FORECAST_TYPES.TREND));

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
        price_metric_source: priceData.metrics?.metric_source,
        direction_metric_source: trendData.metrics?.metric_source,
      };

      await exportCompleteAnalysis({
        priceData,
        directionData: trendData,
        metadata,
      });
      if (exportRequestIdRef.current === exportRequestId) {
        addToast('success', 'Complete analysis exported as ZIP');
      }
    } catch (err) {
      if (err?.name === 'AbortError') return;
      if (exportRequestIdRef.current === exportRequestId) {
        const msg = err.message || 'Failed to export complete analysis.';
        setErrorMsg(msg);
        addToast('error', msg);
      }
    } finally {
      if (exportRequestIdRef.current === exportRequestId) {
        setIsExportLoading(false);
        exportAbortControllerRef.current = null;
      }
    }
  }, [
    abortExport,
    addToast,
    fetchPredictionData,
    forecastCacheRef,
    forecastDays,
    setErrorMsg,
    ticker,
    trainingProfile,
  ]);

  return {
    isExportLoading,
    handleExportCompleteAnalysis,
  };
}
