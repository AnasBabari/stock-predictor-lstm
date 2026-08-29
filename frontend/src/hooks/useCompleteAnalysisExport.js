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
  volatilityServing = true,
}) {
  const [isExportLoading, setIsExportLoading] = useState(false);
  const exportRequestIdRef = useRef(0);
  const exportAbortRef = useRef(null);

  const abortExport = useCallback(() => {
    exportRequestIdRef.current += 1;
    if (exportAbortRef.current) {
      exportAbortRef.current.abort();
      exportAbortRef.current = null;
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
    exportAbortRef.current = controller;

    const priceKey = forecastIdentity(tickerSymbol, forecastDays, FORECAST_TYPES.PRICE);
    const cachedPrice = forecastCacheRef.current.get(priceKey);
    const trendKey = volatilityServing
      ? null
      : forecastIdentity(tickerSymbol, forecastDays, FORECAST_TYPES.TREND);
    const cachedTrend = trendKey ? forecastCacheRef.current.get(trendKey) : null;

    const ensureForecast = async (type) => {
      const data = await fetchPredictionData(
        tickerSymbol,
        forecastDays,
        type,
        controller.signal,
        () => {}
      );
      assertForecastIdentity(data, tickerSymbol, forecastDays, type);
      const key = forecastIdentity(tickerSymbol, forecastDays, type);
      forecastCacheRef.current.set(key, data);
      return data;
    };

    try {
      setIsExportLoading(true);
      const priceData = cachedPrice || (await ensureForecast(FORECAST_TYPES.PRICE));
      if (volatilityServing) {
        const metadata = {
          ticker: tickerSymbol,
          generated_at: new Date().toISOString(),
          forecast_days: forecastDays,
          serving_mode: 'signed_global_volatility',
          metric_source: priceData.metrics?.metric_source,
          model_id: priceData.metadata?.model_version,
          snapshot_id: priceData.metadata?.snapshot_id,
          location_source: priceData.metadata?.engine?.location_source,
        };
        await exportCompleteAnalysis({
          priceData,
          directionData: null,
          metadata,
        });
        if (exportRequestIdRef.current === exportRequestId) {
          addToast('success', 'Volatility evidence exported as ZIP');
        }
        return;
      }
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
        exportAbortRef.current = null;
      }
    }
  }, [
    abortExport,
    addToast,
    fetchPredictionData,
    forecastCacheRef,
    forecastDays,
    volatilityServing,
    setErrorMsg,
    ticker,
  ]);

  return {
    isExportLoading,
    handleExportCompleteAnalysis,
  };
}
