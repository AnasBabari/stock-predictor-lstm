import React, { useCallback } from 'react';
import { exportPriceCSV, exportTrendCSV } from '../utils/exportService';

export default function ForecastChartActions({
  chartRef,
  stockData,
  forecastType,
  onAddWatchlist,
  onToast,
  onExportCompleteAnalysis,
}) {
  const handleExportPng = useCallback(() => {
    if (!chartRef?.current || !stockData) return;
    const link = document.createElement('a');
    link.download = `${stockData.ticker}_forecast.png`;
    link.href = chartRef.current.toBase64Image();
    link.click();
    onToast('success', 'Chart exported as PNG');
  }, [chartRef, onToast, stockData]);

  const handleExportCsv = useCallback(async () => {
    if (!stockData) return;

    if (forecastType === 'trend') {
      await exportTrendCSV(stockData);
      onToast('success', 'Trend forecast exported as CSV');
      return;
    }

    await exportPriceCSV(stockData);
    onToast('success', 'Price forecast exported as CSV');
  }, [forecastType, onToast, stockData]);

  const showPriceActions = Boolean(stockData && forecastType === 'price');
  const showTrendActions = Boolean(stockData && forecastType === 'trend');

  return (
    <div className="chart-actions-row">
      {(showPriceActions || showTrendActions) && (
        <div className="chart-actions">
          {showPriceActions && (
            <button type="button" className="action-chip" onClick={handleExportPng} title="Export chart as PNG">
              <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                <path
                  fillRule="evenodd"
                  d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z"
                />
              </svg>
              PNG
            </button>
          )}
          <button type="button" className="action-chip" onClick={handleExportCsv} title="Export forecast data as CSV">
            <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
              <path
                fillRule="evenodd"
                d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z"
              />
            </svg>
            CSV
          </button>
          {showPriceActions && (
            <button
              type="button"
              className="action-chip accent"
              onClick={() => onAddWatchlist(stockData)}
              title="Add to watchlist"
            >
              <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              Watch
            </button>
          )}
        </div>
      )}
      <div className="chart-actions">
        <button
          type="button"
          className="action-chip accent"
          onClick={onExportCompleteAnalysis}
          title="Export both price and trend forecasts as a ZIP bundle"
        >
          Complete Analysis
        </button>
      </div>
    </div>
  );
}
