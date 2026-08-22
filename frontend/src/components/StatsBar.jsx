import React, { useMemo } from 'react';

export default function StatsBar({ stockData, forecastType }) {
  const stats = useMemo(() => {
    if (!stockData) {
      return null;
    }

    if (forecastType === 'trend') {
      // Direction v2: one three-way call per origin with named probabilities.
      const direction = stockData.direction || '—';
      const probs = stockData.direction_probabilities || {};
      const confidence = probs[direction.toLowerCase()];

      return {
        ticker: stockData.ticker,
        forecastLabel: `Trend Forecast (${stockData.direction_horizon_days ?? stockData.forecast_days ?? '?'}d)`,
        lastClose: '—',
        forecast: direction,
        changeText: confidence != null ? `${(confidence * 100).toFixed(1)}%` : '—',
        trendText: direction,
        isUp: direction === 'Up',
        isFlat: direction === 'Neutral',
      };
    }

    if (!stockData.historical_prices || !stockData.predicted_prices) {
      return null;
    }

    const lastClose = stockData.historical_prices[stockData.historical_prices.length - 1];
    const forecast = stockData.predicted_prices[stockData.predicted_prices.length - 1];
    const isUp = forecast > lastClose;
    const isFlat = forecast === lastClose;
    const change = forecast - lastClose;
    const changePct = ((change / lastClose) * 100).toFixed(2);

    return {
      ticker: stockData.ticker,
      forecastLabel: 'Price Forecast',
      lastClose: `$${lastClose.toFixed(2)}`,
      forecast: `$${forecast.toFixed(2)}`,
      changeText: isFlat ? '0.00%' : `${isUp ? '+' : ''}${changePct}%`,
      trendText: isFlat ? 'Neutral' : isUp ? '▲ Bullish' : '▼ Bearish',
      isUp,
      isFlat,
    };
  }, [forecastType, stockData]);

  if (!stats) return null;

  const color = stats.isFlat ? 'var(--neutral)' : stats.isUp ? 'var(--bullish)' : 'var(--bearish)';

  return (
    <section id="statsBar" className="stats-bar">
      <div className="stat">
        <span className="stat-label">Ticker</span>
        <span className="stat-value mono">{stats.ticker}</span>
      </div>
      <div className="stat">
        <span className="stat-label">Forecast Type</span>
        <span className="stat-value mono">{stats.forecastLabel}</span>
      </div>
      <div className="stat">
        <span className="stat-label">Last Close</span>
        <span className="stat-value mono">{stats.lastClose}</span>
      </div>
      <div className="stat">
        <span className="stat-label">Forecast</span>
        <span className="stat-value mono">{stats.forecast}</span>
      </div>
      <div className="stat">
        <span className="stat-label">Change / Confidence</span>
        <span className="stat-value mono" style={{ color }}>
          {stats.changeText}
        </span>
      </div>
      <div className="stat">
        <span className="stat-label">Trend</span>
        <span className="stat-value" style={{ color }}>
          {stats.trendText}
        </span>
      </div>
    </section>
  );
}
