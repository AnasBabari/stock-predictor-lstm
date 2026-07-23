import React, { useMemo } from 'react';

export default function StatsBar({ stockData }) {
  const stats = useMemo(() => {
    if (!stockData || !stockData.historical_prices || !stockData.predicted_prices) {
      return null;
    }

    const lastClose = stockData.historical_prices[stockData.historical_prices.length - 1];
    const forecast = stockData.predicted_prices[stockData.predicted_prices.length - 1];
    const isUp = forecast > lastClose;
    const change = forecast - lastClose;
    const changePct = ((change / lastClose) * 100).toFixed(2);

    return {
      ticker: stockData.ticker,
      lastClose: `$${lastClose.toFixed(2)}`,
      forecast: `$${forecast.toFixed(2)}`,
      changeText: `${isUp ? '+' : ''}${changePct}%`,
      trendText: isUp ? '▲ Bullish' : '▼ Bearish',
      isUp,
    };
  }, [stockData]);

  if (!stats) return null;

  const color = stats.isUp ? 'var(--bullish)' : 'var(--bearish)';

  return (
    <section id="statsBar" className="stats-bar">
      <div className="stat">
        <span className="stat-label">Ticker</span>
        <span className="stat-value mono">{stats.ticker}</span>
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
        <span className="stat-label">Change</span>
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
