import React, { useMemo } from 'react';
import ValidationBadge from './ValidationBadge';
import { formatPercent, formatPrice } from '../utils/formatting';

export default function StatsBar({ stockData, forecastType }) {
  const stats = useMemo(() => {
    if (!stockData) {
      return null;
    }

    const valState = stockData.validation?.state || (stockData.validation?.promoted ? 'promoted' : 'experimental');
    const isVolatility = stockData.metadata?.engine?.certified_head === 'volatility'
      || stockData.volatility_cone != null;

    if (forecastType === 'trend') {
      const direction = stockData.direction || '—';
      const probs = stockData.direction_probabilities || {};
      const confidence = probs[direction.toLowerCase()];

      return {
        ticker: stockData.ticker,
        forecastLabel: `Trend (${stockData.direction_horizon_days ?? stockData.forecast_days ?? '?'}d)`,
        lastClose: '—',
        forecast: direction,
        changeText: confidence != null ? `${(confidence * 100).toFixed(1)}%` : '—',
        trendText: direction,
        valState,
        isUp: direction === 'Up',
        isFlat: direction === 'Neutral',
      };
    }

    if (!stockData.historical_prices) {
      return null;
    }

    const lastClose = Number(stockData.historical_prices.at(-1));

    if (isVolatility) {
      const low = Number(stockData.volatility_cone?.p05?.at(-1));
      const high = Number(stockData.volatility_cone?.p95?.at(-1));
      const annualizedVolatility = Number(stockData.forecast?.expected_annualized_volatility);
      return {
        ticker: stockData.ticker,
        forecastLabel: `Volatility cone (${stockData.forecast_days || 7}d)`,
        lastClose: formatPrice(lastClose),
        forecast: Number.isFinite(low) && Number.isFinite(high)
          ? `${formatPrice(low)} – ${formatPrice(high)}`
          : '—',
        changeText: Number.isFinite(annualizedVolatility)
          ? formatPercent(annualizedVolatility * 100, { includePlus: false })
          : '—',
        trendText: 'Volatility certified',
        valState,
        isUp: false,
        isFlat: true,
        volatilityOnly: true,
      };
    }

    if (!stockData.predicted_prices) return null;
    const forecast = Number(stockData.predicted_prices.at(-1));
    const isUp = forecast > lastClose;
    const isFlat = Math.abs(forecast - lastClose) < 1e-6;
    const change = forecast - lastClose;
    const changePct = lastClose > 0 ? (change / lastClose) * 100 : 0;

    return {
      ticker: stockData.ticker,
      forecastLabel: `Price (${stockData.forecast_days || 7}d)`,
      lastClose: formatPrice(lastClose),
      forecast: formatPrice(forecast),
      changeText: formatPercent(changePct),
      trendText: isFlat ? 'Neutral' : isUp ? '▲ Bullish' : '▼ Bearish',
      valState,
      isUp,
      isFlat,
    };
  }, [forecastType, stockData]);

  if (!stats) return null;

  const color = stats.isFlat ? 'var(--neutral)' : stats.isUp ? 'var(--bullish)' : 'var(--bearish)';

  return (
    <section id="statsBar" className="stats-bar" aria-label="Forecast summary statistics">
      <div className="stat">
        <span className="stat-label">Ticker</span>
        <span className="stat-value mono">{stats.ticker}</span>
      </div>
      <div className="stat">
        <span className="stat-label">Forecast Horizon</span>
        <span className="stat-value mono">{stats.forecastLabel}</span>
      </div>
      <div className="stat">
        <span className="stat-label">Last Close</span>
        <span className="stat-value mono">{stats.lastClose}</span>
      </div>
      <div className="stat">
        <span className="stat-label">{stats.volatilityOnly ? '90% Forecast Range' : 'Predicted Endpoint'}</span>
        <span className="stat-value mono text-teal">{stats.forecast}</span>
      </div>
      <div className="stat">
        <span className="stat-label">{stats.volatilityOnly ? 'Annualized Volatility' : 'Expected Return'}</span>
        <span className="stat-value mono" style={{ color }}>
          {stats.changeText}
        </span>
      </div>
      <div className="stat">
        <span className="stat-label">Validation</span>
        <span className="stat-value">
          <ValidationBadge state={stats.valState} />
        </span>
      </div>
    </section>
  );
}
