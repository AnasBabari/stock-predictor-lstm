import React from 'react';

export default function MetricsCard({ stockData }) {
  if (!stockData || !stockData.metrics) return null;

  const m = stockData.metrics;

  const rmse = m.rmse != null ? m.rmse.toFixed(2) : '—';
  const mae = m.mae != null ? m.mae.toFixed(2) : '—';
  const r2 = m.r2 != null ? m.r2.toFixed(4) : '—';
  const mape = m.mape != null ? `${m.mape.toFixed(2)}%` : '—';
  const da = m.directional_accuracy != null ? `${(m.directional_accuracy * 100).toFixed(1)}%` : '—';

  return (
    <section id="metricsCard" className="metrics-card">
      <div className="metric">
        <span className="metric-icon" title="Root Mean Squared Error — lower is better">
          <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
            />
          </svg>
        </span>
        <div className="metric-content">
          <span className="metric-label">RMSE</span>
          <span className="metric-value mono">{rmse}</span>
        </div>
      </div>
      <div className="metric-divider"></div>
      <div className="metric">
        <span className="metric-icon" title="Mean Absolute Error">
          <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
            />
          </svg>
        </span>
        <div className="metric-content">
          <span className="metric-label">MAE</span>
          <span className="metric-value mono">{mae}</span>
        </div>
      </div>
      <div className="metric-divider"></div>
      <div className="metric">
        <span className="metric-icon" title="R Squared">
          <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
            />
          </svg>
        </span>
        <div className="metric-content">
          <span className="metric-label">R²</span>
          <span className="metric-value mono">{r2}</span>
        </div>
      </div>
      <div className="metric-divider"></div>
      <div className="metric">
        <span className="metric-icon" title="Mean Absolute Percentage Error">
          <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
            />
          </svg>
        </span>
        <div className="metric-content">
          <span className="metric-label">MAPE</span>
          <span className="metric-value mono">{mape}</span>
        </div>
      </div>
      <div className="metric-divider"></div>
      <div className="metric">
        <span className="metric-icon" title="Directional Accuracy (Test Set)">
          <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z"
            />
          </svg>
        </span>
        <div className="metric-content">
          <span className="metric-label">Dir. Acc</span>
          <span className="metric-value mono">{da}</span>
        </div>
      </div>
    </section>
  );
}
