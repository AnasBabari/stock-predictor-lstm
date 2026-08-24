import React, { useState } from 'react';
import ValidationBadge from './ValidationBadge';
import { formatMultiplier, formatNumber, formatPercent, formatPrice } from '../utils/formatting';

export default function MetricsCard({ stockData, forecastType, onSwitchHorizon }) {
  const [activeTab, setActiveTab] = useState('forecast'); // forecast | evaluation | model | research

  if (!stockData || !stockData.metrics) return null;

  const m = stockData.metrics;
  const val = stockData.validation || {};
  const meta = stockData.metadata || {};
  const engine = meta.engine || {};
  const isTrend = forecastType === 'trend';
  const perHorizon = Array.isArray(m.per_horizon) ? m.per_horizon : [];
  const selectedHorizon = Number(val.selected_horizon || m.horizon || stockData.forecast_days || 7);
  const bestValidatedHorizon = val.best_validated_horizon;

  const latestClose = Number(stockData.historical_prices?.at(-1));
  const forecastPrice = Number(stockData.predicted_prices?.at(-1));
  const priceChangePct = Number.isFinite(latestClose) && Number.isFinite(forecastPrice) && latestClose > 0
    ? ((forecastPrice - latestClose) / latestClose) * 100
    : null;

  return (
    <section id="metricsCard" className="metrics-dashboard-card glow-border" aria-label="Forecast Metrics Dashboard">
      <h3 className="sr-only">{isTrend ? 'Trend Forecast Metrics' : 'Price Forecast Metrics'}</h3>
      <div className="metrics-tabs-header" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'forecast'}
          className={`tab-btn ${activeTab === 'forecast' ? 'active' : ''}`}
          onClick={() => setActiveTab('forecast')}
        >
          Forecast
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'evaluation'}
          className={`tab-btn ${activeTab === 'evaluation' ? 'active' : ''}`}
          onClick={() => setActiveTab('evaluation')}
        >
          Evaluation
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'model'}
          className={`tab-btn ${activeTab === 'model' ? 'active' : ''}`}
          onClick={() => setActiveTab('model')}
        >
          Model Specs
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'research'}
          className={`tab-btn ${activeTab === 'research' ? 'active' : ''}`}
          onClick={() => setActiveTab('research')}
        >
          Research Details
        </button>
      </div>

      <div className="metrics-tab-content">
        {/* Tab 1: Forecast Overview */}
        {activeTab === 'forecast' && (
          <div className="tab-pane">
            <div className="forecast-summary-grid">
              <div className="summary-card">
                <span className="card-label">Validation Status</span>
                <div className="card-value-wrap">
                  <ValidationBadge state={val.state || (val.promoted ? 'promoted' : 'experimental')} />
                </div>
                <span className="card-subtext">
                  {val.promoted
                    ? 'Validated against persistence on held-out data.'
                    : 'Model forecast shown; holdout validation gates were not met.'}
                </span>
              </div>

              {!isTrend ? (
                <>
                  <div className="summary-card">
                    <span className="card-label">Predicted Endpoint</span>
                    <span className="card-value mono text-teal">{formatPrice(forecastPrice)}</span>
                    <span className="card-subtext">Last close: {formatPrice(latestClose)}</span>
                  </div>
                  <div className="summary-card">
                    <span className="card-label">Expected Return</span>
                    <span className="card-value mono">{formatPercent(priceChangePct)}</span>
                    <span className="card-subtext">{selectedHorizon}-day horizon</span>
                  </div>
                </>
              ) : (
                <div className="summary-card">
                  <span className="card-label">Predicted Direction</span>
                  <span className="card-value text-teal">{stockData.direction || 'Neutral'}</span>
                  <span className="card-subtext">3-way probability distribution</span>
                </div>
              )}

              <div className="summary-card">
                <span className="card-label">Active Model Engine</span>
                <span className="card-value font-medium">{engine.role === 'server_pretrained' ? 'server_pretrained' : engine.family ? engine.family.replaceAll('_', ' ') : 'Balanced LSTM'}</span>
                <span className="card-subtext">Compute: {engine.role === 'server_pretrained' ? 'server-pretrained' : engine.backend?.toUpperCase() || 'WEBGPU'}</span>
              </div>
            </div>

            {/* Smart Switch Notification when requested horizon is experimental but another is promoted */}
            {!val.promoted && bestValidatedHorizon && bestValidatedHorizon !== selectedHorizon && onSwitchHorizon && (
              <div className="validated-switch-banner" role="status">
                <div className="switch-banner-text">
                  <strong>{selectedHorizon}-Day forecast is experimental.</strong> A {bestValidatedHorizon}-Day candidate passed all validation gates.
                </div>
                <button
                  type="button"
                  className="switch-horizon-btn"
                  onClick={() => onSwitchHorizon(bestValidatedHorizon)}
                >
                  Switch to {bestValidatedHorizon}-Day
                </button>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Evaluation & Per-Horizon Table */}
        {activeTab === 'evaluation' && (
          <div className="tab-pane">
            <div className="kpi-cards-grid">
              {!isTrend ? (
                <>
                  <div className="kpi-card">
                    <span className="kpi-label">RMSE vs Persistence</span>
                    <span className="kpi-value mono">{formatMultiplier(m.relative_rmse)}</span>
                    <span className="kpi-note">&lt; 1.000× beats persistence</span>
                  </div>
                  <div className="kpi-card">
                    <span className="kpi-label">MAE vs Persistence</span>
                    <span className="kpi-value mono">{formatMultiplier(m.relative_mae)}</span>
                    <span className="kpi-note">&le; 1.000× beats persistence</span>
                  </div>
                  <div className="kpi-card">
                    <span className="kpi-label">Directional Accuracy</span>
                    <span className="kpi-value mono">{formatPercent(m.directional_accuracy ? m.directional_accuracy * 100 : null, { includePlus: false })}</span>
                    <span className="kpi-note">Sign agreement on holdout</span>
                  </div>
                  <div className="kpi-card">
                    <span className="kpi-label">Return R²</span>
                    <span className="kpi-value mono">{formatNumber(m.r2, { decimals: 4 })}</span>
                    <span className="kpi-note">Log return variance explained</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="kpi-card">
                    <span className="kpi-label">Macro Balanced Acc.</span>
                    <span className="kpi-value mono">{formatPercent(m.macro_balanced_accuracy ? m.macro_balanced_accuracy * 100 : null, { includePlus: false })}</span>
                    <span className="kpi-note">Mean per-class recall</span>
                  </div>
                  <div className="kpi-card">
                    <span className="kpi-label">Brier Skill Score</span>
                    <span className="kpi-value mono">{formatPercent(m.brier_skill ? m.brier_skill * 100 : null)}</span>
                    <span className="kpi-note">Above 0% beats base rate</span>
                  </div>
                  <div className="kpi-card">
                    <span className="kpi-label">Macro F1 Score</span>
                    <span className="kpi-value mono">{formatNumber(m.macro_f1, { decimals: 4 })}</span>
                    <span className="kpi-note">Precision/recall balance</span>
                  </div>
                  <div className="kpi-card">
                    <span className="kpi-label">Log Loss</span>
                    <span className="kpi-value mono">{formatNumber(m.log_loss, { decimals: 4 })}</span>
                    <span className="kpi-note">Cross-entropy penalty</span>
                  </div>
                </>
              )}
            </div>

            {!isTrend && perHorizon.length > 0 && (
              <div className="horizon-table-wrap">
                <h4>Per-Horizon Evaluation Matrix</h4>
                <table className="horizon-metrics-table">
                  <thead>
                    <tr>
                      <th>Horizon</th>
                      <th>Rel. RMSE</th>
                      <th>Rel. MAE</th>
                      <th>Direction</th>
                      <th>Rows</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {perHorizon.map((entry) => {
                      const hNum = Number(entry.horizon);
                      const isSelected = hNum === selectedHorizon;
                      const isPromoted = entry.relative_rmse != null && entry.relative_rmse < 1.0 && (entry.relative_mae == null || entry.relative_mae <= 1.0);
                      const rowState = isPromoted ? 'promoted' : (entry.relative_rmse < 1.0 ? 'candidate' : 'experimental');
                      return (
                        <tr key={entry.horizon} className={isSelected ? 'horizon-row--selected' : ''}>
                          <td className="font-semibold">{entry.horizon}d{isSelected ? ' (Selected)' : ''}</td>
                          <td className="mono">{formatMultiplier(entry.relative_rmse)}</td>
                          <td className="mono">{formatMultiplier(entry.relative_mae)}</td>
                          <td className="mono">{formatPercent(entry.directional_accuracy ? entry.directional_accuracy * 100 : null, { includePlus: false })}</td>
                          <td className="mono">{entry.rows || '—'}</td>
                          <td>
                            <ValidationBadge state={rowState} />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Model & Hardware Specs */}
        {activeTab === 'model' && (
          <div className="tab-pane">
            <div className="specs-grid">
              <div className="spec-item">
                <span className="spec-label">Model Architecture</span>
                <span className="spec-value">{meta.architecture || 'balanced_lstm_in_browser'}</span>
              </div>
              <div className="spec-item">
                <span className="spec-label">Feature Schema</span>
                <span className="spec-value mono">Stationary v4 ({meta.feature_count || 28} features)</span>
              </div>
              <div className="spec-item">
                <span className="spec-label">Lookback Window</span>
                <span className="spec-value">{meta.window_size || 60} trading sessions</span>
              </div>
              <div className="spec-item">
                <span className="spec-label">Training Duration</span>
                <span className="spec-value mono">{meta.training_duration_ms ? `${meta.training_duration_ms} ms` : '—'}</span>
              </div>
              <div className="spec-item">
                <span className="spec-label">Selected Epochs</span>
                <span className="spec-value mono">{meta.selected_epochs ?? '—'}</span>
              </div>
              <div className="spec-item">
                <span className="spec-label">Compute Backend</span>
                <span className="spec-value mono">{engine.backend?.toUpperCase() || 'WEBGPU'}</span>
              </div>
              <div className="spec-item full-width">
                <span className="spec-label">Data Snapshot Hash</span>
                <span className="spec-value mono text-xs">{meta.snapshot_id || '—'}</span>
              </div>
            </div>
          </div>
        )}

        {/* Tab 4: Research & Gate Checks */}
        {activeTab === 'research' && (
          <div className="tab-pane">
            <div className="research-pane-content">
              <h4>Holdout Evaluation Protocol</h4>
              <p className="text-sm text-secondary mb-3">
                Evaluation was conducted on an untouched chronological holdout partition with strict horizon purging to eliminate lookahead leakage.
              </p>

              {Array.isArray(val.reasons) && val.reasons.length > 0 && (
                <div className="gate-reasons-box">
                  <h5>Promotion Gate Findings:</h5>
                  <ul>
                    {val.reasons.map((reason, idx) => (
                      <li key={idx}>{reason}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="metric-rows-summary">
                <div className="metric-summary-row">
                  <span>Metric Source:</span>
                  <span className="mono">{m.metric_source || 'browser_purged_holdout'}</span>
                </div>
                <div className="metric-summary-row">
                  <span>Evaluated Rows:</span>
                  <span className="mono">{m.evaluation_rows || '—'} sessions</span>
                </div>
                <div className="metric-summary-row">
                  <span>Dollar RMSE / MAE:</span>
                  <span className="mono">{formatPrice(m.dollar_rmse)} / {formatPrice(m.dollar_mae)}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
