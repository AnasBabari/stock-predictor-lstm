import React from 'react';

function MetricItem({ iconTitle, label, value }) {
  return (
    <div className="metric">
      <span className="metric-icon" title={iconTitle} aria-label={iconTitle}>ⓘ</span>
      <div className="metric-content">
        <span className="metric-label">{label}</span>
        <span className="metric-value mono">{value}</span>
      </div>
    </div>
  );
}

function HorizonTable({ metrics }) {
  const perHorizon = metrics.per_horizon;
  if (!Array.isArray(perHorizon) || !perHorizon.length) return null;
  const selected = Number(metrics.horizon) || null;
  return (
    <div className="metrics-horizons">
      <div className="metric-divider"></div>
      <MetricItem
        iconTitle="Metrics are reported per horizon and pooled. The row for the selected forecast horizon is highlighted."
        label="Metrics by horizon"
        value={`Pooled: ${perHorizon.length} horizons`}
      />
      <table className="horizon-metrics-table">
        <thead>
          <tr>
            <th>Horizon</th>
            <th>MAE</th>
            <th>RMSE</th>
            <th>vs persist.</th>
            <th>Direction</th>
            <th>Rows</th>
          </tr>
        </thead>
        <tbody>
          {perHorizon.map((entry) => {
            const isSelected = Number(entry.horizon) === selected;
            const beats = entry.relative_rmse != null && entry.relative_rmse < 1;
            return (
              <tr key={entry.horizon} className={isSelected ? 'horizon-row--selected' : ''}>
                <td>{entry.horizon}d{isSelected ? ' ✓' : ''}</td>
                <td className="mono">{entry.mae?.toFixed(4)}</td>
                <td className="mono">{entry.rmse?.toFixed(4)}</td>
                <td className="mono">{entry.relative_rmse == null ? '—' : `${entry.relative_rmse.toFixed(3)}×${beats ? '' : ' ✗'}`}</td>
                <td className="mono">{entry.directional_accuracy == null ? '—' : `${(entry.directional_accuracy * 100).toFixed(0)}%`}</td>
                <td className="mono">{entry.rows}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PromotionNotice({ stockData, forecastType }) {
  const engine = stockData.metadata?.engine;
  const promotion = stockData.metadata?.promotion;
  const isTrend = forecastType === 'trend';
  const volatilityRejected = !isTrend && Array.isArray(promotion?.reasons) &&
    promotion.reasons.some((reason) => reason.includes('volatility'));
  const baselineReason = engine?.baseline_fallback || promotion?.promoted === false;
  if (!baselineReason) return null;
  const reasons = Array.isArray(promotion?.reasons) ? promotion.reasons : [];
  return (
    <div className="metrics-warning" role="status">
      {volatilityRejected
        ? 'The learned forecast exceeded its historically observed volatility range. Showing the persistence baseline.'
        : isTrend
          ? 'This learned direction model was not promoted and did not beat the majority-class baseline on untouched out-of-sample evaluation. Showing the majority-class baseline. The learned result remains visible for research only.'
          : 'This learned model was not promoted and did not beat the persistence benchmark on untouched out-of-sample evaluation. Showing the persistence baseline. The learned result remains visible for research only.'}
      {reasons.length > 0 && (
        <ul className="promotion-reasons">
          {reasons.map((reason) => <li key={reason}>{reason}</li>)}
        </ul>
      )}
    </div>
  );
}

export default function MetricsCard({ stockData, forecastType }) {
  if (!stockData?.metrics) return null;
  const m = stockData.metrics;
  const engine = stockData.metadata?.engine;
  const isTrend = forecastType === 'trend';
  const metricSource = m.metric_source === 'browser_walk_forward_out_of_fold'
    ? 'Five-fold expanding purged walk-forward out-of-fold evaluation'
    : m.metric_source === 'browser_purged_holdout'
      ? 'Single untouched purged holdout'
      : m.metric_source === 'walk_forward_out_of_fold'
        ? 'Walk-forward out-of-fold evaluation'
        : 'Baseline definition';

  const priceMetrics = [
    ['RMSE', m.rmse?.toFixed(4), 'Root Mean Squared Error — lower is better'],
    ['MAE', m.mae?.toFixed(4), 'Mean Absolute Error — lower is better'],
    ['RMSE vs persistence', m.relative_rmse == null ? null : `${m.relative_rmse.toFixed(3)}×`, 'Below 1 beats a no-change forecast'],
    ['MAE vs persistence', m.relative_mae == null ? null : `${m.relative_mae.toFixed(3)}×`, 'Below 1 beats a no-change forecast'],
    ['Directional accuracy', m.directional_accuracy == null ? null : `${(m.directional_accuracy * 100).toFixed(1)}%`, 'Share of horizons where predicted return sign matched the realized return'],
    ['Dollar RMSE', m.dollar_rmse == null ? null : `$${m.dollar_rmse.toFixed(2)}`, 'Root Mean Squared Error on reconstructed prices — lower is better'],
    ['Dollar MAE', m.dollar_mae == null ? null : `$${m.dollar_mae.toFixed(2)}`, 'Mean Absolute Error on reconstructed prices — lower is better'],
    ['R²', m.r2?.toFixed(4), 'R Squared'],
    ['MAPE', m.mape == null ? null : `${m.mape.toFixed(2)}%`, 'Mean Absolute Percentage Error'],
  ];
  const trendMetrics = [
    ['Accuracy', m.accuracy == null ? null : `${(m.accuracy * 100).toFixed(1)}%`, 'Fraction of correct direction labels'],
    ['Balanced acc.', m.balanced_accuracy == null ? null : `${(m.balanced_accuracy * 100).toFixed(1)}%`, 'Average recall across up and down classes'],
    ['Precision', m.precision?.toFixed(4), 'Positive-class precision'],
    ['Recall', m.recall?.toFixed(4), 'Positive-class recall'],
    ['F1', m.f1?.toFixed(4), 'Harmonic mean of precision and recall'],
    ['Brier score', m.brier_score?.toFixed(4), 'Probability calibration error — lower is better'],
    ['Majority baseline', m.naive_baseline == null ? null : `${(m.naive_baseline * 100).toFixed(1)}%`, 'Accuracy from always selecting the majority class'],
  ];
  const metrics = (isTrend ? trendMetrics : priceMetrics).map(([label, value, title]) => ({
    label, value: value ?? '—', title,
  }));
  const engineLabel = stockData.metadata?.server_pretrained || engine?.role === 'server_pretrained'
    ? stockData.metadata.model_name || 'Server-Pretrained Model'
    : engine?.family ? engine.family.replaceAll('_', ' ') : 'Prepared model';

  const localStatus = engine?.baseline_fallback
    ? isTrend ? 'Baseline fallback — majority class displayed' : 'Baseline fallback — persistence displayed'
    : stockData.metadata?.server_pretrained || engine?.role === 'server_pretrained'
      ? 'Trained offline on server'
      : engine?.role === 'learned_candidate'
        ? 'Learned candidate'
        : engine?.execution_mode === 'browser_artifact_loaded'
          ? 'Cached on this device'
          : engine?.execution_mode === 'browser_trained'
            ? 'Trained in this browser'
            : 'Learned locally';
  const underperforms = !isTrend && (
    (m.relative_rmse != null && m.relative_rmse >= 1) ||
    (m.relative_mae != null && m.relative_mae >= 1)
  );

  return (
    <section id="metricsCard" className={`metrics-card${engine?.baseline_fallback ? ' metrics-card--baseline' : ''}`}>
      <MetricItem iconTitle="Selected forecast type" label="Mode" value={`${isTrend ? 'Trend' : 'Price'} Forecast Metrics`} />
      <div className="metric-divider"></div>
      <MetricItem iconTitle="Forecast engine and local cache status" label="Active engine" value={`${engineLabel} · ${localStatus}`} />
      <div className="metric-divider"></div>
      <MetricItem iconTitle={metricSource} label="Metric source" value={metricSource} />
      {metrics.map((metric) => (
        <React.Fragment key={metric.label}>
          <div className="metric-divider"></div>
          <MetricItem iconTitle={metric.title} label={metric.label} value={metric.value} />
        </React.Fragment>
      ))}
      {!isTrend && <HorizonTable metrics={m} />}
      <PromotionNotice stockData={stockData} forecastType={forecastType} />
      {underperforms && (
        <div className="metrics-warning" role="status">
          This learned model did not beat the no-change persistence benchmark on its evaluation data.
        </div>
      )}
    </section>
  );
}
