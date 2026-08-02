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
    ['RMSE', m.rmse?.toFixed(2), 'Root Mean Squared Error — lower is better'],
    ['MSE', m.mse?.toFixed(2), 'Mean Squared Error — lower is better'],
    ['MAE', m.mae?.toFixed(2), 'Mean Absolute Error — lower is better'],
    ['RMSE vs persistence', m.relative_rmse == null ? null : `${m.relative_rmse.toFixed(3)}×`, 'Below 1 beats a no-change forecast'],
    ['MAE vs persistence', m.relative_mae == null ? null : `${m.relative_mae.toFixed(3)}×`, 'Below 1 beats a no-change forecast'],
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
  const engineLabel = engine?.family ? engine.family.replaceAll('_', ' ') : 'Prepared model';
  const localStatus = engine?.baseline_fallback
    ? 'Baseline fallback'
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
      {underperforms && (
        <div className="metrics-warning" role="status">
          This learned model did not beat the no-change persistence benchmark on its evaluation data.
        </div>
      )}
    </section>
  );
}
