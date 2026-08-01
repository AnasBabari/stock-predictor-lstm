import React from 'react';

function MetricItem({ iconTitle, label, value }) {
  return (
    <div className="metric">
      <span className="metric-icon" title={iconTitle}>
        <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
          />
        </svg>
      </span>
      <div className="metric-content">
        <span className="metric-label">{label}</span>
        <span className="metric-value mono">{value}</span>
      </div>
    </div>
  );
}

export default function MetricsCard({ stockData, forecastType }) {
  if (!stockData || !stockData.metrics) return null;

  const m = stockData.metrics;
  const engine = stockData.metadata?.engine;
  const isTrend = forecastType === 'trend';
  const modeLabel = isTrend ? 'Trend Forecast Metrics' : 'Price Forecast Metrics';
  const metricSource =
    m.metric_source === 'walk_forward_out_of_fold'
      ? 'Walk-forward out-of-fold evaluation'
      : m.metric_source === 'browser_purged_holdout'
        ? 'Browser purged holdout evaluation'
        : 'Baseline definition';

  const priceMetrics = [
    { label: 'RMSE', value: m.rmse != null ? m.rmse.toFixed(2) : '—', title: 'Root Mean Squared Error — lower is better' },
    { label: 'MAE', value: m.mae != null ? m.mae.toFixed(2) : '—', title: 'Mean Absolute Error' },
    { label: 'MASE', value: m.mase != null ? m.mase.toFixed(3) : '—', title: 'Mean Absolute Scaled Error; below 1 beats the in-sample naïve scale' },
    { label: 'vs persistence', value: m.relative_rmse != null ? `${m.relative_rmse.toFixed(3)}×` : '—', title: 'RMSE divided by a no-change persistence forecast; below 1 is better' },
    { label: 'R²', value: m.r2 != null ? m.r2.toFixed(4) : '—', title: 'R Squared' },
    { label: 'MAPE', value: m.mape != null ? `${m.mape.toFixed(2)}%` : '—', title: 'Mean Absolute Percentage Error' },
    {
      label: 'Dir. Acc',
      value: m.directional_accuracy != null ? `${(m.directional_accuracy * 100).toFixed(1)}%` : '—',
      title: 'Directional accuracy from walk-forward out-of-fold predictions',
    },
  ];

  const trendMetrics = [
    { label: 'Precision', value: m.precision != null ? m.precision.toFixed(4) : '—', title: 'Walk-forward out-of-fold precision' },
    { label: 'Recall', value: m.recall != null ? m.recall.toFixed(4) : '—', title: 'Walk-forward out-of-fold recall' },
    { label: 'F1', value: m.f1 != null ? m.f1.toFixed(4) : '—', title: 'Walk-forward out-of-fold F1 score' },
    { label: 'Balanced acc.', value: m.balanced_accuracy != null ? `${(m.balanced_accuracy * 100).toFixed(1)}%` : '—', title: 'Average recall across up and down classes' },
    { label: 'Brier score', value: m.brier_score != null ? m.brier_score.toFixed(4) : '—', title: 'Probability calibration error; lower is better' },
    {
      label: 'Naive Baseline',
      value: m.naive_baseline != null ? `${(m.naive_baseline * 100).toFixed(1)}%` : '—',
      title: 'Majority-class baseline accuracy',
    },
  ];

  const metrics = isTrend ? trendMetrics : priceMetrics;
  const engineLabel = engine?.family
    ? engine.family.replaceAll('_', ' ')
    : 'Prepared model';
  const evidenceLabel = engine?.baseline_fallback
    ? 'Baseline fallback'
    : engine?.role === 'learned_candidate'
      ? 'Learned candidate'
      : engine?.role === 'browser_learned'
        ? 'Learned locally'
        : 'Baseline fallback';

  return (
    <section
      id="metricsCard"
      className={`metrics-card${engine?.baseline_fallback ? ' metrics-card--baseline' : ''}`}
    >
      <div className="metric">
        <div className="metric-content">
          <span className="metric-label">Mode</span>
          <span className="metric-value">{modeLabel}</span>
        </div>
      </div>
      <div className="metric-divider"></div>
      <MetricItem
        iconTitle={
          engine?.baseline_fallback
            ? 'No learned candidate currently has qualifying serving evidence.'
            : 'Forecast engine selected by the serving pipeline.'
        }
        label="Active engine"
        value={`${engineLabel} · ${evidenceLabel}`}
      />
      <div className="metric-divider"></div>
      <MetricItem iconTitle={metricSource} label="Metric source" value={metricSource} />
      {metrics.map((metric, index) => (
        <React.Fragment key={metric.label}>
          <div className="metric-divider"></div>
          <MetricItem iconTitle={metric.title} label={metric.label} value={metric.value} />
        </React.Fragment>
      ))}
    </section>
  );
}
