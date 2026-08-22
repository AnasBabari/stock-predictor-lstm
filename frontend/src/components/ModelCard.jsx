import React from 'react';

const TARGET_DESCRIPTION = {
  price: 'Cumulative log return r(t,h) = ln(P(t+h) / P(t)); prices reconstructed from the latest close.',
  trend: 'Sign of the next future log return per step (up/down classification).',
};

function row(label, value) {
  return (
    <div className="model-card-row" key={label}>
      <span className="mc-label">{label}</span>
      <span className="mc-value">{value}</span>
    </div>
  );
}

/**
 * Lightweight methodology card. Facts only, derived from the prediction
 * payload; no marketing language. Renders nothing without a payload.
 */
export default function ModelCard({ data: stockData }) {
  const metadata = stockData?.metadata;
  if (!metadata) return null;
  const isTrend = stockData.direction != null;
  const engineRole = metadata.engine?.role || 'unknown';
  const snapshot = metadata.data_snapshot || {};
  const qualityStatus = snapshot.quality?.status;

  return (
    <details className="panel-card" id="modelCard">
      <summary className="panel-header">
        <h3>Model card &amp; methodology</h3>
      </summary>
      <div className="model-card-body">
        {row('Target', TARGET_DESCRIPTION[isTrend ? 'trend' : 'price'])}
        {row('Lookback window', `${metadata.window_size ?? '?'} sessions`)}
        {row('Forecast horizon', `${stockData.forecast_days ?? '?'} day(s)`)}
        {row(
          'Model',
          `${metadata.engine?.family ?? 'baseline'} (${engineRole})`
        )}
        {row('Feature count', String(metadata.feature_count ?? '?'))}
        {row('Schema version', `v${metadata.schema_version ?? '?'}`)}
        {row('Scaling', 'Robust median/IQR, fit on the training partition only')}
        {row(
          'Evaluation split',
          '80/20 chronological with a horizon-purged boundary; metrics are out-of-sample'
        )}
        {row('Metric source', metadata.metric_source || 'not reported')}
        {row(
          'Final refit caveat',
          'Displayed future forecast comes from a model refitted on all history; it is not independently validated.'
        )}
        {row(
          'Data adjustment',
          snapshot.adjusted_prices ? 'Split/dividend-adjusted closes (auto_adjust)' : 'Adjustment policy not reported'
        )}
        {row(
          'Data quality',
          qualityStatus === 'clean'
            ? 'Clean'
            : qualityStatus === 'annotated'
              ? 'Annotated warnings — see snapshot metadata'
              : 'Not reported'
        )}
        {row('Snapshot', metadata.snapshot_id ? `${String(metadata.snapshot_id).slice(0, 12)}…` : 'n/a')}
        <p className="mc-limitations">
          Limitations: daily OHLCV carries limited predictive signal; simple
          baselines are often competitive; nothing here is financial advice.
        </p>
      </div>
    </details>
  );
}
