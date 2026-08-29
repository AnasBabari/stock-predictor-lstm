import React from 'react';

const TARGET_DESCRIPTION = {
  price: 'Cumulative log return r(t,h) = ln(P(t+h) / P(t)); prices reconstructed from the latest close.',
  trend: 'One three-way call per forecast origin: sign of the CUMULATIVE h-day log return with a volatility-aware neutral band (Down/Neutral/Up softmax).',
  volatility: 'Conditional cumulative return variance; the displayed location is the unchanged-close persistence baseline.',
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
  const isVolatility = metadata.engine?.certified_head === 'volatility' || stockData.volatility_cone != null;
  const engineRole = metadata.engine?.role || 'unknown';
  const snapshot = metadata.data_snapshot || {};
  const qualityStatus = snapshot.quality?.status;

  return (
    <details className="panel-card" id="modelCard">
      <summary className="panel-header">
        <h3>Model card &amp; methodology</h3>
      </summary>
      <div className="model-card-body">
        {row('Target', TARGET_DESCRIPTION[isVolatility ? 'volatility' : isTrend ? 'trend' : 'price'])}
        {row('Lookback window', `${metadata.window_size ?? '?'} sessions`)}
        {row('Forecast horizon', `${stockData.forecast_days ?? '?'} day(s)`)}
        {row(
          'Model',
          `${metadata.engine?.family ?? 'baseline'} (${engineRole})`
        )}
        {row(
          'Model state',
          metadata.engine?.execution_mode === 'server_artifact_loaded'
            ? 'Signed server artifact loaded'
            : metadata.engine?.role === 'baseline_fallback'
              ? 'Explicit baseline fallback'
              : metadata.engine?.cache_status || 'Signed server evaluation'
        )}
        {row('Feature count', String(metadata.feature_count ?? '?'))}
        {row('Schema version', `${metadata.schema_version ?? '?'}`)}
        {row('Scaling', isVolatility ? 'Frozen train-only scaler from the signed release' : 'Robust median/IQR, fit on the training partition only')}
        {row(
          'Evaluation split',
          isVolatility ? 'Locked purged walk-forward plus temporal and ticker-transfer holdouts' : '80/20 chronological with a horizon-purged boundary; metrics are out-of-sample'
        )}
        {row('Metric source', metadata.metric_source || 'not reported')}
        {row(
          'Location caveat',
          isVolatility ? 'The center line is an explicit matched persistence baseline; only the volatility cone is learned and certified.' : 'Displayed future forecast comes from a model refitted on all history; it is not independently validated.'
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
