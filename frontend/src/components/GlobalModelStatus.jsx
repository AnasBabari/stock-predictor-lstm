import React from 'react';

/**
 * Renders global-model status when the prediction payload carries one.
 * Shows nothing unless `data.global_model` is present (feature-flagged
 * off by default via VITE_GLOBAL_MODEL_ENABLED).
 */
export default function GlobalModelStatus({ data }) {
  const gm = data?.global_model;
  if (!gm) return null;
  const statusLabel = {
    promoted: 'Global model forecast',
    blended_with_baseline: 'Blended toward persistence because measured edge is limited.',
    experimental_no_demonstrated_edge: 'Experimental model did not beat persistence.',
    insufficient_data: 'Insufficient comparable data for a certified learned forecast.',
  }[gm.status] || gm.status;

  return (
    <div className="global-model-status" role="status">
      <span className="gms-badge">{statusLabel}</span>
      {gm.alpha != null && gm.status !== 'promoted' && (
        <span className="gms-alpha"> α={gm.alpha.toFixed(2)}</span>
      )}
      {gm.interval && (
        <span className="gms-interval">
          {' '}90% CI [{gm.interval.low?.toFixed(2)}, {gm.interval.high?.toFixed(2)}]
        </span>
      )}
    </div>
  );
}
