import React from 'react';

export const PROTOCOL_V3_PROVENANCE = Object.freeze({
  protocolId: 'global-v3-cross-sectional',
  signalName: '3-Day Relative Cross-Sectional Rank (Short-Term Reversal)',
  candidateId: 'short_term_reversal_rank',
  horizonDays: 3,
  developmentStatus: 'Selected (Frozen)',
  prospectiveStatus: 'Pending Certification',
  universeSize: 212,
  validationFolds: 5,
  positiveFolds: 5,
  meanSpearmanIc: '+0.01957',
  bootstrapCi95: '[+0.0094, +0.0306]',
  hacTStat: '3.431 (p = 0.0126)',
  foldConsistency: '5 / 5 Positive (100%)',
  description:
    'Separate from local price level forecasting, Protocol V3 evaluates cross-sectional rank ordering across a 212-stock development universe using 5 purged expanding-window validation folds.',
  disclaimer:
    'Educational research signal only. Not financial advice, trading signals, or execution recommendations. Prospective certification begins after maturity cutoff.',
});

export default function ResearchSignalsPanel() {
  const p = PROTOCOL_V3_PROVENANCE;
  return (
    <section className="research-signals-panel glow-border" aria-labelledby="researchSignalsTitle">
      <div className="research-signals-header">
        <div className="research-signals-badge">RESEARCH SIGNAL · PROTOCOL V3</div>
        <h3 id="researchSignalsTitle">{p.signalName}</h3>
      </div>
      <p className="research-signals-desc">{p.description}</p>

      <div className="research-signals-grid">
        <div className="signal-stat-card">
          <span className="stat-label">Development Status</span>
          <span className="stat-value text-teal">{p.developmentStatus}</span>
        </div>
        <div className="signal-stat-card">
          <span className="stat-label">Prospective Status</span>
          <span className="stat-value text-amber">{p.prospectiveStatus}</span>
        </div>
        <div className="signal-stat-card">
          <span className="stat-label">Mean Spearman Rank IC</span>
          <span className="stat-value mono">{p.meanSpearmanIc}</span>
        </div>
        <div className="signal-stat-card">
          <span className="stat-label">95% Bootstrap CI</span>
          <span className="stat-value mono">{p.bootstrapCi95}</span>
        </div>
        <div className="signal-stat-card">
          <span className="stat-label">HAC t-statistic</span>
          <span className="stat-value mono">{p.hacTStat}</span>
        </div>
        <div className="signal-stat-card">
          <span className="stat-label">Fold Consistency</span>
          <span className="stat-value mono">{p.foldConsistency}</span>
        </div>
      </div>

      <div className="research-signals-footer">
        <span className="info-icon">ℹ</span>
        <span>{p.disclaimer}</span>
      </div>
    </section>
  );
}

