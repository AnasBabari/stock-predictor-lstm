import React from 'react';
import { PROTOCOL_V3_PROVENANCE } from '../data/protocolV3Provenance';

export { PROTOCOL_V3_PROVENANCE } from '../data/protocolV3Provenance';

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
          <span className="stat-label">Raw one-sided HAC p</span>
          <span className="stat-value mono">{p.rawHacP}</span>
        </div>
        <div className="signal-stat-card">
          <span className="stat-label">Holm-adjusted p ({p.hypothesisCount} tests)</span>
          <span className="stat-value mono">{p.holmAdjustedP}</span>
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
