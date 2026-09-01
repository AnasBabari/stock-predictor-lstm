/**
 * Immutable display projection of the already-frozen Protocol V3 record.
 * Updating these values requires updating the underlying certification
 * record; the UI must not recompute or reinterpret them.
 */
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
  hacTStat: '3.431',
  rawHacP: '0.000300',
  holmAdjustedP: '0.01261',
  hypothesisCount: 42,
  foldConsistency: '5 / 5 Positive (100%)',
  description:
    'Separate from local price level forecasting, Protocol V3 evaluates cross-sectional rank ordering across a 212-stock development universe using 5 purged expanding-window validation folds.',
  disclaimer:
    'Educational research signal only. Not financial advice, trading signals, or execution recommendations. Prospective certification begins after maturity cutoff.',
});
