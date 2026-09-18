import React, { useCallback, useEffect, useRef, useState } from 'react';

export default function ForecastLedgerTrackRecord({ ticker, horizon, defaultOpen = false }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const [ledgerData, setLedgerData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('all'); // 'all' | 'live' | 'historical_replay'
  const fetchedTickerRef = useRef(null);

  const fetchLedger = useCallback(async (sym) => {
    if (!sym) return;
    setLoading(true);
    setError(null);
    try {
      const queryHorizon = horizon ? `&horizon=${encodeURIComponent(horizon)}` : '';
      const url = `/api/v1/volatility/ledger?ticker=${encodeURIComponent(sym)}${queryHorizon}`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Failed to load forecast ledger (${res.status})`);
      }
      const data = await res.json();
      setLedgerData(data);
      fetchedTickerRef.current = sym;
    } catch (err) {
      setError(err.message || 'Unable to load past forecasts.');
    } finally {
      setLoading(false);
    }
  }, [horizon]);

  // Lazy fetch: only fetch when section is opened, and reuse cache for same ticker
  useEffect(() => {
    if (isOpen && ticker && fetchedTickerRef.current !== ticker) {
      fetchLedger(ticker);
    }
  }, [isOpen, ticker, fetchLedger]);

  // Reset cache if ticker changes
  useEffect(() => {
    if (ticker && fetchedTickerRef.current && fetchedTickerRef.current !== ticker) {
      setLedgerData(null);
      fetchedTickerRef.current = null;
      if (isOpen) {
        fetchLedger(ticker);
      }
    }
  }, [ticker, isOpen, fetchLedger]);

  if (!ticker) return null;

  const liveTrack = ledgerData?.live_track_record || {};
  const replayTrack = ledgerData?.replay_track_record || {};
  const allEntries = ledgerData?.entries || [];

  const filteredEntries = allEntries.filter((entry) => {
    if (activeTab === 'live') return entry.record_source === 'live';
    if (activeTab === 'historical_replay') return entry.record_source === 'historical_replay';
    return true;
  });

  const displayTrack = activeTab === 'historical_replay' ? replayTrack : liveTrack;

  return (
    <details
      className="panel-card forecast-ledger-card ledger-expandable-section"
      id="forecastLedgerSection"
      aria-label="Past price-movement forecasts"
      open={isOpen}
      onToggle={(e) => setIsOpen(e.currentTarget.open)}
    >
      <summary className="ledger-summary-header">
        <span className="ledger-summary-title">
          <svg
            viewBox="0 0 20 20"
            fill="currentColor"
            width="15"
            height="15"
            style={{ color: 'var(--teal)' }}
            aria-hidden="true"
          >
            <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
            <path
              fillRule="evenodd"
              d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z"
              clipRule="evenodd"
            />
          </svg>
          Past price-movement forecasts
        </span>
        <span className="badge badge-neutral">
          {ticker} {horizon ? `${horizon} ${Number(horizon) === 1 ? 'market day' : 'market days'}` : 'All time periods'}
        </span>
      </summary>

      <div className="ledger-content-wrap">
        <p className="method-note">
          This history evaluates past volatility estimates against realised market movements. It is strictly separate from the price model evaluation.
        </p>

        {loading ? (
          <div className="ledger-loading-state" role="status">Loading past forecasts…</div>
        ) : error ? (
          <div className="ledger-error-state" role="alert">Past forecasts are unavailable right now.</div>
        ) : allEntries.length === 0 ? (
          <p className="ledger-empty-sentence">No past recorded forecasts for this ticker yet.</p>
        ) : (
          <>
            <div className="ledger-tab-group">
              <button
                className={`ledger-tab-btn ${activeTab === 'all' ? 'active' : ''}`}
                onClick={() => setActiveTab('all')}
                type="button"
              >
                All ({allEntries.length})
              </button>
              <button
                className={`ledger-tab-btn ${activeTab === 'live' ? 'active' : ''}`}
                onClick={() => setActiveTab('live')}
                type="button"
              >
                Live ({liveTrack.total_forecasts ?? 0})
              </button>
              <button
                className={`ledger-tab-btn ${activeTab === 'historical_replay' ? 'active' : ''}`}
                onClick={() => setActiveTab('historical_replay')}
                type="button"
              >
                Historical tests ({replayTrack.total_forecasts ?? 0})
              </button>
            </div>

            <div className="ledger-kpi-grid">
              <div className="ledger-kpi-box">
                <span className="kpi-label">
                  {activeTab === 'historical_replay' ? 'Historical tests checked' : 'Live forecasts checked'}
                </span>
                <span className="kpi-value">{displayTrack.scored_forecasts ?? 0}</span>
                <span className="kpi-subtext">
                  {activeTab === 'historical_replay'
                    ? 'Tests using past data'
                    : 'Recorded before the results were known'}
                </span>
              </div>
              <div className="ledger-kpi-box">
                <span className="kpi-label">Average error</span>
                <span className="kpi-value mono">
                  {displayTrack.mean_mae != null ? `${(displayTrack.mean_mae * 100).toFixed(2)}%` : '—'}
                </span>
                <span className="kpi-subtext">Mistakes in estimated price swings</span>
              </div>
              <div className="ledger-kpi-box">
                <span className="kpi-label">Error score</span>
                <span className="kpi-value mono text-teal">
                  {displayTrack.mean_qlike != null ? displayTrack.mean_qlike.toFixed(4) : '—'}
                </span>
                <span className="kpi-subtext">Lower is better</span>
              </div>
              <div className="ledger-kpi-box">
                <span className="kpi-label">Bigger or smaller swings correct</span>
                <span className="kpi-value mono">
                  {displayTrack.direction_accuracy_pct != null
                    ? `${displayTrack.direction_accuracy_pct.toFixed(1)}%`
                    : '—'}
                </span>
                <span className="kpi-subtext">Not the direction of the stock price</span>
              </div>
            </div>

            <div className="ledger-table-wrap">
              <table className="ledger-table" aria-label="Historical forecast entries">
                <thead>
                  <tr>
                    <th>Forecast date</th>
                    <th>Trading days ahead</th>
                    <th>Model</th>
                    <th>Source</th>
                    <th>Estimated price swings</th>
                    <th>Actual price swings</th>
                    <th>Difference</th>
                    <th>Error score</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEntries.slice(0, 20).map((entry) => {
                    const isScored = entry.status === 'scored';
                    const isLive = entry.record_source === 'live';
                    const predVol =
                      entry.predicted_volatility != null
                        ? `${(entry.predicted_volatility * 100).toFixed(1)}%`
                        : '—';
                    const actVol =
                      entry.actual_realized_volatility != null
                        ? `${(entry.actual_realized_volatility * 100).toFixed(1)}%`
                        : '—';
                    const errVal =
                      entry.forecast_error != null
                        ? `${entry.forecast_error > 0 ? '+' : ''}${(
                            entry.forecast_error * 100
                          ).toFixed(1)}%`
                        : '—';
                    const qlikeVal =
                      entry.qlike_loss != null ? entry.qlike_loss.toFixed(4) : '—';

                    return (
                      <tr
                        key={
                          entry.id ||
                          `${entry.forecast_date}-${entry.horizon}-${entry.model_name}-${entry.record_source}`
                        }
                      >
                        <td className="mono">{entry.forecast_date}</td>
                        <td>{entry.horizon} {Number(entry.horizon) === 1 ? 'market day' : 'market days'}</td>
                        <td>
                          <span className="model-chip">{entry.model_name.replace('_', ' ')}</span>
                        </td>
                        <td>
                          <span
                            className={`source-chip ${
                              isLive ? 'source-chip--live' : 'source-chip--replay'
                            }`}
                          >
                            {isLive ? 'Live' : 'Historical test'}
                          </span>
                        </td>
                        <td className="mono font-medium">{predVol}</td>
                        <td className="mono text-teal">{actVol}</td>
                        <td
                          className="mono"
                          style={{
                            color:
                              entry.abs_error != null && entry.abs_error < 0.03
                                ? 'var(--bullish)'
                                : 'inherit',
                          }}
                        >
                          {errVal}
                        </td>
                        <td className="mono">{qlikeVal}</td>
                        <td>
                          <span
                            className={`status-pill ${
                              isScored ? 'status-scored' : 'status-pending'
                            }`}
                          >
                            {isScored ? 'Checked' : 'Waiting for results'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </details>
  );
}
