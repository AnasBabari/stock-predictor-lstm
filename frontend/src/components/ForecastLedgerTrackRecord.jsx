import React, { useEffect, useState } from 'react';

export default function ForecastLedgerTrackRecord({ ticker, horizon }) {
  const [ledgerData, setLedgerData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('all'); // 'all' | 'live' | 'historical_replay'

  useEffect(() => {
    let isMounted = true;
    if (!ticker) return;

    setLoading(true);
    setError(null);

    const queryHorizon = horizon ? `&horizon=${encodeURIComponent(horizon)}` : '';
    const url = `/api/v1/volatility/ledger?ticker=${encodeURIComponent(ticker)}${queryHorizon}`;

    fetch(url)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to load forecast ledger (${res.status})`);
        }
        return res.json();
      })
      .then((data) => {
        if (isMounted) {
          setLedgerData(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [ticker, horizon]);

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
    <section
      className="panel-card forecast-ledger-card"
      id="forecastLedgerSection"
      aria-label="Past price-movement forecasts"
    >
      <div className="panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <h3>
            <svg
              viewBox="0 0 20 20"
              fill="currentColor"
              width="15"
              height="15"
              style={{ color: 'var(--teal)' }}
            >
              <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
              <path
                fillRule="evenodd"
                d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z"
                clipRule="evenodd"
              />
            </svg>
            Past price-movement forecasts
          </h3>
          <span className="badge badge-neutral">
            {ticker} {horizon ? `${horizon} ${Number(horizon) === 1 ? 'market day' : 'market days'}` : 'All time periods'}
          </span>
        </div>
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
      </div>

      <p className="method-note">This history measures the size of price swings, not future share prices. It is separate from the seven-day price estimate above.</p>
      {loading ? (
        <div className="ledger-loading-state">Loading past forecasts…</div>
      ) : error ? (
        <div className="ledger-error-state">Past forecasts are unavailable right now. Please try again later.</div>
      ) : (
        <>
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
            {filteredEntries.length === 0 ? (
              <div className="ledger-empty-container">
                <div className="ledger-empty-icon-box">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    width="20"
                    height="20"
                    aria-hidden="true"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
                    />
                  </svg>
                </div>
                <strong className="ledger-empty-heading">No results to show yet</strong>
                <p className="empty-state">No forecasts recorded in this view yet.</p>
                <span className="ledger-empty-detail">
                  Saved forecasts are checked once enough market days have passed. Historical tests are kept separate from forecasts made before the outcome was known.
                </span>
              </div>
            ) : (
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
            )}
          </div>
        </>
      )}
    </section>
  );
}
