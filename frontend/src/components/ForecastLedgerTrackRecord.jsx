import React, { useEffect, useState } from 'react';

export default function ForecastLedgerTrackRecord({ ticker, horizon }) {
  const [ledgerData, setLedgerData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

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

  const track = ledgerData?.track_record || {};
  const entries = ledgerData?.entries || [];

  return (
    <section className="panel-card forecast-ledger-card" id="forecastLedgerSection" aria-label="Volatility Forecast Ledger">
      <div className="panel-header">
        <h3>
          <svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15" style={{ color: 'var(--teal)' }}>
            <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
            <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
          </svg>
          Forecast Ledger & Track Record
        </h3>
        <span className="badge badge-neutral">{ticker} {horizon ? `${horizon}d` : 'Multi-Horizon'}</span>
      </div>

      {loading ? (
        <div className="ledger-loading-state">Loading verified forecast ledger...</div>
      ) : error ? (
        <div className="ledger-error-state">Unable to load forecast ledger: {error}</div>
      ) : (
        <>
          <div className="ledger-kpi-grid">
            <div className="ledger-kpi-box">
              <span className="kpi-label">Scored Forecasts</span>
              <span className="kpi-value">{track.scored_forecasts ?? 0}</span>
              <span className="kpi-subtext">Out-of-sample settlements</span>
            </div>
            <div className="ledger-kpi-box">
              <span className="kpi-label">Mean MAE</span>
              <span className="kpi-value mono">
                {track.mean_mae != null ? `${(track.mean_mae * 100).toFixed(2)}%` : '—'}
              </span>
              <span className="kpi-subtext">Volatility absolute error</span>
            </div>
            <div className="ledger-kpi-box">
              <span className="kpi-label">Mean QLIKE</span>
              <span className="kpi-value mono text-teal">
                {track.mean_qlike != null ? track.mean_qlike.toFixed(4) : '—'}
              </span>
              <span className="kpi-subtext">Variance penalty metric</span>
            </div>
            <div className="ledger-kpi-box">
              <span className="kpi-label">Direction Hit Rate</span>
              <span className="kpi-value mono">
                {track.direction_accuracy_pct != null
                  ? `${track.direction_accuracy_pct.toFixed(1)}%`
                  : '—'}
              </span>
              <span className="kpi-subtext">Vol expansion / contraction</span>
            </div>
          </div>

          <div className="ledger-table-wrap">
            {entries.length === 0 ? (
              <p className="empty-state">No forecasts recorded in ledger yet.</p>
            ) : (
              <table className="ledger-table" aria-label="Historical forecast entries">
                <thead>
                  <tr>
                    <th>Origin Date</th>
                    <th>Horizon</th>
                    <th>Model</th>
                    <th>Predicted Vol</th>
                    <th>Realized Vol</th>
                    <th>Error (Δ)</th>
                    <th>QLIKE</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.slice(0, 15).map((entry) => {
                    const isScored = entry.status === 'scored';
                    const predVol = entry.predicted_volatility != null ? `${(entry.predicted_volatility * 100).toFixed(1)}%` : '—';
                    const actVol = entry.actual_realized_volatility != null ? `${(entry.actual_realized_volatility * 100).toFixed(1)}%` : '—';
                    const errVal = entry.forecast_error != null ? `${entry.forecast_error > 0 ? '+' : ''}${(entry.forecast_error * 100).toFixed(1)}%` : '—';
                    const qlikeVal = entry.qlike_loss != null ? entry.qlike_loss.toFixed(4) : '—';

                    return (
                      <tr key={entry.id || `${entry.forecast_date}-${entry.horizon}-${entry.model_name}`}>
                        <td className="mono">{entry.forecast_date}</td>
                        <td>{entry.horizon}d</td>
                        <td>
                          <span className="model-chip">{entry.model_name.replace('_', ' ')}</span>
                        </td>
                        <td className="mono font-medium">{predVol}</td>
                        <td className="mono text-teal">{actVol}</td>
                        <td className="mono" style={{ color: entry.abs_error != null && entry.abs_error < 0.03 ? 'var(--bullish)' : 'inherit' }}>
                          {errVal}
                        </td>
                        <td className="mono">{qlikeVal}</td>
                        <td>
                          <span className={`status-pill ${isScored ? 'status-scored' : 'status-pending'}`}>
                            {isScored ? 'Scored' : 'Pending'}
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
