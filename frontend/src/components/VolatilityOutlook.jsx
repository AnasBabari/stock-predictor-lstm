import React from 'react';
import { OUTLOOK_HORIZONS, useVolatilityOutlook } from '../hooks/useVolatilityOutlook';

const TRADING_SESSIONS_PER_YEAR = 252;

export function formatAnnualized(value) {
  if (value == null || value === '') return '—';
  if (!Number.isFinite(Number(value))) return '—';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

/**
 * One-sigma expected range around a reference price over a session window,
 * from an annualised volatility figure. Pure horizon math, no direction.
 */
export function expectedRange(referencePrice, annualized, sessions) {
  const price = Number(referencePrice);
  const annual = Number(annualized);
  const window = Number(sessions);
  if (!Number.isFinite(price) || price <= 0 || !Number.isFinite(annual) || annual <= 0
    || !Number.isFinite(window) || window <= 0) {
    return null;
  }
  const move = (annual / Math.sqrt(TRADING_SESSIONS_PER_YEAR)) * Math.sqrt(window);
  return { low: price * Math.exp(-move), high: price * Math.exp(move) };
}

export function annualisedFromResponse(entry) {
  const value = Number(entry?.forecast?.expected_annualized_volatility ?? entry?.forecast?.predicted_volatility);
  return Number.isFinite(value) && value > 0 ? value : null;
}

function RiskPill({ level }) {
  const key = String(level || 'Unknown').toLowerCase();
  const label = key === 'unknown' ? 'Unknown' : String(level);
  return <span className={`risk-pill risk-${key}`}>{label}</span>;
}

function money(value, currencySymbol) {
  if (!Number.isFinite(Number(value))) return '—';
  const symbol = currencySymbol === 'p' || currencySymbol === 'GBp' ? 'p' : (currencySymbol || '$');
  const decimals = symbol === 'p' ? 1 : 2;
  return `${symbol}${Number(value).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
}

export default function VolatilityOutlook({ ticker, currencySymbol = '$', currentPrice = null, priceEstimate = null }) {
  const { outlook, loading, error, retry } = useVolatilityOutlook(ticker);
  const symbol = String(ticker || '').toUpperCase();
  if (!symbol) return null;

  const rows = outlook
    ? OUTLOOK_HORIZONS.map((horizon) => ({ horizon, entry: outlook.byHorizon[horizon] })).filter((row) => row.entry)
    : [];
  const first = rows[0]?.entry || null;
  const asOf = first?.evidence?.data_as_of || first?.asOf || null;

  const combined = (() => {
    const five = outlook?.byHorizon?.[5];
    const annual = five ? annualisedFromResponse(five) : null;
    if (!priceEstimate || !Number.isFinite(Number(priceEstimate.price)) || !annual || !Number.isFinite(Number(currentPrice))) {
      return null;
    }
    const band = expectedRange(currentPrice, annual, 5);
    if (!band) return null;
    return { ...band, annual, changePct: Number(priceEstimate.changePct) };
  })();

  return (
    <section className="panel outlook-panel" aria-label={`${symbol} volatility outlook`}>
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{symbol} · data through {asOf || '—'}</p>
          <h2>Volatility Outlook</h2>
        </div>
      </div>

      {loading && <div className="loading-text" role="status">Loading volatility outlook…</div>}
      {!loading && error && (
        <div role="alert">
          <p className="empty-copy">{error}</p>
          <button type="button" className="retry-button" onClick={retry}>Retry outlook</button>
        </div>
      )}
      {!loading && !error && rows.length > 0 && (
        <>
          <table className="outlook-table">
            <thead>
              <tr>
                <th scope="col">Period</th>
                <th scope="col">Forecast</th>
                <th scope="col">Risk level</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ horizon, entry }) => (
                <tr key={horizon}>
                  <td>Next {horizon} sessions</td>
                  <td className="mono">{formatAnnualized(annualisedFromResponse(entry))} annualised</td>
                  <td><RiskPill level={entry?.evidence?.risk_level} /></td>
                </tr>
              ))}
            </tbody>
          </table>

          {combined && (
            <div className="combined-outlook" aria-label="Combined seven-day outlook">
              <h3>7-Day Outlook</h3>
              <dl>
                <div>
                  <dt>Estimated price</dt>
                  <dd className="mono">{money(priceEstimate.price, currencySymbol)}</dd>
                </div>
                <div>
                  <dt>Model direction</dt>
                  <dd className={`mono ${combined.changePct >= 0 ? 'up' : 'down'}`}>
                    {combined.changePct >= 0 ? '+' : ''}{combined.changePct.toFixed(1)}%
                  </dd>
                </div>
                <div>
                  <dt>Expected volatility</dt>
                  <dd><RiskPill level={first?.evidence?.risk_level} /></dd>
                </div>
                <div>
                  <dt>Expected range</dt>
                  <dd className="mono">{money(combined.low, currencySymbol)}–{money(combined.high, currencySymbol)}</dd>
                </div>
              </dl>
              <p className="method-note">
                Range shows where the current price would typically move over five sessions;
                it is not a prediction that the estimate itself will be reached.
              </p>
            </div>
          )}

          <p className="method-note">
            <strong>Model:</strong> Enhanced volatility forecast. Uses recent price behaviour,
            trading ranges and volatility structure. Historically outperformed the previous
            rolling-volatility model across the held-out test panel.
          </p>
          <details className="forecast-details">
            <summary>How this forecast works</summary>
            <p>
              The forecast starts from recent market volatility and adjusts it using
              patterns learned across hundreds of stocks. It estimates how much prices
              typically move — not which direction they move in.
            </p>
            <p>
              Each horizon is evaluated independently against realised market volatility
              on data the model never trained on. Past outperformance does not guarantee
              future results.
            </p>
          </details>
          <details className="forecast-details">
            <summary>Advanced · model diagnostics</summary>
            <dl className="diagnostics-list">
              {rows.map(({ horizon, entry }) => {
                const annual = annualisedFromResponse(entry);
                const trailing = Number(entry?.evidence?.trailing_annualized_volatility_60d);
                const adjustment = annual != null && trailing > 0 ? annual / trailing - 1 : null;
                return (
                  <div key={horizon}>
                    <dt>{horizon}-session forecast</dt>
                    <dd className="mono">{formatAnnualized(annual)}</dd>
                    <dt>Recent market volatility (60-session)</dt>
                    <dd className="mono">{formatAnnualized(trailing)}</dd>
                    <dt>Model adjustment</dt>
                    <dd className="mono">
                      {adjustment == null ? '—' : `${adjustment >= 0 ? '+' : ''}${(adjustment * 100).toFixed(1)}%`}
                    </dd>
                    <dt>Last updated</dt>
                    <dd className="mono">{entry?.evidence?.data_as_of || asOf || '—'}</dd>
                  </div>
                );
              })}
            </dl>
          </details>
        </>
      )}
    </section>
  );
}
