import React from 'react';
import { OUTLOOK_HORIZONS, useVolatilityOutlook } from '../hooks/useVolatilityOutlook';

const TRADING_SESSIONS_PER_YEAR = 252;

export function formatAnnualized(value) {
  if (value == null || value === '') return '—';
  if (!Number.isFinite(Number(value))) return '—';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

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

function horizonPlainDescription(riskLevel, annual, trailing) {
  const level = String(riskLevel || '').toLowerCase();
  if (level.includes('elevated') || (annual != null && trailing != null && annual > trailing * 1.15)) {
    return 'Elevated price movement likely compared to recent history';
  }
  if (level.includes('subdued') || level.includes('low') || (annual != null && trailing != null && annual < trailing * 0.85)) {
    return 'Subdued price movement expected relative to typical conditions';
  }
  return 'Typical price fluctuations expected based on recent patterns';
}

function money(value, currencySymbol) {
  if (value == null || value === '' || !Number.isFinite(Number(value))) return '—';
  const isPence = currencySymbol === 'p' || currencySymbol === 'GBp';
  const symbol = isPence ? 'p' : (currencySymbol || '$');
  const decimals = isPence ? 1 : 2;
  const numeric = Number(value);
  const formatted = Math.abs(numeric).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  if (isPence) {
    return `${numeric < 0 ? '-' : ''}${formatted}p`;
  }
  return `${numeric < 0 ? '-' : ''}${symbol}${formatted}`;
}

export default function VolatilityOutlook({
  ticker,
  currencySymbol = '$',
  currentPrice = null,
  priceEstimate = null,
}) {
  const { outlook, loading, error, retry } = useVolatilityOutlook(ticker);
  const symbol = String(ticker || '').toUpperCase();
  if (!symbol) return null;

  const combined = (() => {
    const five = outlook?.byHorizon?.[5];
    const annual = five ? annualisedFromResponse(five) : null;
    if (!priceEstimate || priceEstimate.price == null || !Number.isFinite(Number(priceEstimate.price)) || !annual || currentPrice == null || !Number.isFinite(Number(currentPrice))) {
      return null;
    }
    const band = expectedRange(currentPrice, annual, 5);
    if (!band) return null;
    return { ...band, annual, changePct: Number(priceEstimate.changePct) };
  })();

  if (loading) {
    return (
      <div className="volatility-status-row" role="status">
        <span className="loading-dot" aria-hidden="true" />
        <span>Loading volatility outlook…</span>
      </div>
    );
  }

  if (error || !outlook) {
    return (
      <div className="volatility-status-row error" role="alert">
        <span>{error || 'Volatility outlook is unavailable for this stock right now.'}</span>
        <button type="button" className="retry-action-btn" onClick={retry}>
          Retry outlook
        </button>
      </div>
    );
  }

  const rows = OUTLOOK_HORIZONS
    .map((horizon) => ({ horizon, entry: outlook.byHorizon?.[horizon] }))
    .filter((row) => row.entry);

  if (rows.length === 0) {
    return (
      <div className="volatility-status-row" role="alert">
        <span>No volatility horizons are available for {symbol}.</span>
        <button type="button" className="retry-action-btn" onClick={retry}>
          Retry
        </button>
      </div>
    );
  }

  const first = rows[0]?.entry || null;
  const asOf = first?.evidence?.data_as_of || first?.asOf || null;
  const modelName = first?.evidence?.model_name || first?.forecast?.model || 'Enhanced statistical model';

  return (
    <section className="panel outlook-panel" aria-label={`${symbol} volatility outlook`}>
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{symbol} · data through {asOf || '—'}</p>
          <h2>Volatility Outlook</h2>
        </div>
      </div>

      <div className="horizon-rows-list">
        {rows.map(({ horizon, entry }) => {
          const annual = annualisedFromResponse(entry);
          const trailing = Number(entry?.evidence?.trailing_annualized_volatility_60d);
          const riskLevel = entry?.evidence?.risk_level || 'Normal';
          const plainDesc = horizonPlainDescription(riskLevel, annual, trailing);
          const range = currentPrice && annual ? expectedRange(currentPrice, annual, horizon) : null;
          const adjustment = annual != null && trailing > 0 ? (annual / trailing - 1) : null;

          return (
            <article key={horizon} className="horizon-compact-row">
              <div className="horizon-main-line">
                <div className="horizon-badge-col">
                  <strong className="horizon-name">{horizon} sessions</strong>
                  <RiskPill level={riskLevel} />
                </div>
                <div className="horizon-desc-col">
                  <span className="horizon-plain-desc">{plainDesc}</span>
                  <span className="horizon-annual-val mono">
                    {formatAnnualized(annual)} annualised
                  </span>
                </div>
              </div>

              <details className="horizon-expandable-details">
                <summary>View numerical breakdown</summary>
                <div className="horizon-details-content">
                  <div className="detail-item">
                    <span>Forecast volatility:</span>
                    <strong className="mono">{formatAnnualized(annual)}</strong>
                  </div>
                  <div className="detail-item">
                    <span>Trailing 60-day volatility:</span>
                    <strong className="mono">{formatAnnualized(trailing)}</strong>
                  </div>
                  {adjustment != null && (
                    <div className="detail-item">
                      <span>Adjustment vs trailing:</span>
                      <strong className="mono">
                        {adjustment >= 0 ? '+' : ''}{(adjustment * 100).toFixed(1)}%
                      </strong>
                    </div>
                  )}
                  {!combined && range && (
                    <div className="detail-item">
                      <span>Expected price range (±1σ):</span>
                      <strong className="mono">
                        {money(range.low, currencySymbol)} – {money(range.high, currencySymbol)}
                      </strong>
                    </div>
                  )}
                  <div className="detail-item">
                    <span>Data as of:</span>
                    <strong className="mono">{entry?.evidence?.data_as_of || asOf || '—'}</strong>
                  </div>
                </div>
              </details>
            </article>
          );
        })}
      </div>

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
        </div>
      )}

      <details className="about-outlook-details">
        <summary>About this outlook</summary>
        <div className="about-outlook-body">
          <p>
            <strong>Model:</strong> Enhanced volatility forecast. Evaluated against realised market volatility on held-out test panel.
          </p>
          <p>
            <strong>Interpretation:</strong> Volatility estimates describe the expected magnitude of price swings over each period, not whether prices will rise or fall.
          </p>
          <p>
            <strong>Fallback Policy:</strong> If machine learning estimates encounter missing data, a validated statistical baseline is automatically used.
          </p>
          <p>
            Past performance does not guarantee future results.
          </p>
        </div>
      </details>
    </section>
  );
}
