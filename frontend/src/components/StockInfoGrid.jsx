import React, { useMemo } from 'react';

function formatLargeNum(n) {
  if (n == null) return '—';
  if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toLocaleString();
}

export default function StockInfoGrid({ info }) {
  const cards = useMemo(() => {
    if (!info) return [];

    return [
      { label: 'Company', value: info.name || '—', mono: false },
      { label: 'Sector', value: info.sector || '—', mono: false },
      { label: 'Market Cap', value: formatLargeNum(info.marketCap), mono: true },
      {
        label: 'P/E Ratio',
        value: info.peRatio ? info.peRatio.toFixed(2) : '—',
        mono: true,
      },
      {
        label: '52W High',
        value: info.fiftyTwoWeekHigh ? `$${info.fiftyTwoWeekHigh.toFixed(2)}` : '—',
        mono: true,
      },
      {
        label: '52W Low',
        value: info.fiftyTwoWeekLow ? `$${info.fiftyTwoWeekLow.toFixed(2)}` : '—',
        mono: true,
      },
      { label: 'Avg Volume', value: formatLargeNum(info.avgVolume), mono: true },
      {
        label: 'Prev Close',
        value: info.previousClose ? `$${info.previousClose.toFixed(2)}` : '—',
        mono: true,
      },
    ];
  }, [info]);

  if (!info) return null;

  return (
    <section id="stockInfo" className="stock-info">
      <div className="section-label">
        <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
          <path d="M4 4a2 2 0 00-2 2v1h16V6a2 2 0 00-2-2H4z" />
          <path
            fillRule="evenodd"
            d="M18 9H2v5a2 2 0 002 2h12a2 2 0 002-2V9zM4 13a1 1 0 011-1h1a1 1 0 110 2H5a1 1 0 01-1-1zm5-1a1 1 0 100 2h1a1 1 0 100-2H9z"
          />
        </svg>
        Company Overview
      </div>
      <div className="info-grid">
        {cards.map((c, i) => (
          <div
            key={c.label}
            className="info-card"
            style={{ animation: `fadeUp 0.3s ${0.05 * i}s ease both` }}
          >
            <span className="info-card-label">{c.label}</span>
            <span className={`info-card-value${c.mono ? ' mono' : ''}`}>{c.value}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
