function formatHistoryDate(createdAt) {
  if (!createdAt) return '';
  const parsed = new Date(createdAt);
  if (isNaN(parsed.getTime())) return '';
  return parsed.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
  });
}

function formatPrice(value) {
  return typeof value === 'number' ? `$${value.toFixed(2)}` : '—';
}

export default function PredictionHistory({ items, onSelectTicker, onClearAll }) {
  return (
    <section className="panel-card" id="historySection">
      <div className="panel-header">
        <h3>
          <svg
            viewBox="0 0 20 20"
            fill="currentColor"
            width="14"
            height="14"
            style={{ color: 'var(--accent)' }}
          >
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z"
            />
          </svg>
          Prediction History
        </h3>
        <button type="button" className="text-btn" onClick={onClearAll} title="Clear all">
          Clear
        </button>
      </div>
      <div id="historyItems" className="panel-items">
        {items.length === 0 ? (
          <p className="empty-state">
            No predictions yet.
            <br />
            Search a ticker to get started.
          </p>
        ) : (
          items.map((h, index) => {
            const hasChange = typeof h.changePercent === 'number' && Number.isFinite(h.changePercent);
            const isUp = hasChange && h.changePercent >= 0;
            const color = !hasChange ? 'var(--muted)' : isUp ? 'var(--bullish)' : 'var(--bearish)';
            const arrow = !hasChange ? '•' : isUp ? '▲' : '▼';
            const dateStr = formatHistoryDate(h.createdAt);
            const detail =
              h.forecastType === 'trend'
                ? `P(up) ${typeof h.predictedValue === 'number' ? (h.predictedValue * 100).toFixed(0) : '—'}% · ${h.horizon ?? '?'}d`
                : `${formatPrice(h.lastClose)} → ${formatPrice(h.predictedValue)} · ${h.horizon ?? '?'}d`;

            return (
              <button
                type="button"
                key={`${h.ticker}-${h.createdAt ?? index}`}
                className="history-item"
                onClick={() => onSelectTicker(h.ticker)}
              >
                <span className="hi-ticker">{h.ticker}</span>
                <span className="hi-detail">{detail}</span>
                <span className="hi-change" style={{ color }}>
                  {arrow} {hasChange ? `${isUp ? '+' : ''}${h.changePercent.toFixed(2)}%` : 'n/a'}
                </span>
                <span className="hi-date">{dateStr}</span>
              </button>
            );
          })
        )}
      </div>
    </section>
  );
}
