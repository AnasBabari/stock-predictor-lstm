function formatHistoryDate(dateInput) {
  if (!dateInput) return '';
  const parsed = new Date(dateInput);
  if (isNaN(parsed.getTime())) return '';
  return parsed.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
  });
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
          items.map((h) => {
            const isUp = parseFloat(h.change) >= 0;
            const color = isUp ? 'var(--bullish)' : 'var(--bearish)';
            const arrow = isUp ? '▲' : '▼';
            const dateStr = formatHistoryDate(h.date);

            return (
              <button
                type="button"
                key={`${h.ticker}-${h.date}`}
                className="history-item"
                onClick={() => onSelectTicker(h.ticker)}
              >
                <span className="hi-ticker">{h.ticker}</span>
                <span className="hi-detail">
                  ${h.lastClose?.toFixed(2)} → ${h.forecast?.toFixed(2)} · {h.days}d
                </span>
                <span className="hi-change" style={{ color }}>
                  {arrow} {isUp ? '+' : ''}
                  {h.change}%
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
