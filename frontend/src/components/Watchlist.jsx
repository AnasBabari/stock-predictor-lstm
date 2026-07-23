import React from 'react';

export default function Watchlist({ items, onSelectTicker, onRemoveItem, onClearAll }) {
  return (
    <section className="panel-card" id="watchlistSection">
      <div className="panel-header">
        <h3>
          <svg
            viewBox="0 0 20 20"
            fill="currentColor"
            width="14"
            height="14"
            style={{ color: 'var(--accent)' }}
          >
            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
          </svg>
          Watchlist
        </h3>
        <button type="button" className="text-btn" onClick={onClearAll} title="Clear all">
          Clear
        </button>
      </div>
      <div id="watchlistItems" className="panel-items">
        {items.length === 0 ? (
          <p className="empty-state">
            No tickers saved yet.
            <br />
            Click <strong>Watch</strong> after predicting to add one.
          </p>
        ) : (
          items.map((w, index) => (
            <div key={w.ticker} className="watchlist-item">
              <button
                type="button"
                className="wl-select-btn"
                onClick={() => onSelectTicker(w.ticker)}
              >
                <span className="wl-ticker">{w.ticker}</span>
                <span className="wl-name">{w.name || ''}</span>
                <span className="wl-price">
                  {w.lastPrice ? `$${w.lastPrice.toFixed(2)}` : '—'}
                </span>
              </button>
              <button
                type="button"
                className="wl-remove"
                title="Remove"
                onClick={() => onRemoveItem(index)}
              >
                ✕
              </button>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
