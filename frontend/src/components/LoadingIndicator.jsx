import React from 'react';

export default function LoadingIndicator({ isLoading }) {
  if (!isLoading) return null;

  return (
    <div id="loading" className="loading">
      <div className="loading-visual">
        <div className="pulse-ring"></div>
        <div className="pulse-ring delay-1"></div>
        <div className="pulse-ring delay-2"></div>
        <svg
          className="loading-brain"
          viewBox="0 0 24 24"
          fill="none"
          stroke="var(--accent)"
          strokeWidth="1.5"
          width="32"
          height="32"
        >
          <path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 01-2 2h-4a2 2 0 01-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7z" />
          <path d="M9 21h6M10 17v4M14 17v4" />
        </svg>
      </div>
      <p className="loading-text">Running LSTM model…</p>
      <p className="loading-hint">
        First prediction for a new ticker may take longer while the model trains.
      </p>
      <div className="loading-bar">
        <div className="loading-bar-fill"></div>
      </div>
    </div>
  );
}
