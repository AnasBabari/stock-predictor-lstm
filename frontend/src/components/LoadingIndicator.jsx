import React from 'react';

function formatDuration(milliseconds) {
  const seconds = Math.max(0, Math.round(Number(milliseconds || 0) / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

export default function LoadingIndicator({ isLoading, stage, progress, onCancel }) {
  if (!isLoading) return null;

  return (
    <div id="loading" className="loading" aria-live="polite">
      <div className="loading-visual">
        <div className="pulse-ring"></div>
        <div className="pulse-ring delay-1"></div>
        <div className="pulse-ring delay-2"></div>
        <svg className="loading-brain" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5" width="32" height="32">
          <path d="M12 2a7 7 0 017 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 01-2 2h-4a2 2 0 01-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 017-7z" />
          <path d="M9 21h6M10 17v4M14 17v4" />
        </svg>
      </div>
      <p className="loading-text">{stage || 'Evaluating certified global volatility forecast…'}</p>
      <div className="loading-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100">
        <div className="loading-progress-fill indeterminate" />
      </div>
      <p className="loading-hint">
        Signed global volatility model · fail-closed server serving
        {progress?.elapsed_ms != null ? ` · elapsed ${formatDuration(progress.elapsed_ms)}` : ''}
      </p>
      <button type="button" className="training-cancel-button" onClick={onCancel}>
        Cancel request
      </button>
    </div>
  );
}
