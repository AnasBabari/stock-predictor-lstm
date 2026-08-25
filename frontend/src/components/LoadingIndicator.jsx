import React from 'react';
import { expectedDurationLabel } from '../ml/trainingProfiles';

function formatDuration(milliseconds) {
  const seconds = Math.max(0, Math.round(Number(milliseconds || 0) / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function progressPercent(progress) {
  if (!progress) return null;
  const epoch = Number(progress.epoch || 0);
  const epochs = Number(progress.total_epochs || 0);
  if (!epochs) return progress.stage === 'cache_hit' ? 100 : null;
  if (progress.stage === 'evaluating_fold') {
    const fold = Math.max(1, Number(progress.fold || 1));
    const folds = Math.max(1, Number(progress.folds || 5));
    return Math.min(90, (((fold - 1) + epoch / epochs) / (folds + 1)) * 100);
  }
  if (progress.stage === 'final_fit') return 85 + (epoch / epochs) * 15;
  return Math.min(80, (epoch / epochs) * 80);
}

export default function LoadingIndicator({ isLoading, stage, progress, profile = 'balanced', onCancel, volatilityServing = false }) {
  if (!isLoading) return null;
  const percent = progressPercent(progress);
  const fold = progress?.fold && progress?.folds ? `Fold ${progress.fold} of ${progress.folds} · ` : '';
  const epoch = progress?.epoch && progress?.total_epochs
    ? `Epoch ${progress.epoch} of ${progress.total_epochs}`
    : '';
  const remainingMs = percent > 1 && percent < 100
    ? Number(progress?.elapsed_ms || 0) * ((100 - percent) / percent)
    : null;

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
      <p className="loading-text">{stage || (volatilityServing ? 'Verifying signed volatility release…' : 'Preparing local browser training…')}</p>
      {(fold || epoch) && <p className="loading-detail">{fold}{epoch}</p>}
      <div className="loading-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={percent == null ? undefined : Math.round(percent)}>
        <div className={`loading-progress-fill${percent == null ? ' indeterminate' : ''}`} style={percent == null ? undefined : { width: `${percent}%` }} />
      </div>
      <p className="loading-hint">
        {volatilityServing ? 'Signed global model · server CPU inference' : `${profile[0].toUpperCase() + profile.slice(1)} · expected ${expectedDurationLabel(profile)}`}
        {progress?.backend ? ` · ${progress.backend.toUpperCase()}` : ''}
        {progress?.elapsed_ms != null ? ` · elapsed ${formatDuration(progress.elapsed_ms)}` : ''}
        {remainingMs != null ? ` · about ${formatDuration(remainingMs)} remaining` : ''}
      </p>
      {!volatilityServing && progress?.benchmark_ms != null && (
        <p className="loading-hint">Local capability check: {progress.benchmark_ms} ms. Models and metrics stay on this device.</p>
      )}
      <button type="button" className="training-cancel-button" onClick={onCancel}>{volatilityServing ? 'Cancel request' : 'Cancel local training'}</button>
    </div>
  );
}
