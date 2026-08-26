import React from 'react';

const BADGE_CONFIG = {
  promoted: {
    label: 'PROMOTED',
    className: 'badge--promoted',
    title: 'Validated against persistence on held-out evaluation.',
    icon: '✓',
  },
  certified_volatility: {
    label: 'VOLATILITY CERTIFIED',
    className: 'badge--promoted',
    title: 'The conditional-volatility head passed locked certification; price direction is not certified.',
    icon: '✓',
  },
  candidate: {
    label: 'CANDIDATE',
    className: 'badge--candidate',
    title: 'Competitive with benchmark; validation evidence is accumulating.',
    icon: '●',
  },
  experimental: {
    label: 'EXPERIMENTAL',
    className: 'badge--experimental',
    title: 'Model forecast shown for research; validation gates were not met.',
    icon: '⚑',
  },
  unavailable: {
    label: 'UNAVAILABLE',
    className: 'badge--unavailable',
    title: 'Validation unavailable: evaluation could not be verified.',
    icon: '—',
  },
};

export default function ValidationBadge({ state, className = '' }) {
  const normalizedState = (state || 'experimental').toLowerCase();
  const config = BADGE_CONFIG[normalizedState] || BADGE_CONFIG.experimental;

  return (
    <span
      className={`validation-badge ${config.className} ${className}`}
      title={config.title}
      role="status"
    >
      <span className="badge-icon" aria-hidden="true">{config.icon}</span>
      <span className="badge-text">{config.label}</span>
    </span>
  );
}
