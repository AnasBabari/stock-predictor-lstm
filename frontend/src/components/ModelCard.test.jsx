import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import ModelCard from './ModelCard';

const payload = {
  ticker: 'MSFT',
  forecast_days: 7,
  predicted_prices: [1, 2],
  metadata: {
    window_size: 60,
    feature_count: 28,
    schema_version: 4,
    metric_source: 'browser_purged_holdout',
    snapshot_id: 'abcdef1234567890',
    engine: { family: 'balanced_tfjs_lstm', role: 'browser_learned' },
    data_snapshot: { adjusted_prices: true, quality: { status: 'annotated' } },
  },
};

describe('ModelCard', () => {
  it('renders methodology facts from the payload', () => {
    render(<ModelCard data={payload} />);
    expect(screen.getByText('Cumulative log return r(t,h) = ln(P(t+h) / P(t)); prices reconstructed from the latest close.')).toBeInTheDocument();
    expect(screen.getByText('60 sessions')).toBeInTheDocument();
    expect(screen.getByText('balanced_tfjs_lstm (browser_learned)')).toBeInTheDocument();
    expect(screen.getByText(/Robust median\/IQR, fit on the training partition only/)).toBeInTheDocument();
    expect(screen.getByText('browser_purged_holdout')).toBeInTheDocument();
    expect(screen.getByText(/Split\/dividend-adjusted closes/)).toBeInTheDocument();
    expect(screen.getByText('Annotated warnings — see snapshot metadata')).toBeInTheDocument();
    expect(screen.getByText('abcdef123456…')).toBeInTheDocument();
  });

  it('discloses the final-refit caveat and limitations', () => {
    render(<ModelCard data={payload} />);
    expect(screen.getAllByText(/refitted on all history/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/simple baselines are often competitive/i)).toBeInTheDocument();
    expect(screen.getByText(/financial advice/i)).toBeInTheDocument();
  });

  it('renders nothing without a payload', () => {
    const { container } = render(<ModelCard data={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('handles missing optional metadata gracefully', () => {
    render(<ModelCard data={{ metadata: {} }} />);
    expect(screen.getAllByText('?').length).toBeGreaterThan(0);
    expect(screen.queryByText(/undefined/)).not.toBeInTheDocument();
  });
});
