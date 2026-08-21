import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import HoldoutComparisonChart from './HoldoutComparisonChart';

const series = {
  horizon: 7,
  step: 6,
  metric_scope: 'untouched_post_purge_holdout',
  dates: ['2025-01-06', '2025-01-07', '2025-01-08'],
  actual: [101, 102, 100.5],
  model: [103, 104, 103.2],
  persistence: [100, 100, 100],
};

describe('HoldoutComparisonChart', () => {
  it('renders nothing without a series (baselines, research profile)', () => {
    const { container } = render(<HoldoutComparisonChart data={{}} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('labels the holdout comparison and the final-refit caveat', () => {
    render(<HoldoutComparisonChart data={{ evaluation_series: series }} />);
    expect(screen.getByText(/Out-of-sample holdout/i)).toBeInTheDocument();
    expect(
      screen.getByText(/final model refitted on all available history/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/Untouched holdout only/i)).toBeInTheDocument();
  });

  it('reports truncation when capped', () => {
    const truncated = { ...series, truncated: 140 };
    render(<HoldoutComparisonChart data={{ evaluation_series: truncated }} />);
    expect(screen.getByText(/Showing the latest 3 of 143 origins/i)).toBeInTheDocument();
  });
});
