import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import LoadingIndicator from './LoadingIndicator';

it('renders the server-reported loading stage without percentage progress', () => {
  render(<LoadingIndicator isLoading stage="Training a new model for this ticker…" />);

  expect(screen.getByText('Training a new model for this ticker…')).toBeInTheDocument();
  expect(screen.queryByText(/%/)).not.toBeInTheDocument();
});

describe('LoadingIndicator profile label', () => {
  it('reflects the selected training profile in the duration hint', () => {
    const { container, rerender } = render(
      <LoadingIndicator isLoading stage="Training…" progress={null} profile="quick" />
    );
    expect(container.textContent).toMatch(/Quick · expected/);

    rerender(<LoadingIndicator isLoading stage="Training…" progress={null} profile="research" />);
    expect(container.textContent).toMatch(/Research · expected/);
  });
});
