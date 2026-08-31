import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import LoadingIndicator from './LoadingIndicator';

describe('LoadingIndicator', () => {
  it('renders the loading stage and hint', () => {
    render(<LoadingIndicator isLoading stage="Evaluating certified global volatility forecast…" />);

    expect(screen.getByText('Evaluating certified global volatility forecast…')).toBeInTheDocument();
    expect(screen.getByText(/Causal market baseline/)).toBeInTheDocument();
  });

  it('triggers onCancel when Cancel request button is clicked', () => {
    const onCancel = vi.fn();
    render(<LoadingIndicator isLoading onCancel={onCancel} />);

    fireEvent.click(screen.getByRole('button', { name: /cancel request/i }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('returns null when isLoading is false', () => {
    const { container } = render(<LoadingIndicator isLoading={false} />);
    expect(container).toBeEmptyDOMElement();
  });
});
