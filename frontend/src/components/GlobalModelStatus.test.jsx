import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import GlobalModelStatus from './GlobalModelStatus';

describe('GlobalModelStatus', () => {
  it('renders nothing without global_model payload', () => {
    const { container } = render(<GlobalModelStatus data={{}} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows promoted status without alpha', () => {
    render(<GlobalModelStatus data={{
      global_model: { status: 'promoted', interval: null },
    }} />);
    expect(screen.getByText('Global model forecast')).toBeInTheDocument();
  });

  it('shows blended status with alpha and interval', () => {
    render(<GlobalModelStatus data={{
      global_model: {
        status: 'blended_with_baseline',
        alpha: 0.4,
        interval: { low: 98.2, high: 104.5 },
      },
    }} />);
    expect(screen.getByText(/Blended toward persistence/)).toBeInTheDocument();
    expect(screen.getByText(/α=0.40/)).toBeInTheDocument();
    expect(screen.getByText(/\[98\.20, 104\.50\]/)).toBeInTheDocument();
  });

  it('shows abstention message', () => {
    render(<GlobalModelStatus data={{
      global_model: { status: 'insufficient_data' },
    }} />);
    expect(screen.getByText(/Insufficient comparable data/)).toBeInTheDocument();
  });
});
