import React from 'react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import VolatilityOutlook, { expectedRange, formatAnnualized } from './VolatilityOutlook';

vi.mock('../hooks/useVolatilityOutlook', () => ({
  useVolatilityOutlook: vi.fn(),
  OUTLOOK_HORIZONS: [5, 10, 20],
}));

import { useVolatilityOutlook } from '../hooks/useVolatilityOutlook';

function entry(annual, risk, trailing) {
  return {
    forecast: { expected_annualized_volatility: annual },
    evidence: {
      risk_level: risk,
      trailing_annualized_volatility_60d: trailing,
      data_as_of: '2026-09-04',
    },
  };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('expectedRange', () => {
  it('computes a symmetric one-sigma band that widens with horizon', () => {
    const short = expectedRange(425.3, 0.214, 5);
    expect(short.low).toBeLessThan(425.3);
    expect(short.high).toBeGreaterThan(425.3);
    expect(short.low).toBeCloseTo(412.4, 0);
    expect(short.high).toBeCloseTo(438.6, 0);
    const long = expectedRange(425.3, 0.214, 20);
    expect(long.high - long.low).toBeGreaterThan(short.high - short.low);
  });

  it('returns null for invalid inputs', () => {
    expect(expectedRange(0, 0.2, 5)).toBeNull();
    expect(expectedRange(100, -0.2, 5)).toBeNull();
    expect(expectedRange(100, 0.2, 0)).toBeNull();
  });

  it('formats annualised percentages', () => {
    expect(formatAnnualized(0.214)).toBe('21.4%');
    expect(formatAnnualized(null)).toBe('—');
  });
});

describe('VolatilityOutlook', () => {
  it('renders the outlook table with risk levels and no model codenames', () => {
    useVolatilityOutlook.mockReturnValue({
      outlook: {
        ticker: 'MSFT',
        byHorizon: { 5: entry(0.214, 'Elevated', 0.162), 10: entry(0.231, 'Elevated', 0.17), 20: entry(0.208, 'Moderate', 0.19) },
      },
      loading: false,
      error: '',
      retry: vi.fn(),
    });
    const { container } = render(<VolatilityOutlook ticker="MSFT" currencySymbol="$" currentPrice={425.3} />);
    expect(screen.getByRole('heading', { name: 'Volatility Outlook' })).toBeInTheDocument();
    expect(screen.getByText('21.4% annualised')).toBeInTheDocument();
    // Elevated repeats per elevated row (no combined block without a price estimate here).
    expect(screen.getAllByText('Elevated')).toHaveLength(2);
    expect(screen.getByText('Moderate')).toBeInTheDocument();
    expect(container.textContent).not.toMatch(/G3|XGBoost|HAR|QLIKE/i);
    expect(screen.getByText(/held-out test panel/)).toBeInTheDocument();
  });

  it('combines the price estimate with the volatility band', () => {
    useVolatilityOutlook.mockReturnValue({
      outlook: { ticker: 'MSFT', byHorizon: { 5: entry(0.214, 'Elevated', 0.162) } },
      loading: false,
      error: '',
      retry: vi.fn(),
    });
    render(
      <VolatilityOutlook
        ticker="MSFT"
        currencySymbol="$"
        currentPrice={425.3}
        priceEstimate={{ price: 431.2, changePct: 1.5 }}
      />,
    );
    expect(screen.getByText('$431.20')).toBeInTheDocument();
    expect(screen.getByText('+1.5%')).toBeInTheDocument();
    expect(screen.getByText(/\$412\.4.*\$438\.6|\$412.*\$438/)).toBeInTheDocument();
  });

  it('shows diagnostics with baseline comparison and adjustment', () => {
    useVolatilityOutlook.mockReturnValue({
      outlook: { ticker: 'MSFT', byHorizon: { 5: entry(0.214, 'Elevated', 0.162) } },
      loading: false,
      error: '',
      retry: vi.fn(),
    });
    render(<VolatilityOutlook ticker="MSFT" currencySymbol="$" currentPrice={425.3} />);
    expect(screen.getByText('+32.1%')).toBeInTheDocument();
    expect(screen.getByText('2026-09-04')).toBeInTheDocument();
  });

  it('shows loading and retry states', () => {
    useVolatilityOutlook.mockReturnValue({ outlook: null, loading: true, error: '', retry: vi.fn() });
    const { unmount } = render(<VolatilityOutlook ticker="MSFT" />);
    expect(screen.getByRole('status')).toHaveTextContent('Loading volatility outlook');
    unmount();
    const onRetry = vi.fn();
    useVolatilityOutlook.mockReturnValue({ outlook: null, loading: false, error: 'Down', retry: onRetry });
    render(<VolatilityOutlook ticker="MSFT" />);
    fireEvent.click(screen.getByRole('button', { name: 'Retry outlook' }));
    expect(onRetry).toHaveBeenCalled();
  });
});
