import React from 'react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import PriceChart from './PriceChart';

vi.mock('../hooks/usePriceHistory', () => ({
  usePriceHistory: vi.fn(),
}));

vi.mock('../hooks/useVolatilityOutlook', () => ({
  useVolatilityOutlook: vi.fn(() => ({ outlook: null, loading: false, error: '', retry: vi.fn() })),
}));

import { usePriceHistory } from '../hooks/usePriceHistory';
import { useVolatilityOutlook } from '../hooks/useVolatilityOutlook';

let lastChartProps = null;
vi.mock('./LazyLineChart', () => ({
  __esModule: true,
  default: React.forwardRef((props, ref) => {
    lastChartProps = props;
    React.useImperativeHandle(ref, () => ({
      chartArea: { left: 0, right: 400, top: 0, bottom: 300, width: 400 },
      scales: {
        x: { getValueForPixel: (px) => (px / 400) * 199 },
        y: { getPixelForValue: () => 150 },
      },
      config: { options: {} },
      update: () => {},
    }));
    return <canvas data-testid="mock-chart" />;
  }),
}));

function historyFixture(sessions = 300, withIntraday = true) {
  return {
    ticker: 'MSFT',
    daily: Array.from({ length: sessions }, (_, i) => ({
      d: new Date(Date.UTC(2024, 5, 1 - (sessions - 1 - i))).toISOString().slice(0, 10),
      c: 100 + i,
    })),
    intraday: withIntraday
      ? Array.from({ length: 20 }, (_, i) => ({ t: `2024-06-01T14:${String(i).padStart(2, '0')}:00+00:00`, c: 200 + i }))
      : null,
  };
}

const forecastFixture = {
  ticker: 'MSFT',
  data_as_of: '2024-06-01',
  current_price: 399,
  future_dates: ['2024-06-03', '2024-06-04', '2024-06-05', '2024-06-06', '2024-06-07', '2024-06-10', '2024-06-11'],
  predicted_prices: [401, 402, 403, 404, 405, 406, 407],
  historical_error_band: {
    upper_prices: [405, 406, 407, 408, 409, 410, 411],
    lower_prices: [397, 398, 399, 400, 401, 402, 403],
  },
};

beforeEach(() => {
  vi.clearAllMocks();
  lastChartProps = null;
});

describe('PriceChart', () => {
  it('defaults to 1M and hides 5Y for a 2024 IPO', () => {
    usePriceHistory.mockReturnValue({ history: historyFixture(500, true), loading: false, error: '', retry: vi.fn() });
    render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    expect(screen.getByRole('tab', { name: '1M' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.queryByRole('tab', { name: '5Y' })).toBeNull();
    expect(screen.getByRole('tab', { name: '1Y' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'MAX' })).toBeInTheDocument();
  });

  it('hides 24H when intraday bars failed to load', () => {
    usePriceHistory.mockReturnValue({ history: historyFixture(3000, false), loading: false, error: '', retry: vi.fn() });
    render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    expect(screen.queryByRole('tab', { name: '24H' })).toBeNull();
    expect(screen.getByRole('tab', { name: '5Y' })).toBeInTheDocument();
  });

  it('shows header price and period change', () => {
    usePriceHistory.mockReturnValue({ history: historyFixture(300, true), loading: false, error: '', retry: vi.fn() });
    render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    expect(screen.getByText('MSFT')).toBeInTheDocument();
    expect(screen.getByText(/\+5\.56%/)).toBeInTheDocument();
  });

  it('switches ranges and overlays the forecast path', () => {
    usePriceHistory.mockReturnValue({ history: historyFixture(300, true), loading: false, error: '', retry: vi.fn() });
    render(<PriceChart ticker="MSFT" currencySymbol="$" forecast={forecastFixture} />);
    fireEvent.click(screen.getByRole('tab', { name: '1M' }));
    expect(screen.getByRole('tab', { name: '1M' })).toHaveAttribute('aria-selected', 'true');
    const labels = lastChartProps.data.datasets.map((ds) => ds.label);
    expect(labels).toContain('Price');
    expect(labels).toContain('7-day estimate');
    expect(lastChartProps.data.labels.length).toBe(22 + 7);
    // Honesty: estimate is dashed, detached from history, inside a marked region.
    const estimate = lastChartProps.data.datasets.find((ds) => ds.label === '7-day estimate');
    expect(estimate.borderDash).toEqual([6, 4]);
    expect(estimate.data.slice(0, 22).every((v) => v == null)).toBe(true);
    expect(lastChartProps.data.forecastSplitIndex).toBe(21);
    expect(lastChartProps.plugins.map((p) => p.id)).toContain('t212ForecastRegion');
  });

  it('wheel zoom reveals a reset control and double-click clears it', () => {
    usePriceHistory.mockReturnValue({ history: historyFixture(300, true), loading: false, error: '', retry: vi.fn() });
    const { container } = render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    fireEvent.click(screen.getByRole('tab', { name: '1M' }));
    expect(screen.queryByRole('button', { name: 'Reset' })).toBeNull();
    fireEvent.wheel(container.querySelector('.t212-chart-wrap'), { clientX: 200, deltaY: -100 });
    expect(screen.getByRole('button', { name: 'Reset' })).toBeInTheDocument();
    fireEvent.doubleClick(container.querySelector('.t212-chart-wrap'));
    expect(screen.queryByRole('button', { name: 'Reset' })).toBeNull();
  });
  it('shows loading and error states', () => {
    const onSettled = vi.fn();
    usePriceHistory.mockReturnValue({ history: null, loading: true, error: '', meta: null, retry: vi.fn() });
    const { unmount } = render(<PriceChart ticker="MSFT" currencySymbol="$" onHistorySettled={onSettled} />);
    expect(screen.getByRole('status')).toHaveTextContent('Loading price history');
    expect(onSettled).not.toHaveBeenCalled();
    unmount();
    const onRetry = vi.fn();
    usePriceHistory.mockReturnValue({ history: null, loading: false, error: 'Down', meta: null, retry: onRetry });
    render(<PriceChart ticker="MSFT" currencySymbol="$" onHistorySettled={onSettled} />);
    fireEvent.click(screen.getByRole('button', { name: 'Retry chart' }));
    expect(onRetry).toHaveBeenCalled();
    expect(onSettled).toHaveBeenCalledWith(expect.objectContaining({ ok: false }));
  });

  it('reports settlement timing once per ticker', () => {
    const onSettled = vi.fn();
    usePriceHistory.mockReturnValue({
      history: historyFixture(300, true), loading: false, error: '',
      meta: { fetchMs: 412, fromCache: false }, retry: vi.fn(),
    });
    const { rerender } = render(<PriceChart ticker="MSFT" currencySymbol="$" onHistorySettled={onSettled} />);
    expect(onSettled).toHaveBeenCalledTimes(1);
    expect(onSettled).toHaveBeenCalledWith(expect.objectContaining({ ok: true, fetchMs: 412, fromCache: false }));
    rerender(<PriceChart ticker="MSFT" currencySymbol="$" onHistorySettled={onSettled} />);
    expect(onSettled).toHaveBeenCalledTimes(1);
  });

  it('falls back to forecast-embedded history when the history endpoint fails', () => {
    usePriceHistory.mockReturnValue({ history: null, loading: false, error: 'Down', meta: null, retry: vi.fn() });
    const fallbackForecast = {
      ...forecastFixture,
      historical_dates: ['2024-05-30', '2024-06-01'],
      historical_prices: [398, 399],
    };
    render(<PriceChart ticker="MSFT" currencySymbol="$" forecast={fallbackForecast} />);
    expect(screen.queryByRole('button', { name: 'Retry chart' })).toBeNull();
    expect(screen.getByRole('tab', { name: 'MAX' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.queryByRole('tab', { name: '5D' })).toBeNull();
    const labels = lastChartProps.data.datasets.map((ds) => ds.label);
    expect(labels).toContain('7-day estimate');
  });

  it('removes automatic uncertainty and volatility bands from this price view by default', () => {
    usePriceHistory.mockReturnValue({ history: historyFixture(300, true), loading: false, error: '', meta: null, retry: vi.fn() });
    render(<PriceChart ticker="MSFT" currencySymbol="$" forecast={forecastFixture} />);
    const labels = lastChartProps.data.datasets.map((ds) => ds.label);
    expect(labels).not.toContain('Expected volatility range (upper)');
    expect(labels).not.toContain('Expected volatility range (lower)');
    expect(labels).not.toContain('Estimate range (upper)');
  });

  it('refuses manufactured history instead of charting it', () => {
    usePriceHistory.mockReturnValue({ history: null, loading: false, error: 'Down', meta: null, retry: vi.fn() });
    render(
      <PriceChart
        ticker="MSFT"
        currencySymbol="$"
        forecast={{
          ...forecastFixture,
          historical_dates: ['2024-05-30', '2024-05-31'],
          historical_prices: [398, 399],
          historical_provenance: 'synthetic',
        }}
      />,
    );
    expect(screen.getByRole('button', { name: 'Retry chart' })).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'MAX' })).toBeNull();
    expect(screen.queryByTestId('mock-chart')).toBeNull();
  });

  it('never shows placeholder zeros while history is loading', () => {
    usePriceHistory.mockReturnValue({ history: null, loading: true, error: '', meta: null, retry: vi.fn() });
    const { container } = render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    // The ticker stays visible, but an unknown price must read as unknown
    // rather than as a real "$0.00 (+0.00%)".
    expect(screen.getByText('MSFT')).toBeInTheDocument();
    expect(screen.queryByText(/\$0\.00/)).toBeNull();
    expect(screen.queryByText(/\(0\.00%\)/)).toBeNull();
    expect(container.querySelector('.t212-price-skeleton')).not.toBeNull();
  });

  it('puts the sign outside the currency symbol for a falling period', () => {
    const falling = {
      ticker: 'MSFT',
      daily: Array.from({ length: 300 }, (_, i) => ({
        d: new Date(Date.UTC(2024, 5, 1 - (299 - i))).toISOString().slice(0, 10),
        c: 400 - i,
      })),
      intraday: null,
    };
    usePriceHistory.mockReturnValue({ history: falling, loading: false, error: '', retry: vi.fn() });
    render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    // "-$1.00" reads correctly; "$-1.00" does not.
    expect(screen.getByText(/\(-\d+\.\d{2}%\)/)).toBeInTheDocument();
    expect(screen.queryByText(/\$-/)).toBeNull();
  });

  it('renders a skeleton chart instead of a blank loading region', () => {
    usePriceHistory.mockReturnValue({ history: null, loading: true, error: '', meta: null, retry: vi.fn() });
    const { container } = render(<PriceChart ticker="MSFT" currencySymbol="$" />);
    expect(screen.getByText('Loading price history…')).toBeInTheDocument();
    expect(container.querySelector('.t212-chart-skeleton')).not.toBeNull();
    expect(container.querySelectorAll('.t212-skeleton-bars i').length).toBeGreaterThan(0);
    expect(screen.queryByTestId('mock-chart')).toBeNull();
  });
});
