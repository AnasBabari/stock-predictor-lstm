import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

vi.mock('./components/SplashScreen', () => ({
  default: () => null,
}));

vi.mock('./components/Navbar', () => ({
  default: () => null,
}));

vi.mock('./components/HeroSection', () => ({
  default: () => null,
}));

vi.mock('./components/LoadingIndicator', () => ({
  default: ({ isLoading }) => (isLoading ? <div>Loading...</div> : null),
}));

vi.mock('./components/StockInfoGrid', () => ({
  default: () => null,
}));

vi.mock('./components/Watchlist', () => ({
  default: () => null,
}));

vi.mock('./components/PredictionHistory', () => ({
  default: () => null,
}));

vi.mock('./components/ToastContainer', () => ({
  default: () => null,
}));

vi.mock('./components/LazyLineChart', () => ({
  default: React.forwardRef(function MockLineChart({ data }, ref) {
    React.useImperativeHandle(ref, () => ({ toBase64Image: () => 'data:image/png;base64,mock' }));
    return <div data-testid="line-chart">{data?.labels?.join(',') || 'chart'}</div>;
  }),
}));

vi.mock('./components/StockChart', async () => {
  const actual = await vi.importActual('./components/StockChart');
  return actual;
});

vi.mock('./utils/exportService', () => ({
  exportPriceCSV: vi.fn(),
  exportTrendCSV: vi.fn(),
  exportAttentionCSV: vi.fn(),
  exportCompleteAnalysis: vi.fn(),
}));

const priceResponse = {
  ticker: 'TSLA',
  historical_dates: ['2026-07-18', '2026-07-21'],
  historical_prices: [390.0, 400.0],
  future_dates: ['2026-07-24', '2026-07-25'],
  predicted_prices: [405.0, 410.0],
  forecast_days: 2,
  metrics: { rmse: 1.2, mae: 0.8, r2: 0.99, mape: 0.5, directional_accuracy: 0.75 },
};

const trendResponse = {
  ticker: 'TSLA',
  forecast_days: 2,
  future_dates: ['2026-07-24', '2026-07-25'],
  directions: ['Up', 'Down'],
  probabilities: [0.65, 0.42],
  attention_weights: [
    { index: 0, date: '2026-07-18', weight: 0.2 },
    { index: 1, date: '2026-07-21', weight: 0.8 },
  ],
  metrics: { precision: 0.7, recall: 0.6, f1: 0.65, naive_baseline: 0.5 },
  sentiment: { score: 0.1, status: 'ok', provider: 'test', method: 'mock' },
};

const infoResponse = {
  ticker: 'TSLA',
  name: 'Tesla',
  sector: 'Automotive',
};

function mockFetchSequence() {
  global.fetch = vi.fn((url) => {
    const requestUrl = String(url);
    if (requestUrl.includes('/api/v1/search')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ results: [] }) });
    }
    if (requestUrl.includes('/api/v1/info')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(infoResponse) });
    }
    if (requestUrl.includes('/api/v1/predict/direction')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(trendResponse) });
    }
    if (requestUrl.includes('/api/v1/predict')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(priceResponse) });
    }
    return Promise.reject(new Error(`Unhandled fetch: ${requestUrl}`));
  });
}

describe('forecast toggle integration', () => {
  beforeEach(() => {
    mockFetchSequence();
    localStorage.clear();
  });

  it('switches between forecast types without stale state', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByPlaceholderText(/search tickers/i), 'TSLA');
    await user.click(screen.getByRole('button', { name: /^price forecast$/i }));
    await user.click(screen.getByRole('button', { name: /^predict$/i }));

    expect(await screen.findByText('TSLA')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^price forecast$/i })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText(/Historical vs Predicted/i)).toBeInTheDocument();
    expect(screen.getByText('Price Forecast Metrics')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /^trend forecast$/i }));
    expect(screen.queryByText(/Historical vs Predicted/i)).not.toBeInTheDocument();
    expect(screen.queryByText('Price Forecast Metrics')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /^predict$/i }));

    await waitFor(() => expect(screen.getByText('Trend Forecast Metrics')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /^trend forecast$/i })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.queryByText(/Historical vs Predicted/i)).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /^price forecast$/i }));
    expect(screen.queryByText('Trend Forecast Metrics')).not.toBeInTheDocument();
    expect(screen.queryByText(/Historical vs Predicted/i)).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /^predict$/i }));

    await waitFor(() => expect(screen.getByText('Price Forecast Metrics')).toBeInTheDocument());
    expect(screen.getByText(/Historical vs Predicted/i)).toBeInTheDocument();
    expect(screen.queryByText('Trend Forecast Metrics')).not.toBeInTheDocument();
  });
});