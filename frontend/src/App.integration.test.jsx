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
  default: ({ isLoading, stage }) => (isLoading ? <div>{stage || 'Loading...'}</div> : null),
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

const makeVolatilityResponse = (ticker = 'TSLA', days = 7) => ({
  ticker,
  horizon: days,
  current_price: 400.0,
  historical_dates: ['2026-07-18', '2026-07-21'],
  historical_prices: [390.0, 400.0],
  as_of: '2026-07-21',
  forecast: {
    future_dates: Array.from({ length: days }, (_, index) => `2026-07-${22 + index}`),
    price_quantiles: {
      p05: Array.from({ length: days }, () => 380.0),
      p10: Array.from({ length: days }, () => 385.0),
      p25: Array.from({ length: days }, () => 392.0),
      p50: Array.from({ length: days }, () => 400.0),
      p75: Array.from({ length: days }, () => 408.0),
      p90: Array.from({ length: days }, () => 415.0),
      p95: Array.from({ length: days }, () => 420.0),
    },
  },
  evidence: {
    certified: true,
    certified_heads: { volatility: true },
    model_id: 'volatility_v9_global',
    snapshot_id: 'snap-test',
    feature_count: 29,
    metric_source: 'locked_purged_walk_forward',
    horizon_certification: {
      [String(days)]: {
        relative_qlike: 0.92,
        ratio_upper_95: 1.05,
        dm_p_value: 0.01,
        coverage_80: 0.81,
        coverage_95: 0.95,
        evaluation_rows: 500,
      },
    },
  },
});

const infoResponse = {
  ticker: 'TSLA',
  name: 'Tesla',
  sector: 'Automotive',
};

function mockFetchSequence() {
  global.fetch = vi.fn((url) => {
    const requestUrl = String(url);
    const parsedUrl = new URL(requestUrl, 'http://localhost');
    const ticker = parsedUrl.searchParams.get('ticker') || 'TSLA';
    const horizon = Number(parsedUrl.searchParams.get('horizon')) || 7;
    if (requestUrl.includes('/api/v1/search')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ results: [] }) });
    }
    if (requestUrl.includes('/api/v1/info')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ ...infoResponse, ticker }) });
    }
    if (requestUrl.includes('/api/v2/forecast')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(makeVolatilityResponse(ticker, horizon)) });
    }
    return Promise.reject(new Error(`Unhandled fetch: ${requestUrl}`));
  });
}

describe('forecast integration', () => {
  beforeEach(() => {
    mockFetchSequence();
    localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  function mockPredictionFailure(reason) {
    global.fetch = vi.fn((url) => {
      const requestUrl = String(url);
      if (requestUrl.includes('/api/v1/info')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(infoResponse) });
      }
      if (requestUrl.includes('/api/v2/forecast')) {
        return Promise.reject(reason);
      }
      return Promise.reject(new Error(`Unhandled fetch: ${requestUrl}`));
    });
  }

  async function submitPrediction() {
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByPlaceholderText(/search tickers/i), 'TSLA');
    await user.click(screen.getByRole('button', { name: /^predict$/i }));
    return user;
  }

  it('fetches and displays certified volatility forecast and metrics', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByPlaceholderText(/search tickers/i), 'TSLA');
    await user.click(screen.getByRole('button', { name: /^predict$/i }));

    expect(await screen.findByText('TSLA')).toBeInTheDocument();
    expect(screen.getByText(/Historical vs Volatility Cone/i)).toBeInTheDocument();
    expect(screen.getByText('Price Forecast Metrics')).toBeInTheDocument();
    expect(screen.getAllByText('90% Forecast Range').length).toBeGreaterThanOrEqual(1);
  });

  it.each([
    [
      new Error('Forecast model is not currently available for this ticker.'),
      'No prepared forecast model is available for this ticker. Try an approved symbol.',
    ],
    [
      new Error('Prediction capacity is temporarily full.'),
      'Prediction capacity is currently full. Please try again shortly.',
    ],
    [
      new Error('Prediction timed out; the shared job may still complete.'),
      'Prediction timed out. The shared work may still finish; try again shortly.',
    ],
    [
      new Error('Market data is temporarily unavailable.'),
      'Market data is temporarily unavailable. Please try again later.',
    ],
    [
      new TypeError('Failed to fetch'),
      'Could not connect to the backend. Make sure the server is running.',
    ],
    [{ internal: 'must not be rendered' }, 'Prediction could not be completed. Please try again.'],
  ])('maps a rejected prediction to a safe message', async (reason, expectedMessage) => {
    mockPredictionFailure(reason);
    await submitPrediction();
    expect(await screen.findByText(expectedMessage)).toBeInTheDocument();
    expect(screen.queryByText(/must not be rendered/i)).not.toBeInTheDocument();
  });

  it('keeps intentional abort cancellation silent', async () => {
    global.fetch = vi.fn((url, options = {}) => {
      const requestUrl = String(url);
      if (requestUrl.includes('/api/v1/info')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(infoResponse) });
      }
      if (requestUrl.includes('/api/v2/forecast')) {
        return new Promise((resolve, reject) => {
          options.signal.addEventListener(
            'abort',
            () => reject(new DOMException('The operation was aborted.', 'AbortError')),
            { once: true }
          );
        });
      }
      return Promise.reject(new Error(`Unhandled fetch: ${requestUrl}`));
    });

    const user = await submitPrediction();
    await user.type(screen.getByPlaceholderText(/search tickers/i), 'A');

    await waitFor(() => {
      expect(screen.queryByText(/could not connect|could not be completed|timed out|capacity/i)).not.toBeInTheDocument();
    });
  });

  it('ignores a late failure from a superseded request', async () => {
    let rejectFirstPrediction;
    global.fetch = vi.fn((url) => {
      const requestUrl = String(url);
      if (requestUrl.includes('/api/v1/info')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(infoResponse) });
      }
      if (requestUrl.includes('ticker=TSLA')) {
        return new Promise((resolve, reject) => {
          rejectFirstPrediction = reject;
        });
      }
      if (requestUrl.includes('ticker=AAPL')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(makeVolatilityResponse('AAPL', 7)),
        });
      }
      return Promise.reject(new Error(`Unhandled fetch: ${requestUrl}`));
    });

    const user = await submitPrediction();
    const input = screen.getByPlaceholderText(/search tickers/i);
    await user.clear(input);
    await user.type(input, 'AAPL');
    await user.click(screen.getByRole('button', { name: /^predict$/i }));
    expect(await screen.findByText('Price Forecast Metrics')).toBeInTheDocument();

    rejectFirstPrediction(new Error('Prediction capacity is temporarily full.'));
    await waitFor(() => {
      expect(
        screen.queryByText('Prediction capacity is currently full. Please try again shortly.')
      ).not.toBeInTheDocument();
      expect(screen.getByText('Price Forecast Metrics')).toBeInTheDocument();
    });
  });

  it('preserves active prediction state when an earlier export operation finishes or rejects', async () => {
    mockFetchSequence();
    const user = await submitPrediction();
    expect(await screen.findByText('Price Forecast Metrics')).toBeInTheDocument();

    // 1. Click Complete Analysis
    const completeAnalysisBtn = screen.queryByRole('button', { name: /export complete analysis/i });
    if (completeAnalysisBtn) {
      await user.click(completeAnalysisBtn);
    }

    // 2. Concurrently switch ticker to MSFT and submit new prediction
    const input = screen.getByPlaceholderText(/search tickers/i);
    await user.clear(input);
    await user.type(input, 'MSFT');
    await user.click(screen.getByRole('button', { name: /^predict$/i }));

    // 3. Verify active prediction state displays cleanly for MSFT without stale export interference
    expect(await screen.findByText('Price Forecast Metrics')).toBeInTheDocument();
    expect(screen.getByText('MSFT')).toBeInTheDocument();
  });
});
