import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

const browserTrainingState = vi.hoisted(() => ({ defer: false, resolve: null, reject: null, signal: null }));

vi.mock('./ml/browserTrainingClient', () => ({
  browserTrainingSupported: vi.fn(() => true),
  clearBrowserModelCache: vi.fn(() => Promise.resolve()),
  trainBrowserForecast: vi.fn(({ forecastType, days, signal, onProgress }) => {
    browserTrainingState.signal = signal;
    onProgress?.({ stage: 'training', message: 'Training epoch 4 of 12…' });
    const result = forecastType === 'trend'
      ? { directions: Array.from({ length: days }, () => 'Up'), probabilities: Array.from({ length: days }, () => 0.65), metrics: { metric_source: 'browser_purged_holdout', accuracy: 0.6 }, cacheStatus: 'stored', backend: 'cpu', executionMode: 'browser_trained' }
      : { predictedPrices: Array.from({ length: days }, (_, index) => 405 + index), metrics: { metric_source: 'browser_purged_holdout', rmse: 1.2, mae: 0.8 }, cacheStatus: 'stored', backend: 'cpu', executionMode: 'browser_trained' };
    return new Promise((resolve, reject) => {
      browserTrainingState.reject = reject;
      signal?.addEventListener('abort', () => reject(new DOMException('The operation was aborted.', 'AbortError')), { once: true });
      if (browserTrainingState.defer) browserTrainingState.resolve = () => resolve(result);
      else resolve(result);
    });
  }),
}));

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

const priceResponse = {
  ticker: 'TSLA',
  historical_dates: ['2026-07-18', '2026-07-21'],
  historical_prices: [390.0, 400.0],
  future_dates: Array.from({ length: 7 }, (_, index) => `2026-07-${24 + index}`),
  predicted_prices: Array.from({ length: 7 }, (_, index) => 405.0 + index),
  forecast_days: 7,
  metrics: { rmse: 1.2, mae: 0.8, r2: 0.99, mape: 0.5, directional_accuracy: 0.75 },
};

const trendResponse = {
  ticker: 'TSLA',
  forecast_days: 7,
  future_dates: Array.from({ length: 7 }, (_, index) => `2026-07-${24 + index}`),
  directions: ['Up', 'Down', 'Up', 'Up', 'Down', 'Up', 'Down'],
  probabilities: [0.65, 0.42, 0.7, 0.6, 0.4, 0.8, 0.3],
  attention_weights: [
    { index: 0, date: '2026-07-18', weight: 0.2 },
    { index: 1, date: '2026-07-21', weight: 0.8 },
  ],
  metrics: { precision: 0.7, recall: 0.6, f1: 0.65, naive_baseline: 0.5 },
  sentiment: { score: 0.1, status: 'ok', provider: 'test', method: 'mock' },
};

const trainingSnapshot = {
  ticker: 'TSLA',
  schema_version: 4,
  snapshot_id: 'snapshot-test',
  feature_names: Array.from({ length: 29 }, (_, index) => `Feature_${index}`),
  window_size: 60,
  output_width: 30,
  dates: ['2026-07-18', '2026-07-21'],
  features: [],
  historical_prices: [390.0, 400.0],
  future_dates: Array.from({ length: 30 }, (_, index) => `2026-07-${24 + index}`),
  data_snapshot: { snapshot_id: 'snapshot-test' },
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
    if (requestUrl.includes('/api/v1/training-data')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(trainingSnapshot) });
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
    browserTrainingState.defer = false;
    browserTrainingState.resolve = null;
    browserTrainingState.reject = null;
    browserTrainingState.signal = null;
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
      if (requestUrl.includes('/api/v1/prediction-status/')) {
        return Promise.resolve({ ok: false });
      }
      if (requestUrl.includes('/api/v1/info')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(infoResponse) });
      }
      if (requestUrl.includes('/api/v1/predict')) {
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

  it('reports browser training progress and uses the feature snapshot', async () => {
    browserTrainingState.defer = true;
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByPlaceholderText(/search tickers/i), 'TSLA');
    await user.click(screen.getByRole('button', { name: /^predict$/i }));

    expect(await screen.findByText('Training epoch 4 of 12…')).toBeInTheDocument();
    expect(browserTrainingState.signal).toBeTruthy();
    browserTrainingState.resolve();
    await waitFor(() => expect(screen.getByText('Price Forecast Metrics')).toBeInTheDocument());
    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/api/v1/training-data?ticker=TSLA'), expect.anything());
  });

  it('aborts browser training on unmount without server status polling', async () => {
    browserTrainingState.defer = true;
    const user = userEvent.setup();
    const { unmount } = render(<App />);
    await user.type(screen.getByPlaceholderText(/search tickers/i), 'TSLA');
    await user.click(screen.getByRole('button', { name: /^predict$/i }));
    await screen.findByText('Training epoch 4 of 12…');

    unmount();
    expect(browserTrainingState.signal.aborted).toBe(true);
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
      if (requestUrl.includes('/api/v1/prediction-status/')) {
        return Promise.resolve({ ok: false });
      }
      if (requestUrl.includes('/api/v1/info')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(infoResponse) });
      }
      if (requestUrl.includes('/api/v1/predict')) {
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
      if (requestUrl.includes('/api/v1/prediction-status/')) {
        return Promise.resolve({ ok: false });
      }
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
          json: () => Promise.resolve({ ...priceResponse, ticker: 'AAPL' }),
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
});
