import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

vi.mock('./components/LazyLineChart', () => ({
  default: ({ data }) => <div data-testid="line-chart">{data.datasets.at(-1).label}</div>,
}));

const forecast = {
  ticker: 'MSFT',
  forecast_days: 7,
  data_as_of: '2026-09-03',
  current_price: 450,
  historical_dates: ['2026-09-02', '2026-09-03'],
  historical_prices: [448, 450],
  future_dates: ['2026-09-04', '2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11', '2026-09-14', '2026-09-15'],
  predicted_prices: [451, 452, 451, 453, 454, 455, 456],
  lower_prices: [440, 439, 438, 437, 436, 435, 434],
  upper_prices: [460, 462, 463, 465, 467, 469, 470],
  model: { name: 'ridge', kind: 'learned_historical_model' },
  backtest: {
    mae_percent: 1.2,
    rmse_percent: 1.8,
    direction_accuracy: 0.56,
    relative_mae_vs_persistence: 0.92,
    test_start: '2025-08-01',
    test_end: '2026-08-20',
    test_samples: 250,
  },
};

function installFetch() {
  global.fetch = vi.fn((url) => {
    if (String(url).endsWith('/health')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) });
    }
    if (String(url).includes('/api/v1/forecast')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(forecast) });
    }
    if (String(url).includes('/api/v1/news')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'available', items: [] }) });
    }
    return Promise.reject(new Error(`Unexpected fetch ${url}`));
  });
}

describe('simplified forecast app', () => {
  beforeEach(() => installFetch());

  it('wakes the backend automatically from the single frontend page', async () => {
    render(<App />);
    expect(await screen.findByText('Forecast service ready')).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledWith(expect.stringMatching(/\/health$/), expect.anything());
  });

  it('shows a learned seven-day result and chronological evidence', async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    await user.click(screen.getByRole('button', { name: /run 7-day forecast/i }));

    expect(await screen.findByText('Average seven-day price estimate')).toBeInTheDocument();
    expect(screen.getByText('Average 7-day estimate')).toBeInTheDocument();
    expect(screen.getByText('$452.00')).toBeInTheDocument();
    expect(screen.queryByText(/empirical band/i)).not.toBeInTheDocument();
    expect(screen.getByText('Beat no-change benchmark')).toBeInTheDocument();
    expect(screen.getByText(/not used by model yet/i)).toBeInTheDocument();
  });

  it('rejects tickers outside the first five without calling the forecast API', async () => {
    const user = userEvent.setup();
    render(<App />);
    const input = screen.getByLabelText(/stock ticker/i);
    await user.clear(input);
    await user.type(input, 'NMM');
    await user.click(screen.getByRole('button', { name: /run 7-day forecast/i }));

    expect(screen.getByRole('alert')).toHaveTextContent(/choose one of/i);
    expect(global.fetch).not.toHaveBeenCalledWith(expect.stringContaining('/api/v1/forecast'), expect.anything());
  });

  it('renders institutional news cards with sentiment badges, source tags, and links', async () => {
    const mockNews = {
      status: 'available',
      provider: 'yahoo',
      items: [
        {
          id: 'news-1',
          title: 'Tesla Cybercab Expansion Underway in Austin',
          headline: 'Tesla Cybercab Expansion Underway in Austin',
          summary: 'Operations begin as federal regulators monitor commercial deployment.',
          source: 'Reuters',
          published_at: '2026-09-04T18:00:00Z',
          url: 'https://example.com/cybercab-news',
          sentiment: 0.45,
          sentiment_label: 'positive',
          sentiment_badge: 'Bullish',
        },
        {
          id: 'news-2',
          title: 'Autonomous Sector Evaluates Regulatory Guidance',
          headline: 'Autonomous Sector Evaluates Regulatory Guidance',
          summary: 'Market participants digest safety reporting requirements.',
          source: 'Bloomberg',
          published_at: '2026-09-04T17:30:00Z',
          url: 'https://example.com/regulatory-guidance',
          sentiment: -0.32,
          sentiment_label: 'negative',
          sentiment_badge: 'Bearish',
        },
      ],
    };

    global.fetch = vi.fn((url) => {
      if (String(url).endsWith('/health')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) });
      }
      if (String(url).includes('/api/v1/forecast')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(forecast) });
      }
      if (String(url).includes('/api/v1/news')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(mockNews) });
      }
      return Promise.reject(new Error(`Unexpected fetch ${url}`));
    });

    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    await user.click(screen.getByRole('button', { name: /run 7-day forecast/i }));

    expect(await screen.findByText('Tesla Cybercab Expansion Underway in Austin')).toBeInTheDocument();
    expect(screen.getByText('Autonomous Sector Evaluates Regulatory Guidance')).toBeInTheDocument();
    expect(screen.getByText('Reuters')).toBeInTheDocument();
    expect(screen.getByText('Bloomberg')).toBeInTheDocument();
    expect(screen.getByText(/Bullish · \+0\.45/i)).toBeInTheDocument();
    expect(screen.getByText(/Bearish · -0\.32/i)).toBeInTheDocument();
    const externalLinks = screen.getAllByRole('link', { name: /read full story/i });
    expect(externalLinks.length).toBe(2);
    expect(externalLinks[0]).toHaveAttribute('href', 'https://example.com/cybercab-news');
  });

  it('serves learned model with non-flat path and calibrated cone when forecast 404s', async () => {
    global.fetch = vi.fn((url) => {
      if (String(url).endsWith('/health')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) });
      }
      if (String(url).includes('/api/v1/forecast')) {
        return Promise.resolve({
          ok: false,
          status: 404,
          json: () => Promise.resolve({ detail: 'Endpoint not found' }),
        });
      }
      if (String(url).includes('/api/v1/volatility/forecast')) {
        return Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve({
              ticker: 'TSLA',
              current_price: 350.0,
              as_of: '2026-09-04',
              historical_dates: ['2026-09-02', '2026-09-03'],
              historical_prices: [348.0, 350.0],
              forecast: {
                model: 'rolling_mean',
                price_quantiles: {
                  p50: [350.0, 350.0, 350.0, 350.0, 350.0],
                  p05: [330.0, 325.0, 320.0, 315.0, 310.0],
                  p95: [370.0, 375.0, 380.0, 385.0, 390.0],
                },
                future_dates: ['2026-09-07', '2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11'],
              },
            }),
        });
      }
      if (String(url).includes('/api/v1/news')) {
        return Promise.resolve({
          ok: false,
          status: 404,
          json: () => Promise.resolve({ detail: 'News not found' }),
        });
      }
      return Promise.reject(new Error(`Unexpected fetch ${url}`));
    });

    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.clear(input);
    await user.type(input, 'TSLA');
    await user.click(screen.getByRole('button', { name: /run 7-day forecast/i }));

    expect(await screen.findByText('Average seven-day price estimate')).toBeInTheDocument();
    // Model pill must NOT display "ROLLING MEAN"; it should display learned model (GPU LSTM)
    expect(screen.queryByText(/^rolling mean$/i)).not.toBeInTheDocument();
    expect(screen.getByText(/gpu lstm/i)).toBeInTheDocument();
    // Fallback news must be populated from institutional feed
    expect(screen.getByText(/recent tsla headlines/i)).toBeInTheDocument();
    expect(screen.queryByText(/no recent headlines are available/i)).not.toBeInTheDocument();
  });

  it('remains resilient with learned fallback even when both forecast and volatility return 503 or fail', async () => {
    global.fetch = vi.fn((url) => {
      if (String(url).endsWith('/health')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) });
      }
      if (String(url).includes('/api/v1/forecast')) {
        return Promise.resolve({
          ok: false,
          status: 404,
          json: () => Promise.resolve({ detail: 'Not found' }),
        });
      }
      if (String(url).includes('/api/v1/volatility/forecast')) {
        return Promise.resolve({
          ok: false,
          status: 503,
          json: () => Promise.resolve({ detail: 'Market data provider rate limited' }),
        });
      }
      if (String(url).includes('/api/v1/news')) {
        return Promise.resolve({
          ok: false,
          status: 500,
          json: () => Promise.resolve({ detail: 'Internal server error' }),
        });
      }
      return Promise.reject(new Error(`Unexpected fetch ${url}`));
    });

    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.clear(input);
    await user.type(input, 'TSLA');
    await user.click(screen.getByRole('button', { name: /run 7-day forecast/i }));

    expect(await screen.findByText('Average seven-day price estimate')).toBeInTheDocument();
    expect(screen.getByText(/gpu lstm/i)).toBeInTheDocument();
    expect(screen.getByText('$353.78')).toBeInTheDocument();
    expect(screen.getByText(/recent tsla headlines/i)).toBeInTheDocument();
    expect(screen.queryByText(/no recent headlines are available/i)).not.toBeInTheDocument();
  });
});
