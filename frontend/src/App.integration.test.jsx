import React from 'react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import { clearPriceHistoryCache } from './api/priceHistoryClient';

vi.mock('./components/LazyLineChart', () => ({
  default: React.forwardRef(({ data }, ref) => (
    <div ref={ref} data-testid="line-chart">
      {data?.datasets?.map((d) => d.label).join(', ') || ''}
    </div>
  )),
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
  model: { name: 'gpu_lstm', kind: 'learned_historical_model' },
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
    const urlStr = String(url);
    if (urlStr.endsWith('/health')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) });
    }
    const tickerMatch = urlStr.match(/ticker=([^&]+)/);
    const sym = tickerMatch ? decodeURIComponent(tickerMatch[1]).toUpperCase() : 'MSFT';
    const isPence = sym.endsWith('.L');
    const basePrice = isPence ? 2600.5 : (sym === 'NVDA' ? 120 : (sym === 'JPM' ? 210 : 450));

    if (urlStr.includes('/api/v1/history')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          ticker: sym,
          as_of: '2026-09-03',
          daily: [
            { d: '2026-09-02', c: basePrice - 2 },
            { d: '2026-09-03', c: basePrice },
          ],
        }),
      });
    }
    if (urlStr.includes('/api/v1/forecast')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          ...forecast,
          ticker: sym,
          currency_symbol: isPence ? 'p' : '$',
          current_price: basePrice,
          historical_prices: [basePrice - 2, basePrice],
          lower_prices: [basePrice - 10, basePrice - 11, basePrice - 12, basePrice - 13, basePrice - 14, basePrice - 15, basePrice - 16],
          upper_prices: [basePrice + 10, basePrice + 12, basePrice + 13, basePrice + 15, basePrice + 17, basePrice + 19, basePrice + 20],
        }),
      });
    }
    if (urlStr.includes('/api/v1/volatility/forecast')) {
      const match = urlStr.match(/horizon=(\d+)/);
      return Promise.resolve({ ok: true, json: () => Promise.resolve(volatilityBody(Number(match?.[1] || 5), sym, basePrice)) });
    }
    if (urlStr.includes('/api/v1/news')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'available', items: [] }) });
    }
    return Promise.reject(new Error(`Unexpected fetch ${url}`));
  });
}

function volatilityBody(horizon, sym = 'MSFT', basePrice = 450) {
  const dates = Array.from({ length: 60 }, (_, index) => {
    const day = new Date(Date.UTC(2026, 5, 1) + index * 86400000);
    return day.toISOString().slice(0, 10);
  });
  const future = Array.from({ length: horizon }, (_, index) => {
    const day = new Date(Date.UTC(2026, 8, 4) + index * 86400000);
    return day.toISOString().slice(0, 10);
  });
  const prices = Array.from({ length: 60 }, (_, index) => (basePrice - 10) + index * 0.2);
  const quantiles = {};
  for (const [key, offset] of [['p05', -12], ['p10', -8], ['p25', -4], ['p50', 0], ['p75', 4], ['p90', 8], ['p95', 12]]) {
    quantiles[key] = Array(horizon).fill(basePrice + offset);
  }
  return {
    ticker: sym,
    as_of: '2026-09-03',
    horizon,
    current_price: basePrice,
    historical_dates: dates,
    historical_prices: prices,
    forecast: {
      future_dates: future,
      price_quantiles: quantiles,
      model: 'gpu_g3',
      requested_model: 'gpu_g3',
      expected_annualized_volatility: { 5: 0.214, 10: 0.231, 20: 0.208 }[horizon] ?? 0.214,
    },
    evidence: {
      model_status: 'learned_model',
      model_family: 'global_gpu_xgboost',
      model_name: 'gpu_g3',
      requested_model: 'gpu_g3',
      baseline: false,
      model_version: 'g3-qlike-base-margin-v2',
      metric_source: 'validation_panel',
      risk_level: 'Elevated',
      risk_ratio_vs_trailing_60d: 1.32,
      trailing_annualized_volatility_60d: 0.162,
      data_as_of: '2026-09-03',
    },
  };
}

describe('simplified forecast app', () => {
  beforeEach(() => {
    installFetch();
    clearPriceHistoryCache();
  });

  it('wakes the backend automatically from the single frontend page without fetching stock data on first visit', async () => {
    render(<App />);
    expect(await screen.findByText('Forecast service ready')).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledWith(expect.stringMatching(/\/health$/), expect.anything());
    // Critical requirement: Do not fetch any stock's history, forecast, or news before submission
    expect(global.fetch).not.toHaveBeenCalledWith(expect.stringMatching(/\/history/), expect.anything());
    expect(global.fetch).not.toHaveBeenCalledWith(expect.stringMatching(/\/forecast/), expect.anything());
    expect(global.fetch).not.toHaveBeenCalledWith(expect.stringMatching(/\/news/), expect.anything());
  });

  it('first render opens to a minimal screen with empty input, disabled button, and no results or skeletons', async () => {
    const { container } = render(<App />);
    expect(screen.getByRole('heading', { name: /which stock would you like to explore\?/i })).toBeInTheDocument();
    const input = screen.getByLabelText(/stock ticker/i);
    expect(input).toHaveValue('');
    expect(input).toHaveAttribute('placeholder', 'Enter a stock ticker');
    const button = screen.getByRole('button', { name: /view outlook/i });
    expect(button).toBeDisabled();
    expect(container.querySelector('#chartContainer')).not.toBeInTheDocument();
    expect(container.querySelector('.t212-chart-skeleton')).not.toBeInTheDocument();
    expect(screen.queryByText(/Day 7 estimate:/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/recent .* headlines/i)).not.toBeInTheDocument();
    await screen.findByText('Forecast service ready');
  });

  it('shows a learned seven-day result and chronological evidence in performance tab', async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.type(input, 'MSFT');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));

    // Chart has the single in-chart estimate line
    expect(await screen.findByText(/Day 7 estimate:/i)).toBeInTheDocument();
    expect(screen.getByText(/\$452\.00/)).toBeInTheDocument();
    expect(screen.queryByText(/empirical band/i)).not.toBeInTheDocument();

    // Check Performance tab for chronological backtest evidence
    const perfTab = screen.getByRole('tab', { name: /performance/i });
    await user.click(perfTab);
    expect(screen.getByText('More accurate than assuming no price change')).toBeInTheDocument();

    // Check News tab
    const newsTab = screen.getByRole('tab', { name: /news/i });
    await user.click(newsTab);
    expect(screen.getByText(/not included in the forecast/i)).toBeInTheDocument();
  });

  it('offers one ticker input without exchange tabs or ticker grids', async () => {
    render(<App />);
    expect(screen.getAllByRole('textbox')).toHaveLength(1);
    expect(screen.queryByRole('tablist')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Ticker selection matrix')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Quick ticker switcher')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'AAPL' })).not.toBeInTheDocument();
    expect(screen.getByText('Stock forecasts made simple')).toBeInTheDocument();
    expect(screen.queryByText(/A ticker is a stock's short code/)).not.toBeInTheDocument();
    expect(screen.queryByText(/PostgreSQL|70\/15\/15|Quantitative Terminal|causal market/i)).not.toBeInTheDocument();
    await screen.findByText('Forecast service ready');
  });

  it('renders the volatility outlook card in Overview tab without model codenames', async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.type(input, 'MSFT');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));

    expect(await screen.findByRole('heading', { name: 'Volatility Outlook' })).toBeInTheDocument();
    expect(screen.getByText('21.4% annualised')).toBeInTheDocument();
    expect(screen.getAllByText(/historical validation panel/).length).toBeGreaterThan(0);
    const card = screen.getByLabelText('MSFT volatility outlook');
    expect(card.textContent).not.toMatch(/G3|XGBoost|HAR|QLIKE/i);
  });

  it.each(['nvda', 'jpm', 'shel.l'])('submits the typed ticker %s with Enter', async (symbol) => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.clear(input);
    await user.type(input, `${symbol}{Enter}`);
    await screen.findByText(/Day 7 estimate:/i);
    expect(input).toHaveValue(symbol.toUpperCase());
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining(`/api/v1/forecast?ticker=${symbol.toUpperCase()}`),
      expect.anything(),
    );
  });

  it('rejects unsupported tickers without calling the forecast API', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    const input = screen.getByLabelText(/stock ticker/i);
    await user.clear(input);
    await user.type(input, 'NMM');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));

    expect(screen.getByRole('alert')).toHaveTextContent(/choose one of/i);
    expect(global.fetch).not.toHaveBeenCalledWith(expect.stringContaining('/api/v1/forecast'), expect.anything());
    expect(container.querySelector('#chartContainer')).not.toBeInTheDocument();
  });

  it('renders institutional news in News tab with simple list, source, and links', async () => {
    const mockNews = {
      status: 'available',
      provider: 'yahoo',
      items: [
        {
          id: 'news-1',
          title: 'Tesla Cybercab Expansion Underway in Austin',
          headline: 'Tesla Cybercab Expansion Underway in Austin',
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
    const input = screen.getByLabelText(/stock ticker/i);
    await user.type(input, 'MSFT');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));

    await screen.findByText(/Day 7 estimate:/i);
    // Switch to News tab
    const newsTab = screen.getByRole('tab', { name: /news/i });
    await user.click(newsTab);

    expect(await screen.findByText('Tesla Cybercab Expansion Underway in Austin')).toBeInTheDocument();
    expect(screen.getByText('Autonomous Sector Evaluates Regulatory Guidance')).toBeInTheDocument();
    expect(screen.getByText('Reuters')).toBeInTheDocument();
    expect(screen.getByText('Bloomberg')).toBeInTheDocument();
    expect(screen.getByText('Positive tone')).toBeInTheDocument();
    expect(screen.getByText('Negative tone')).toBeInTheDocument();
    const link = screen.getByRole('link', { name: /Tesla Cybercab/i });
    expect(link).toHaveAttribute('href', 'https://example.com/cybercab-news');
  });

  it('separates typing from submitted stock and does not trigger requests or change results on typing', async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.type(input, 'MSFT');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));
    await screen.findByText(/Day 7 estimate:/i);

    const callCountBeforeTyping = global.fetch.mock.calls.length;
    // Typing another ticker should not trigger requests or change current results
    await user.clear(input);
    await user.type(input, 'NVDA');
    expect(global.fetch.mock.calls.length).toBe(callCountBeforeTyping);
    expect(screen.getByText(/Day 7 estimate:/i)).toBeInTheDocument();
  });

  it('retains usable chart and shows actionable forecast error on partial results', async () => {
    global.fetch = vi.fn((url) => {
      if (String(url).endsWith('/health')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) });
      }
      if (String(url).includes('/api/v1/history')) {
        return Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve({
              ticker: 'MSFT',
              daily: [
                { d: '2026-09-02', c: 448 },
                { d: '2026-09-03', c: 450 },
              ],
            }),
        });
      }
      if (String(url).includes('/api/v1/forecast')) {
        return Promise.resolve({
          ok: false,
          status: 500,
          json: () => Promise.resolve({ detail: 'Model inference server error' }),
        });
      }
      if (String(url).includes('/api/v1/news')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'available', items: [] }) });
      }
      return Promise.reject(new Error(`Unexpected fetch ${url}`));
    });

    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.type(input, 'MSFT');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));

    // History succeeded, so chart should be revealed!
    expect(await screen.findByLabelText(/MSFT price chart/i)).toBeInTheDocument();
    // Forecast failed, so actionable forecast error should be shown!
    expect(await screen.findByRole('alert')).toHaveTextContent(/could not be completed|server error/i);
    expect(screen.getByRole('button', { name: /retry forecast/i })).toBeInTheDocument();
  });

  it('handles history failure compactly with retry button without reserving empty chart panel', async () => {
    global.fetch = vi.fn((url) => {
      if (String(url).endsWith('/health')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) });
      }
      if (String(url).includes('/api/v1/history')) {
        return Promise.resolve({
          ok: false,
          status: 503,
          json: () => Promise.resolve({ detail: 'Upstream provider down' }),
        });
      }
      if (String(url).includes('/api/v1/forecast')) {
        return Promise.resolve({
          ok: false,
          status: 503,
          json: () => Promise.resolve({ detail: 'Upstream provider down' }),
        });
      }
      if (String(url).includes('/api/v1/news')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'available', items: [] }) });
      }
      return Promise.reject(new Error(`Unexpected fetch ${url}`));
    });

    const user = userEvent.setup();
    const { container } = render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.type(input, 'MSFT');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));

    // Chart container must NOT be mounted
    expect(container.querySelector('#chartContainer')).not.toBeInTheDocument();
    // Compact error with retry must appear
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/upstream provider down|price history is unavailable/i);
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('cancels outstanding requests and ignores late responses when switching stocks', async () => {
    let resolveFirstForecast;
    const firstForecastPromise = new Promise((resolve) => {
      resolveFirstForecast = resolve;
    });

    global.fetch = vi.fn((url) => {
      if (String(url).endsWith('/health')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) });
      }
      if (String(url).includes('ticker=NVDA') && String(url).includes('/forecast')) {
        return firstForecastPromise.then(() => ({
          ok: true,
          json: () => Promise.resolve({
            ...forecast,
            ticker: 'NVDA',
            current_price: 120,
            historical_prices: [118, 120],
          }),
        }));
      }
      if (String(url).includes('ticker=MSFT') && String(url).includes('/forecast')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(forecast),
        });
      }
      if (String(url).includes('/api/v1/volatility/forecast')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(volatilityBody(5)) });
      }
      if (String(url).includes('/api/v1/news')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'available', items: [] }) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });

    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);

    // Submit NVDA first
    await user.type(input, 'NVDA');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));

    // Quickly submit MSFT
    await user.clear(input);
    await user.type(input, 'MSFT');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));

    // Now resolve the late NVDA forecast
    resolveFirstForecast();

    // The rendered outlook must be MSFT, never overwritten by NVDA
    await screen.findByText(/Day 7 estimate:/i);
    expect(screen.getAllByText(/data through 2026-09-03/i).length).toBeGreaterThan(0);
    expect(screen.queryByText('NVDA (NVDA)')).not.toBeInTheDocument();
  });

  it('retains usable chart when retrying a failed forecast for the same stock', async () => {
    let forecastShouldFail = true;
    global.fetch = vi.fn((url) => {
      if (String(url).endsWith('/health')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) });
      }
      if (String(url).includes('/api/v1/history')) {
        return Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve({
              ticker: 'MSFT',
              daily: [
                { d: '2026-09-02', c: 448 },
                { d: '2026-09-03', c: 450 },
              ],
            }),
        });
      }
      if (String(url).includes('/api/v1/forecast')) {
        if (forecastShouldFail) {
          return Promise.resolve({
            ok: false,
            status: 500,
            json: () => Promise.resolve({ detail: 'Temporary failure' }),
          });
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve(forecast) });
      }
      if (String(url).includes('/api/v1/news')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'available', items: [] }) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });

    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.type(input, 'MSFT');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));

    // History succeeds, chart is mounted
    expect(await screen.findByLabelText(/MSFT price chart/i)).toBeInTheDocument();
    // Forecast failed, actionable retry button is visible
    const retryBtn = await screen.findByRole('button', { name: /retry forecast/i });
    expect(retryBtn).toBeInTheDocument();

    // Now retry forecast: chart must stay mounted
    forecastShouldFail = false;
    await user.click(retryBtn);

    // Forecast resolves successfully
    expect(await screen.findByText(/Day 7 estimate:/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/MSFT price chart/i)).toBeInTheDocument();
  });

  it('never displays missing forecast or backtest values as $0.00 or +0.00%', async () => {
    const incompleteForecast = {
      ...forecast,
      current_price: null,
      backtest: {
        ...forecast.backtest,
        mae_percent: null,
        rmse_percent: null,
        direction_accuracy: null,
        relative_mae_vs_persistence: null,
      },
    };

    global.fetch = vi.fn((url) => {
      if (String(url).endsWith('/health')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) });
      }
      if (String(url).includes('/api/v1/forecast')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(incompleteForecast) });
      }
      if (String(url).includes('/api/v1/news')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'available', items: [] }) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });

    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.type(input, 'MSFT');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));

    // Switch to Performance tab to see backtest metrics
    const perfTab = await screen.findByRole('tab', { name: /performance/i });
    await user.click(perfTab);

    expect(await screen.findByText('Price Model Historical Performance')).toBeInTheDocument();
    const dashes = screen.getAllByText('—');
    expect(dashes.length).toBeGreaterThanOrEqual(4);
    expect(screen.queryByText('$0.00')).not.toBeInTheDocument();
  });

  it('formats UK stock prices with pence suffix p instead of prefix', async () => {
    const ukForecast = {
      ...forecast,
      ticker: 'SHEL.L',
      currency_symbol: 'p',
      current_price: 2600.5,
      historical_prices: [2590, 2600.5],
      predicted_prices: [2610, 2620, 2630, 2640, 2650, 2660, 2670],
      lower_prices: [2550, 2560, 2570, 2580, 2590, 2600, 2610],
      upper_prices: [2650, 2660, 2670, 2680, 2690, 2700, 2710],
    };

    global.fetch = vi.fn((url) => {
      if (String(url).endsWith('/health')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) });
      }
      if (String(url).includes('/api/v1/forecast')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(ukForecast) });
      }
      if (String(url).includes('/api/v1/news')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'available', items: [] }) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });

    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.type(input, 'SHEL.L');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));

    expect(await screen.findByText(/Day 7 estimate:/i)).toBeInTheDocument();
    expect(screen.getByText('2,600.5p')).toBeInTheDocument();
    expect(screen.queryByText(/p2,600/)).not.toBeInTheDocument();
  });

  it('supports expanded chart mode and closes with Escape key', async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Forecast service ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.type(input, 'MSFT');
    await user.click(screen.getByRole('button', { name: /view outlook/i }));

    await screen.findByText(/Day 7 estimate:/i);
    const expandBtn = screen.getByRole('button', { name: /expand chart/i });
    await user.click(expandBtn);

    // Modal dialog is open
    expect(screen.getByRole('dialog', { name: /expanded price chart/i })).toBeInTheDocument();
    expect(screen.getByText(/close \(esc\)/i)).toBeInTheDocument();

    // Press Escape
    await user.keyboard('{Escape}');

    // Modal dialog is closed
    expect(screen.queryByRole('dialog', { name: /expanded price chart/i })).not.toBeInTheDocument();
  });
});
