import React from 'react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

vi.mock('./components/LazyLineChart', () => ({
  default: ({ data }) => <div data-testid="line-chart">{data.datasets.at(-1).label}</div>,
}));

function volatilityBody(horizon) {
  const dates = Array.from({ length: 60 }, (_, index) => {
    const day = new Date(Date.UTC(2026, 5, 1) + index * 86400000);
    return day.toISOString().slice(0, 10);
  });
  const future = Array.from({ length: horizon }, (_, index) => {
    const day = new Date(Date.UTC(2026, 8, 4) + index * 86400000);
    return day.toISOString().slice(0, 10);
  });
  const prices = Array.from({ length: 60 }, (_, index) => 440 + index * 0.2);
  const quantiles = {};
  for (const [key, offset] of [['p05', -12], ['p10', -8], ['p25', -4], ['p50', 0], ['p75', 4], ['p90', 8], ['p95', 12]]) {
    quantiles[key] = Array(horizon).fill(450 + offset);
  }
  return {
    ticker: 'MSFT',
    as_of: '2026-09-03',
    horizon,
    current_price: 450,
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
      model_status: 'gpu_promoted',
      model_family: 'global_gpu_xgboost',
      model_name: 'gpu_g3',
      requested_model: 'gpu_g3',
      baseline: false,
      model_version: 'g3-gpu-xgb-v1',
      metric_source: 'held_out_test_panel',
      risk_level: 'Elevated',
      risk_ratio_vs_trailing_60d: 1.32,
      trailing_annualized_volatility_60d: 0.162,
      data_as_of: '2026-09-03',
    },
  };
}

function installFetch() {
  global.fetch = vi.fn((url) => {
    const urlStr = String(url);
    if (urlStr.endsWith('/health')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) });
    }
    if (urlStr.includes('/api/v1/history')) {
      return Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            ticker: 'MSFT',
            daily: [
              { d: '2026-09-01', c: 448 },
              { d: '2026-09-02', c: 450 },
            ],
            intraday: null,
          }),
      });
    }
    if (urlStr.includes('/api/v1/volatility/forecast')) {
      const match = urlStr.match(/horizon=(\d+)/);
      const horizon = Number(match?.[1] || 5);
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(volatilityBody(horizon)),
      });
    }
    return Promise.reject(new Error(`Unexpected fetch ${url}`));
  });
}

describe('Signal Seven Volatility App', () => {
  beforeEach(() => {
    installFetch();
  });

  it('wakes the backend automatically from the single frontend page', async () => {
    render(<App />);
    expect(await screen.findByText('Volatility engine ready')).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledWith(expect.stringMatching(/\/health$/), expect.anything());
  });

  it('renders the single-screen volatility outlook and price chart', async () => {
    render(<App />);
    expect(await screen.findByText('Volatility engine ready')).toBeInTheDocument();
    expect(screen.getByText('Equity Volatility Engine')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /Forecast Stock/i })).toBeInTheDocument();
    expect(await screen.findByRole('heading', { name: 'Volatility Outlook' })).toBeInTheDocument();
    expect(screen.getByText('21.4% annualised')).toBeInTheDocument();
    expect(screen.getByText('Historical Prices')).toBeInTheDocument();
    expect(screen.getByText('Expected 5D Volatility Cone (p05–p95)')).toBeInTheDocument();
  });

  it('offers one ticker input without exchange tabs or complex grids', () => {
    render(<App />);
    expect(screen.getAllByRole('textbox')).toHaveLength(1);
    expect(screen.queryByRole('tablist', { name: 'Ticker selection matrix' })).not.toBeInTheDocument();
    expect(screen.getByText('286 supported US & UK tickers')).toBeInTheDocument();
  });

  it.each(['nvda', 'jpm', 'shel.l'])('submits the typed ticker %s', async (symbol) => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText('Volatility engine ready');
    const input = screen.getByLabelText(/stock ticker/i);
    await user.clear(input);
    await user.type(input, `${symbol}{Enter}`);
    expect(input).toHaveValue(symbol.toUpperCase());
    expect(await screen.findByRole('heading', { name: 'Volatility Outlook' })).toBeInTheDocument();
  });

  it('rejects unsupported tickers', async () => {
    const user = userEvent.setup();
    render(<App />);
    const input = screen.getByLabelText(/stock ticker/i);
    await user.clear(input);
    await user.type(input, 'INVALIDXYZ{Enter}');
    expect(screen.getByRole('alert')).toHaveTextContent(/choose one of/i);
  });
});
