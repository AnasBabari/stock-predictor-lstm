import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import ForecastLedgerTrackRecord from './ForecastLedgerTrackRecord';

const mockLedgerResponse = {
  ticker: 'AAPL',
  horizon: 5,
  track_record: {
    total_forecasts: 20,
    scored_forecasts: 18,
    mean_mae: 0.0245,
    mean_rmse: 0.0312,
    mean_qlike: 0.7206,
    direction_accuracy_pct: 68.4,
  },
  entries: [
    {
      id: 1,
      forecast_date: '2026-08-20',
      ticker: 'AAPL',
      horizon: 5,
      target_date: '2026-08-27',
      model_name: 'rolling_mean',
      predicted_volatility: 0.245,
      actual_realized_volatility: 0.231,
      forecast_error: 0.014,
      abs_error: 0.014,
      qlike_loss: 0.5123,
      status: 'scored',
    },
    {
      id: 2,
      forecast_date: '2026-08-25',
      ticker: 'AAPL',
      horizon: 5,
      target_date: '2026-09-01',
      model_name: 'rolling_mean',
      predicted_volatility: 0.252,
      actual_realized_volatility: null,
      forecast_error: null,
      abs_error: null,
      qlike_loss: null,
      status: 'pending',
    },
  ],
};

describe('ForecastLedgerTrackRecord', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockLedgerResponse),
      })
    ));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders track record metrics and forecast entries', async () => {
    render(<ForecastLedgerTrackRecord ticker="AAPL" horizon={5} />);

    expect(screen.getByText(/Loading verified forecast ledger/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Forecast Ledger & Track Record')).toBeInTheDocument();
    });

    expect(screen.getByText('18')).toBeInTheDocument();
    expect(screen.getByText('2.45%')).toBeInTheDocument();
    expect(screen.getByText('0.7206')).toBeInTheDocument();
    expect(screen.getByText('68.4%')).toBeInTheDocument();

    expect(screen.getByText('2026-08-20')).toBeInTheDocument();
    expect(screen.getByText('24.5%')).toBeInTheDocument();
    expect(screen.getByText('23.1%')).toBeInTheDocument();
    expect(screen.getByText('+1.4%')).toBeInTheDocument();
    expect(screen.getByText('Scored')).toBeInTheDocument();
    expect(screen.getByText('Pending')).toBeInTheDocument();
  });

  it('renders null when ticker is omitted', () => {
    const { container } = render(<ForecastLedgerTrackRecord ticker="" horizon={5} />);
    expect(container.firstChild).toBeNull();
  });
});
