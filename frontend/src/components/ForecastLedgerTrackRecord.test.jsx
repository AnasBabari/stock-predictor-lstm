import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import ForecastLedgerTrackRecord from './ForecastLedgerTrackRecord';

const mockLedgerResponse = {
  ticker: 'AAPL',
  horizon: 5,
  live_track_record: {
    record_source: 'live',
    total_forecasts: 2,
    scored_forecasts: 1,
    mean_mae: 0.014,
    mean_rmse: 0.014,
    mean_qlike: 0.5123,
    direction_accuracy_pct: 100.0,
  },
  replay_track_record: {
    record_source: 'historical_replay',
    total_forecasts: 25,
    scored_forecasts: 25,
    mean_mae: 0.0245,
    mean_rmse: 0.0312,
    mean_qlike: 0.7206,
    direction_accuracy_pct: 68.4,
  },
  track_record: {
    record_source: 'live',
    total_forecasts: 2,
    scored_forecasts: 1,
    mean_mae: 0.014,
    mean_rmse: 0.014,
    mean_qlike: 0.5123,
    direction_accuracy_pct: 100.0,
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
      record_source: 'live',
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
      record_source: 'live',
    },
    {
      id: 3,
      forecast_date: '2026-07-15',
      ticker: 'AAPL',
      horizon: 5,
      target_date: '2026-07-22',
      model_name: 'rolling_mean',
      predicted_volatility: 0.22,
      actual_realized_volatility: 0.21,
      forecast_error: 0.01,
      abs_error: 0.01,
      qlike_loss: 0.6,
      status: 'scored',
      record_source: 'historical_replay',
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

  it('renders track record metrics and forecast entries without verified wording', async () => {
    render(<ForecastLedgerTrackRecord ticker="AAPL" horizon={5} />);

    expect(screen.getByText(/Loading forecast track record/i)).toBeInTheDocument();
    expect(screen.queryByText(/verified/i)).not.toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Forecast Ledger & Track Record')).toBeInTheDocument();
    });
    expect(screen.getByText('AAPL 5 sessions')).toBeInTheDocument();
    expect(screen.getAllByText('5 sessions').length).toBeGreaterThan(0);

    // Headline live settlements KPI shows 1 scored live forecast
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('1.40%')).toBeInTheDocument();
    expect(screen.getAllByText('0.5123').length).toBeGreaterThanOrEqual(1);

    // Source chips
    expect(screen.getAllByText('Live').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Replay').length).toBeGreaterThan(0);

    // Filter by replay tab
    const replayTab = screen.getByRole('button', { name: /Replay \(25\)/i });
    fireEvent.click(replayTab);

    // Under replay tab, replay metrics are shown
    expect(screen.getByText('25')).toBeInTheDocument();
    expect(screen.getByText('2.45%')).toBeInTheDocument();
    expect(screen.getByText('0.7206')).toBeInTheDocument();
  });

  it('renders null when ticker is omitted', () => {
    const { container } = render(<ForecastLedgerTrackRecord ticker="" horizon={5} />);
    expect(container.firstChild).toBeNull();
  });
});
