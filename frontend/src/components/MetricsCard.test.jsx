import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import MetricsCard from './MetricsCard';

const sampleStockData = {
  ticker: 'MSFT',
  historical_prices: [100, 102],
  predicted_prices: [104, 106],
  forecast_days: 7,
  metrics: {
    relative_rmse: 0.95,
    relative_mae: 0.94,
    directional_accuracy: 0.60,
    r2: 0.05,
    dollar_rmse: 2.1,
    dollar_mae: 1.8,
    per_horizon: [
      { horizon: 1, relative_rmse: 0.98, relative_mae: 0.97, directional_accuracy: 0.55, rows: 200 },
      { horizon: 7, relative_rmse: 0.95, relative_mae: 0.94, directional_accuracy: 0.60, rows: 200 },
    ],
  },
  validation: {
    state: 'promoted',
    promoted: true,
    selected_horizon: 7,
    promoted_horizons: [1, 7],
    best_validated_horizon: 7,
  },
  metadata: {
    engine: {
      family: 'balanced_tfjs_lstm',
      backend: 'webgpu',
    },
    feature_count: 28,
    window_size: 60,
    training_duration_ms: 1200,
  },
};

describe('MetricsCard component', () => {
  it('renders forecast tab with validation badge and metrics overview', () => {
    render(<MetricsCard stockData={sampleStockData} forecastType="price" />);
    expect(screen.getByRole('tab', { name: /forecast/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /evaluation/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /model specs/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /research details/i })).toBeInTheDocument();

    expect(screen.getByText('PROMOTED')).toBeInTheDocument();
    expect(screen.getByText('balanced tfjs lstm')).toBeInTheDocument();
  });

  it('switches tabs smoothly to show evaluation matrix and model specs', () => {
    render(<MetricsCard stockData={sampleStockData} forecastType="price" />);

    // Switch to Evaluation tab
    fireEvent.click(screen.getByRole('tab', { name: /evaluation/i }));
    expect(screen.getByText(/Per-Horizon Evaluation Matrix/i)).toBeInTheDocument();
    expect(screen.getAllByText(/0.950×/i).length).toBeGreaterThanOrEqual(1);

    // Switch to Model Specs tab
    fireEvent.click(screen.getByRole('tab', { name: /model specs/i }));
    expect(screen.getByText(/Stationary v4 \(28 features\)/i)).toBeInTheDocument();
    expect(screen.getByText(/60 trading sessions/i)).toBeInTheDocument();
  });

  it('renders switch horizon banner when requested horizon is experimental but another is promoted', () => {
    const onSwitch = vi.fn();
    const experimentalData = {
      ...sampleStockData,
      validation: {
        state: 'experimental',
        promoted: false,
        selected_horizon: 7,
        promoted_horizons: [3],
        best_validated_horizon: 3,
      },
    };
    render(<MetricsCard stockData={experimentalData} forecastType="price" onSwitchHorizon={onSwitch} />);

    expect(screen.getByText(/7-Day forecast is experimental/i)).toBeInTheDocument();
    const switchBtn = screen.getByRole('button', { name: /switch to 3-day/i });
    expect(switchBtn).toBeInTheDocument();
    fireEvent.click(switchBtn);
    expect(onSwitch).toHaveBeenCalledWith(3);
  });
});
