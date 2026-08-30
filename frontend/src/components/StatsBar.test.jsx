import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import StatsBar from './StatsBar';

describe('StatsBar certified volatility presentation', () => {
  it('shows the forecast range and volatility without a flat price claim', () => {
    render(
      <StatsBar
        forecastType="price"
        stockData={{
          ticker: 'MSFT',
          forecast_days: 7,
          historical_prices: [100],
          predicted_prices: null,
          volatility_cone: { p05: [94], p95: [109] },
          forecast: { expected_annualized_volatility: 0.27 },
          validation: { state: 'certified_volatility', promoted: true },
          metadata: { engine: { certified_head: 'volatility' } },
        }}
      />,
    );
    expect(screen.getByText('90% Forecast Range')).toBeInTheDocument();
    expect(screen.getByText('$94.00 – $109.00')).toBeInTheDocument();
    expect(screen.getByText('Annualized Volatility')).toBeInTheDocument();
    expect(screen.getByText('27.00%')).toBeInTheDocument();
    expect(screen.getByText('VOLATILITY CERTIFIED')).toBeInTheDocument();
    expect(screen.queryByText(/unchanged close|0\.00%/i)).not.toBeInTheDocument();
  });

  it('shows the certified return-distribution median and expected return', () => {
    render(
      <StatsBar
        forecastType="price"
        stockData={{
          ticker: 'MSFT',
          forecast_days: 7,
          historical_prices: [100],
          predicted_prices: [101, 103],
          volatility_cone: { p05: [94, 95], p95: [109, 112] },
          forecast: { expected_annualized_volatility: 0.27 },
          validation: { state: 'certified_return_distribution', promoted: true },
          metadata: { engine: { certified_head: 'return_distribution' } },
        }}
      />,
    );
    expect(screen.getByText('Return distribution (7d)')).toBeInTheDocument();
    expect(screen.getByText('$103.00')).toBeInTheDocument();
    expect(screen.getByText('Expected Return')).toBeInTheDocument();
    expect(screen.getByText('+3.00%')).toBeInTheDocument();
    expect(screen.getByText('DISTRIBUTION CERTIFIED')).toBeInTheDocument();
    expect(screen.queryByText('Volatility certified')).not.toBeInTheDocument();
  });
});
