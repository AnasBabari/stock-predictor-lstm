import React from 'react';
import { render, screen } from '@testing-library/react';
import SimpleForecastChart, { midpointPrices } from './SimpleForecastChart';

vi.mock('./LazyLineChart', () => ({
  default: ({ data }) => (
    <div data-testid="chart-data">{JSON.stringify(data.datasets)}</div>
  ),
}));

describe('SimpleForecastChart', () => {
  it('calculates the midpoint between matching lower and upper estimates', () => {
    expect(midpointPrices([100, 120], [200, 180])).toEqual([150, 150]);
  });

  it('plots one average forecast line without upper or lower estimate lines', () => {
    render(
      <SimpleForecastChart
        forecast={{
          historical_dates: ['2026-09-02', '2026-09-03'],
          historical_prices: [148, 150],
          future_dates: ['2026-09-04', '2026-09-08'],
          lower_prices: [100, 120],
          upper_prices: [200, 180],
        }}
      />,
    );

    const datasets = JSON.parse(screen.getByTestId('chart-data').textContent);
    expect(datasets.map(({ label }) => label)).toEqual([
      'Historical close',
      'Average 7-day estimate',
    ]);
    expect(datasets.at(-1).data).toEqual([null, 150, 150, 150]);
  });
});
