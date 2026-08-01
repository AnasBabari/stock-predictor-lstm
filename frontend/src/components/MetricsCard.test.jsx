import { render, screen } from '@testing-library/react';

import MetricsCard from './MetricsCard';

const baselineResponse = {
  metrics: {
    metric_source: 'baseline_definition',
    relative_rmse: 1,
  },
  metadata: {
    engine: {
      family: 'persistence',
      role: 'baseline_fallback',
      baseline_fallback: true,
    },
  },
};

test('identifies a baseline without presenting it as a learned model', () => {
  render(<MetricsCard stockData={baselineResponse} forecastType="price" />);

  expect(screen.getByText('persistence · Baseline fallback')).toBeInTheDocument();
  expect(screen.getByText('Baseline definition')).toBeInTheDocument();
  expect(document.querySelector('.metrics-card--baseline')).toBeInTheDocument();
});

test('identifies a qualifying learned candidate', () => {
  render(
    <MetricsCard
      stockData={{
        ...baselineResponse,
        metadata: {
          engine: {
            family: 'bilstm_attention_direction',
            role: 'learned_candidate',
            baseline_fallback: false,
          },
        },
      }}
      forecastType="trend"
    />,
  );

  expect(
    screen.getByText('bilstm attention direction · Learned candidate'),
  ).toBeInTheDocument();
});
