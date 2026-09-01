import React from 'react';
import LazyLineChart from './LazyLineChart';

const SERIES_COLORS = {
  actual: '#4da3ff',
  model: '#f5a623',
  persistence: '#8b93a7',
};

/**
 * Out-of-sample holdout comparison for price forecasts on the
 * single-holdout profiles (Quick/Balanced). Renders actual vs model vs
 * persistence on identical backend-generated target dates. Absent series =>
 * renders nothing (Research profile and baselines have no aligned timeline).
 */
export default function HoldoutComparisonChart({ data }) {
  const series = data?.evaluation_series;
  if (!series || !Array.isArray(series.dates) || series.dates.length === 0) {
    return null;
  }

  const datasets = [
    {
      label: 'Actual',
      data: series.actual,
      borderColor: SERIES_COLORS.actual,
      backgroundColor: 'transparent',
      pointRadius: 0,
      borderWidth: 1.5,
      tension: 0.1,
    },
    {
      label: 'Model (holdout)',
      data: series.model,
      borderColor: SERIES_COLORS.model,
      backgroundColor: 'transparent',
      pointRadius: 0,
      borderWidth: 1.5,
      borderDash: [],
      tension: 0.1,
    },
    {
      label: 'Persistence baseline',
      data: series.persistence,
      borderColor: SERIES_COLORS.persistence,
      backgroundColor: 'transparent',
      pointRadius: 0,
      borderWidth: 1.5,
      borderDash: [6, 4],
      tension: 0,
    },
  ];

  return (
    <section className="panel-card" id="holdoutComparison">
      <div className="panel-header">
        <h3>
          Out-of-sample holdout · final step of {series.horizon}d horizon
        </h3>
      </div>
      <p className="panel-subtitle">
        Untouched holdout only — the future forecast above comes from a final
        model refitted on all available history and is not independently
        validated.
        {series.truncated ? ` Showing the latest ${series.dates.length} of ${series.dates.length + series.truncated} origins.` : ''}
      </p>
      <div style={{ position: 'relative', minHeight: '260px' }}>
        <LazyLineChart
          data={{ labels: series.dates, datasets }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
              legend: { position: 'bottom', labels: { boxWidth: 12 } },
              tooltip: { callbacks: {} },
            },
            scales: {
              x: { ticks: { maxTicksLimit: 8 } },
              y: { title: { display: true, text: 'Price' } },
            },
          }}
        />
      </div>
    </section>
  );
}
