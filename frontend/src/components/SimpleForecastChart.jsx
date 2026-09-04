import React from 'react';
import LazyLineChart from './LazyLineChart';

export function midpointPrices(lowerPrices = [], upperPrices = []) {
  const length = Math.min(lowerPrices.length, upperPrices.length);
  return Array.from({ length }, (_, index) => {
    const lower = Number(lowerPrices[index]);
    const upper = Number(upperPrices[index]);
    return Number.isFinite(lower) && Number.isFinite(upper) ? (lower + upper) / 2 : null;
  });
}

export default function SimpleForecastChart({ forecast }) {
  if (!forecast) return null;
  const historyDates = forecast.historical_dates || [];
  const historyPrices = forecast.historical_prices || [];
  const futureDates = forecast.future_dates || [];
  const joinPadding = Array(Math.max(0, historyPrices.length - 1)).fill(null);
  const labels = [...historyDates, ...futureDates];
  const current = historyPrices.at(-1);
  const averagePrices = midpointPrices(forecast.lower_prices, forecast.upper_prices);
  const data = {
    labels,
    datasets: [
      {
        label: 'Historical close',
        data: [...historyPrices, ...Array(futureDates.length).fill(null)],
        borderColor: '#94a3b8',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.2,
      },
      {
        label: 'Average 7-day estimate',
        data: [...joinPadding, current, ...averagePrices],
        borderColor: '#38bdf8',
        backgroundColor: '#38bdf8',
        borderWidth: 3,
        pointRadius: 4,
        pointBackgroundColor: '#38bdf8',
        pointBorderColor: '#070a12',
        pointBorderWidth: 2,
        pointHoverRadius: 7,
        tension: 0.28,
      },
    ],
  };
  const currencySymbol = forecast.currency_symbol || '$';
  const isPence = currencySymbol === 'p' || currencySymbol === 'GBp';
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { intersect: false, mode: 'index' },
    plugins: {
      legend: {
        position: 'top',
        align: 'end',
        labels: {
          color: '#94a3b8',
          usePointStyle: true,
          pointStyle: 'circle',
          boxWidth: 7,
          font: { family: 'Inter, sans-serif', size: 11, weight: '500' },
          padding: 16,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(11, 15, 25, 0.95)',
        borderColor: 'rgba(255, 255, 255, 0.12)',
        borderWidth: 1,
        padding: 12,
        cornerRadius: 8,
        titleColor: '#f8fafc',
        titleFont: { family: 'Inter, sans-serif', size: 12, weight: '600' },
        bodyColor: '#cbd5e1',
        bodyFont: { family: 'JetBrains Mono, monospace', size: 12 },
        callbacks: {
          label: (context) => {
            const raw = Number(context.raw);
            if (!Number.isFinite(raw)) return null;
            const formatted = isPence ? `${raw.toFixed(1)}p` : `$${raw.toFixed(2)}`;
            return ` ${context.dataset.label}: ${formatted}`;
          },
        },
      },
    },
    scales: {
      x: {
        ticks: {
          color: '#94a3b8',
          maxTicksLimit: 9,
          maxRotation: 0,
          font: { family: 'JetBrains Mono, monospace', size: 10 },
        },
        grid: { display: false },
        border: { color: 'rgba(255, 255, 255, 0.12)' },
      },
      y: {
        ticks: {
          color: '#94a3b8',
          callback: (value) => (isPence ? `${value}p` : `$${value}`),
          font: { family: 'JetBrains Mono, monospace', size: 10 },
        },
        grid: { color: 'rgba(255, 255, 255, 0.08)' },
        border: { color: 'rgba(255, 255, 255, 0.12)' },
      },
    },
  };
  return (
    <div className="forecast-chart" aria-label="Historical prices and seven-day forecast chart">
      <LazyLineChart data={data} options={options} />
    </div>
  );
}
