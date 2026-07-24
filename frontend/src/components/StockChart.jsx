import React, { useMemo, useCallback, lazy, Suspense, forwardRef } from 'react';

const LazyLineChart = lazy(() => import('./LazyLineChart'));

const TIMEFRAMES = [
  { label: '1W', days: 5 },
  { label: '1M', days: 21 },
  { label: '3M', days: 63 },
  { label: '6M', days: 126 },
  { label: '1Y', days: 252 },
];

const StockChart = forwardRef(function StockChart(
  {
  stockData,
  forecastType,
  daysView,
  setDaysView,
  theme,
  },
  ref
) {
  const isDark = theme === 'dark';

  const chartData = useMemo(() => {
    if (forecastType !== 'price') {
      return null;
    }

    if (!stockData || !stockData.historical_prices || !stockData.predicted_prices) {
      return null;
    }

    const total = stockData.historical_prices.length;
    const sliceIdx = Math.max(0, total - daysView);
    const sliceDates = stockData.historical_dates.slice(sliceIdx);
    const slicePrices = stockData.historical_prices.slice(sliceIdx);

    const allDates = [...sliceDates, ...stockData.future_dates];

    const historicalPadded = [
      ...slicePrices,
      ...Array(stockData.future_dates.length).fill(null),
    ];
    const predictedPadded = [
      ...Array(slicePrices.length - 1).fill(null),
      slicePrices[slicePrices.length - 1],
      ...stockData.predicted_prices,
    ];

    const histColor = isDark ? '#58a6ff' : '#3b82f6';
    const predColor = isDark ? '#00f5a0' : '#10b981';

    return {
      labels: allDates,
      datasets: [
        {
          label: 'Historical Price',
          data: historicalPadded,
          borderColor: histColor,
          backgroundColor: (context) => {
            const ctx = context.chart.ctx;
            const grad = ctx.createLinearGradient(0, 0, 0, 400);
            grad.addColorStop(0, isDark ? 'rgba(88,166,255,0.12)' : 'rgba(59,130,246,0.08)');
            grad.addColorStop(1, 'transparent');
            return grad;
          },
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: histColor,
          tension: 0.35,
          fill: true,
          spanGaps: false,
        },
        {
          label: 'Predicted Price',
          data: predictedPadded,
          borderColor: predColor,
          backgroundColor: (context) => {
            const ctx = context.chart.ctx;
            const grad = ctx.createLinearGradient(0, 0, 0, 400);
            grad.addColorStop(0, isDark ? 'rgba(0,245,160,0.12)' : 'rgba(16,185,129,0.08)');
            grad.addColorStop(1, 'transparent');
            return grad;
          },
          borderWidth: 2.5,
          pointRadius: 4,
          pointBackgroundColor: predColor,
          pointHoverRadius: 6,
          borderDash: [6, 3],
          tension: 0.35,
          fill: true,
          spanGaps: false,
        },
      ],
    };
  }, [stockData, daysView, isDark]);

  const chartOptions = useMemo(() => {
    const gridColor = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.05)';
    const tickColor = isDark ? '#5a5a7a' : '#94a3b8';
    const tooltipBg = isDark ? '#0d0d1a' : '#ffffff';
    const tooltipTitle = isDark ? '#e8e8f0' : '#1e293b';
    const tooltipBody = isDark ? '#a0a0c0' : '#475569';
    const tooltipBorder = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
    const legendColor = isDark ? '#a0a0c0' : '#475569';

    return {
      responsive: true,
      maintainAspectRatio: true,
      interaction: { mode: 'nearest', intersect: false },
      plugins: {
        legend: {
          labels: {
            color: legendColor,
            font: { size: 12, family: 'Inter' },
            usePointStyle: true,
            padding: 20,
          },
        },
        tooltip: {
          backgroundColor: tooltipBg,
          titleColor: tooltipTitle,
          bodyColor: tooltipBody,
          borderColor: tooltipBorder,
          borderWidth: 1,
          cornerRadius: 10,
          padding: 12,
          titleFont: { family: 'Inter', weight: '600' },
          bodyFont: { family: 'Inter' },
          callbacks: {
            label: (ctx) => {
              const val = ctx.parsed.y;
              return val != null ? ` $${val.toFixed(2)}` : null;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: tickColor, maxTicksLimit: 10, font: { family: 'Inter', size: 11 } },
          grid: { color: gridColor },
        },
        y: {
          ticks: {
            color: tickColor,
            font: { family: 'Inter', size: 11 },
            callback: (v) => `$${v.toFixed(0)}`,
          },
          grid: { color: gridColor },
        },
      },
    };
  }, [isDark]);

  if (!stockData || !chartData) return null;

  return (
    <section id="chartContainer" className="chart-section">
      <div className="chart-header">
        <h2 id="chartTitle">{stockData.ticker} — Historical vs Predicted</h2>
      </div>
      <div className="timeframe-filters">
        {TIMEFRAMES.map((t) => (
          <button
            key={t.label}
            type="button"
            className={`time-btn ${daysView === t.days ? 'active' : ''}`}
            onClick={() => setDaysView(t.days)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="chart-canvas-wrap">
        <Suspense fallback={<div className="loading-text">Loading Chart...</div>}>
          <LazyLineChart ref={ref} data={chartData} options={chartOptions} />
        </Suspense>
      </div>
    </section>
  );
});

export default StockChart;
