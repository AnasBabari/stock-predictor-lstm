import React, { useMemo, useRef, useCallback, lazy, Suspense } from 'react';

const LazyLineChart = lazy(() => import('./LazyLineChart'));

const TIMEFRAMES = [
  { label: '1W', days: 5 },
  { label: '1M', days: 21 },
  { label: '3M', days: 63 },
  { label: '6M', days: 126 },
  { label: '1Y', days: 252 },
];

export default function StockChart({
  stockData,
  daysView,
  setDaysView,
  theme,
  onAddWatchlist,
  onToast,
}) {
  const chartRef = useRef(null);
  const isDark = theme === 'dark';

  const chartData = useMemo(() => {
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

  const handleExportPng = useCallback(() => {
    if (!chartRef.current) return;
    const link = document.createElement('a');
    link.download = `${stockData?.ticker || 'chart'}_forecast.png`;
    link.href = chartRef.current.toBase64Image();
    link.click();
    onToast('success', 'Chart exported as PNG');
  }, [stockData, onToast]);

  const handleExportCsv = useCallback(() => {
    if (!stockData) return;
    const d = stockData;
    let csv = 'Date,Price,Type\n';

    d.historical_dates.forEach((dt, i) => {
      csv += `${dt},${d.historical_prices[i].toFixed(2)},Historical\n`;
    });
    d.future_dates.forEach((dt, i) => {
      csv += `${dt},${d.predicted_prices[i].toFixed(2)},Predicted\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = `${d.ticker}_forecast.csv`;
    link.href = url;
    link.click();
    setTimeout(() => {
      URL.revokeObjectURL(url);
    }, 500);
    onToast('success', 'Data exported as CSV');
  }, [stockData, onToast]);

  if (!stockData || !chartData) return null;

  return (
    <section id="chartContainer" className="chart-section">
      <div className="chart-header">
        <h2 id="chartTitle">{stockData.ticker} — Historical vs Predicted</h2>
        <div className="chart-actions">
          <button
            type="button"
            className="action-chip"
            onClick={handleExportPng}
            title="Export chart as PNG"
          >
            <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
              <path
                fillRule="evenodd"
                d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z"
              />
            </svg>
            PNG
          </button>
          <button
            type="button"
            className="action-chip"
            onClick={handleExportCsv}
            title="Export data as CSV"
          >
            <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
              <path
                fillRule="evenodd"
                d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z"
              />
            </svg>
            CSV
          </button>
          <button
            type="button"
            className="action-chip accent"
            onClick={() => onAddWatchlist(stockData)}
            title="Add to watchlist"
          >
            <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
            Watch
          </button>
        </div>
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
          <LazyLineChart ref={chartRef} data={chartData} options={chartOptions} />
        </Suspense>
      </div>
    </section>
  );
}
