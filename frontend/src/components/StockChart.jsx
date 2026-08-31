import React, { useMemo, useState, lazy, Suspense, forwardRef } from 'react';
import { buildPriceSeries } from '../utils/chartSeries';
import { formatPrice } from '../utils/formatting';

const LazyLineChart = lazy(() => import('./LazyLineChart'));

const TIMEFRAMES = [
  { label: '1W', days: 5 },
  { label: '1M', days: 21 },
  { label: '3M', days: 63 },
  { label: '6M', days: 126 },
  { label: '1Y', days: 252 },
];

const forecastRegionPlugin = {
  id: 'forecastRegion',
  beforeDraw: (chart) => {
    const { ctx, chartArea, scales } = chart;
    const splitIdx = chart.data?.forecastSplitIndex;
    if (splitIdx == null || !scales.x || !chartArea) return;
    const xPos = scales.x.getPixelForValue(splitIdx);
    if (xPos == null || xPos < chartArea.left || xPos > chartArea.right) return;

    ctx.save();
    // Shaded future background
    ctx.fillStyle = chart.config.options?.isDark ? 'rgba(0, 245, 160, 0.03)' : 'rgba(16, 185, 129, 0.03)';
    ctx.fillRect(xPos, chartArea.top, chartArea.right - xPos, chartArea.bottom - chartArea.top);

    // Vertical dashed divider
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = chart.config.options?.isDark ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.2)';
    ctx.lineWidth = 1.5;
    ctx.moveTo(xPos, chartArea.top);
    ctx.lineTo(xPos, chartArea.bottom);
    ctx.stroke();

    // Label "Forecast"
    ctx.setLineDash([]);
    ctx.fillStyle = chart.config.options?.isDark ? '#00f5a0' : '#10b981';
    ctx.font = '600 10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('FORECAST', xPos + 6, chartArea.top + 14);
    ctx.restore();
  },
};

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
  const isVolatility = stockData?.metadata?.engine?.certified_head === 'volatility'
    || stockData?.volatility_cone != null;
  const isBaseline = stockData?.metadata?.engine?.baseline_fallback === true
    || stockData?.metadata?.engine?.execution_mode === 'baseline'
    || stockData?.forecast_status?.state === 'baseline';
  const hasReturnDistribution = stockData?.metadata?.engine?.certified_head === 'return_distribution'
    && Array.isArray(stockData?.predicted_prices);
  const [showBenchmark, setShowBenchmark] = useState(false);

  const chartData = useMemo(() => {
    if (forecastType !== 'price') {
      return null;
    }
    return buildPriceSeries(stockData, daysView, isDark, showBenchmark);
  }, [stockData, daysView, isDark, showBenchmark]);

  const yBounds = useMemo(() => {
    if (!chartData?.datasets) return {};
    let min = Infinity;
    let max = -Infinity;
    for (const ds of chartData.datasets) {
      for (const val of ds.data || []) {
        if (val != null && Number.isFinite(val)) {
          if (val < min) min = val;
          if (val > max) max = val;
        }
      }
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return {};
    const lastPrice = chartData.lastClose || (min + max) / 2;
    const padding = Math.max((max - min) * 0.10, lastPrice * 0.02, 0.005);
    return {
      suggestedMin: Math.max(0, min - padding),
      suggestedMax: max + padding,
    };
  }, [chartData]);

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
      maintainAspectRatio: false,
      isDark,
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
              return val != null ? ` ${ctx.dataset.label}: ${formatPrice(val)}` : null;
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
          ...yBounds,
          ticks: {
            color: tickColor,
            font: { family: 'Inter', size: 11 },
            callback: (v) => formatPrice(v),
          },
          grid: { color: gridColor },
        },
      },
    };
  }, [isDark, yBounds]);

  if (!stockData || !chartData) return null;

  return (
    <section id="chartContainer" className="chart-section">
      <div className="chart-header">
                <h2 id="chartTitle">
                  {stockData.ticker} — {hasReturnDistribution
                    ? 'Historical vs Certified Return Distribution'
                    : isVolatility ? `Historical vs ${isBaseline ? 'Causal Volatility Baseline' : 'Volatility Cone'}` : 'Historical vs Predicted'}
                </h2>
        <div className="chart-header-actions">
          {(!isVolatility || hasReturnDistribution) && <label className="benchmark-toggle" htmlFor="showBenchmarkCheck">
            <input
              id="showBenchmarkCheck"
              type="checkbox"
              checked={showBenchmark}
              onChange={(e) => setShowBenchmark(e.target.checked)}
            />
            <span>Show benchmark</span>
          </label>}
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
      <div className="chart-canvas-wrap" style={{ height: 'clamp(340px, 48vh, 520px)', position: 'relative' }}>
        <Suspense fallback={<div className="loading-text">Loading Chart...</div>}>
          <LazyLineChart ref={ref} data={chartData} options={chartOptions} plugins={[forecastRegionPlugin]} />
        </Suspense>
      </div>
    </section>
  );
});

export default StockChart;
