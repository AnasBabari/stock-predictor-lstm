/**
 * Pure price-chart dataset assembly.
 *
 * Core Product Contract:
 * - The primary series is ALWAYS the Model Forecast (solid bright teal/green line).
 * - Promotion status controls the badge and scientific claim, never mutates the forecast into a flat line.
 * - Persistence Benchmark is optional and hidden by default, drawn as a thin dashed grey line only when showBenchmark is enabled.
 */

const COLORS = {
  histDark: '#58a6ff',
  histLight: '#3b82f6',
  modelDark: '#00f5a0',
  modelLight: '#10b981',
  benchDark: '#8b93a7',
  benchLight: '#64748b',
  bandFillDark: 'rgba(0, 245, 160, 0.08)',
  bandFillLight: 'rgba(16, 185, 129, 0.08)',
};

function padForecast(values, leadNulls, anchorPrice) {
  return [
    ...Array(Math.max(0, leadNulls)).fill(null),
    ...(anchorPrice != null ? [anchorPrice] : []),
    ...(values || []),
  ];
}

export function buildPriceSeries(stockData, daysView, isDark, showBenchmark = false) {
  if (!stockData || !stockData.historical_prices || !stockData.predicted_prices) {
    return null;
  }
  if (!Array.isArray(stockData.historical_dates) || stockData.historical_dates.length === 0) {
    return null;
  }

  const total = stockData.historical_prices.length;
  const sliceIdx = Math.max(0, total - (daysView || total));
  const sliceDates = stockData.historical_dates.slice(sliceIdx);
  const slicePrices = stockData.historical_prices.slice(sliceIdx);
  const futureCount = Array.isArray(stockData.future_dates) ? stockData.future_dates.length : 0;
  if (futureCount === 0) return null;

  const allDates = [...sliceDates, ...stockData.future_dates];
  const historicalPadded = [...slicePrices, ...Array(futureCount).fill(null)];
  const lastClose = slicePrices[slicePrices.length - 1];
  const forecastLead = Math.max(0, slicePrices.length - 1);
  const forecastSplitIndex = slicePrices.length - 1;

  const modelColor = isDark ? COLORS.modelDark : COLORS.modelLight;
  const benchColor = isDark ? COLORS.benchDark : COLORS.benchLight;
  const histColor = isDark ? COLORS.histDark : COLORS.histLight;

  const datasets = [
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
      tension: 0.3,
      fill: true,
      spanGaps: false,
    },
    {
      label: 'Model Forecast',
      data: padForecast(stockData.predicted_prices, forecastLead, lastClose),
      borderColor: modelColor,
      backgroundColor: (context) => {
        const ctx = context.chart.ctx;
        const grad = ctx.createLinearGradient(0, 0, 0, 400);
        grad.addColorStop(0, isDark ? 'rgba(0,245,160,0.12)' : 'rgba(16,185,129,0.08)');
        grad.addColorStop(1, 'transparent');
        return grad;
      },
      borderWidth: 2.5,
      pointRadius: 4,
      pointBackgroundColor: modelColor,
      pointHoverRadius: 6,
      borderDash: [],
      tension: 0.3,
      fill: true,
      spanGaps: false,
    },
  ];

  // Optional Forecast Error Range Band
  const errorBand = stockData.historical_error_band;
  if (errorBand && Array.isArray(errorBand.upper_prices) && Array.isArray(errorBand.lower_prices)) {
    datasets.push({
      label: '90% Empirical Error Range (Upper)',
      data: padForecast(errorBand.upper_prices, forecastLead, lastClose),
      borderColor: 'transparent',
      backgroundColor: isDark ? COLORS.bandFillDark : COLORS.bandFillLight,
      pointRadius: 0,
      fill: '+1', // Fill down to the lower band
      tension: 0.3,
      spanGaps: false,
    });
    datasets.push({
      label: '90% Empirical Error Range (Lower)',
      data: padForecast(errorBand.lower_prices, forecastLead, lastClose),
      borderColor: 'transparent',
      backgroundColor: 'transparent',
      pointRadius: 0,
      fill: false,
      tension: 0.3,
      spanGaps: false,
    });
  }

  // Optional Persistence Benchmark: shown only when explicitly requested
  if (showBenchmark && (stockData.persistence_forecast || stockData.benchmark?.prices)) {
    const benchPrices = stockData.persistence_forecast || stockData.benchmark?.prices;
    datasets.push({
      label: 'Persistence Benchmark',
      data: padForecast(benchPrices, forecastLead, lastClose),
      borderColor: benchColor,
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 2,
      pointBackgroundColor: benchColor,
      borderDash: [4, 4],
      tension: 0,
      fill: false,
      spanGaps: false,
    });
  }

  return {
    labels: allDates,
    datasets,
    forecastSplitIndex,
    lastClose,
    promoted: stockData.validation?.promoted === true || stockData.forecast_status?.state === 'promoted',
  };
}

export function directionStatusText(status) {
  if (!status) return '';
  return status.label || '';
}
