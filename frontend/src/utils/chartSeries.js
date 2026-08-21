/**
 * Pure price-chart dataset assembly (overhaul slice 1).
 *
 * Contract:
 * - The decision path (what policy selected) is always drawn prominently and
 *   labelled truthfully — "Forecast" when promoted, "No-change baseline"
 *   when the learned model failed promotion.
 * - When the decision path is the baseline, the raw learned path is still
 *   drawn as a dashed diagnostic so a safety fallback can never masquerade
 *   as an LSTM output.
 */

const COLORS = {
  histDark: '#58a6ff',
  histLight: '#3b82f6',
  decisionDark: '#00f5a0',
  decisionLight: '#10b981',
  learnedDark: '#8b93a7',
  learnedLight: '#64748b',
};

function padDecision(values, leadNulls, anchorPrice) {
  return [
    ...Array(Math.max(0, leadNulls)).fill(null),
    ...(anchorPrice != null ? [anchorPrice] : []),
    ...values,
  ];
}

export function buildPriceSeries(stockData, daysView, isDark) {
  if (!stockData || !stockData.historical_prices || !stockData.predicted_prices) {
    return null;
  }
  if (!Array.isArray(stockData.historical_dates) || stockData.historical_dates.length === 0) {
    return null;
  }

  const total = stockData.historical_prices.length;
  const sliceIdx = Math.max(0, total - daysView);
  const sliceDates = stockData.historical_dates.slice(sliceIdx);
  const slicePrices = stockData.historical_prices.slice(sliceIdx);
  const futureCount = Array.isArray(stockData.future_dates) ? stockData.future_dates.length : 0;
  if (futureCount === 0) return null;

  const allDates = [...sliceDates, ...stockData.future_dates];
  const historicalPadded = [...slicePrices, ...Array(futureCount).fill(null)];
  const lastClose = slicePrices[slicePrices.length - 1];
  // Forecast lines start at the last historical close: one null per earlier
  // historical point, then the anchor.
  const forecastLead = Math.max(0, slicePrices.length - 1);

  const status = stockData.forecast_status || { state: 'promoted', decision: 'model', alpha: 1 };
  const promoted = status.decision !== 'persistence';
  const decisionLabel = promoted ? 'Predicted Price' : 'No-change baseline';
  const decisionColor = isDark ? COLORS.decisionDark : COLORS.decisionLight;
  const learnedColor = isDark ? COLORS.learnedDark : COLORS.learnedLight;

  const datasets = [
    {
      label: 'Historical Price',
      data: historicalPadded,
      borderColor: isDark ? COLORS.histDark : COLORS.histLight,
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
      pointHoverBackgroundColor: isDark ? COLORS.histDark : COLORS.histLight,
      tension: 0.35,
      fill: true,
      spanGaps: false,
    },
    {
      label: decisionLabel,
      data: padDecision(stockData.predicted_prices, forecastLead, lastClose),
      borderColor: decisionColor,
      backgroundColor: (context) => {
        if (!promoted) return 'transparent';
        const ctx = context.chart.ctx;
        const grad = ctx.createLinearGradient(0, 0, 0, 400);
        grad.addColorStop(0, isDark ? 'rgba(0,245,160,0.12)' : 'rgba(16,185,129,0.08)');
        grad.addColorStop(1, 'transparent');
        return grad;
      },
      borderWidth: 2.5,
      pointRadius: 4,
      pointBackgroundColor: decisionColor,
      pointHoverRadius: 6,
      borderDash: promoted ? [6, 3] : [4, 4],
      tension: promoted ? 0.35 : 0,
      fill: true,
      spanGaps: false,
    },
  ];

  // Learned diagnostic path: shown whenever it differs from the decision path
  // so users can see what the model actually produced before policy applied.
  let annotation = null;
  if (!promoted && Array.isArray(stockData.learned_prices) && stockData.learned_prices.length) {
    datasets.push({
      label: 'Learned model (not promoted)',
      data: padDecision(stockData.learned_prices, forecastLead, lastClose),
      borderColor: learnedColor,
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 2,
      pointBackgroundColor: learnedColor,
      borderDash: [2, 3],
      tension: 0.35,
      fill: false,
      spanGaps: false,
    });
    annotation =
      'Green path is the no-change baseline: the local model did not beat persistence on its holdout. Dashed grey shows what the model actually predicted.';
  } else if (!promoted) {
    annotation =
      'Green path is the no-change baseline: the local model did not beat persistence on its holdout.';
  }

  return { labels: allDates, datasets, annotation, promoted };
}

// Direction charts/panels need labelled trios too; kept alongside the price
// builder so status language stays in one place.
export function directionStatusText(status) {
  if (!status) return '';
  if (status.state === 'promoted') return 'Promoted: beat the pre-evaluation base rate.';
  return (
    'Experimental direction model did not demonstrate edge; probabilities shown are the pre-evaluation base rate.'
  );
}
