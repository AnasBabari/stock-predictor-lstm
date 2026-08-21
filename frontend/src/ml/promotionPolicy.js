export const PROMOTION_THRESHOLDS = Object.freeze({
  maximumRelativeMae: 0.98,
  maximumRelativeRmse: 0.98,
  minimumWinningFolds: 4,
  maximumFoldRelativeRmse: 1.25,
  minimumEvaluationRows: 60,
  maximumVolatilityMultiple: 4,
  volatilityPercentile: 0.995,
  recentVolatilityWindow: 60,
  hardHorizonRelativeCap: 1.0,
  minimumBalancedAccuracy: 0.55,
  maximumRelativeBrier: 0.98,
  minimumDirectionFoldAccuracy: 0.5,
});

function finite(value) {
  return Number.isFinite(Number(value));
}

export function dailyLogReturns(closingPrices) {
  const prices = (closingPrices || []).map(Number);
  const returns = [];
  for (let index = 1; index < prices.length; index += 1) {
    if (prices[index - 1] > 0 && prices[index] > 0) {
      returns.push(Math.log(prices[index] / prices[index - 1]));
    }
  }
  return returns;
}

export function standardDeviation(values) {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

export function percentile(values, quantile) {
  if (!values.length) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const position = quantile * (ordered.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower];
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower);
}

export function observedHorizonReturns(closingPrices, horizon) {
  const prices = (closingPrices || []).map(Number);
  const returns = [];
  for (let index = 0; index + horizon < prices.length; index += 1) {
    if (prices[index] > 0 && prices[index + horizon] > 0) {
      returns.push(Math.abs(Math.log(prices[index + horizon] / prices[index])));
    }
  }
  return returns;
}

export function volatilityAssessment({ closingPrices, horizon, predictedCumulativeReturn, thresholds = PROMOTION_THRESHOLDS }) {
  const h = Math.max(1, Math.round(Number(horizon) || 1));
  const window = Math.max(30, Math.round(Number(thresholds.recentVolatilityWindow) || 60));
  const returns = dailyLogReturns(closingPrices).slice(-window);
  const predicted = Number(predictedCumulativeReturn);
  const history = observedHorizonReturns(closingPrices, h);
  const dailyVol = standardDeviation(returns);
  const annualizedVol = dailyVol * Math.sqrt(252);
  const horizonVolatility = dailyVol * Math.sqrt(h);
  const multipleLimit = Number(thresholds.maximumVolatilityMultiple) * horizonVolatility;
  const percentileLimit = percentile(history, Number(thresholds.volatilityPercentile));
  const insufficientHistory = returns.length < 30 || history.length < 30;
  const exceedsMultiple = Number.isFinite(predicted) && Math.abs(predicted) > multipleLimit;
  const exceedsPercentile = Number.isFinite(predicted) && Math.abs(predicted) > percentileLimit;
  const plausible = insufficientHistory || !exceedsMultiple || !exceedsPercentile;
  return {
    dailyVol,
    annualizedVol,
    horizonVolatility,
    multipleLimit,
    percentileLimit,
    exceedsMultiple,
    exceedsPercentile,
    plausible,
    insufficientHistory,
    predictedCumulativeReturn: Number.isFinite(predicted) ? predicted : null,
    observations: { recent: returns.length, horizon: history.length },
  };
}

export function evaluatePromotion({
  forecastType = 'price',
  metrics,
  evaluation,
  horizon,
  predictedCumulativeReturn,
  closingPrices,
  thresholds = PROMOTION_THRESHOLDS,
} = {}) {
  const checks = { applicable: true };
  if (forecastType === 'direction') {
    return evaluateDirectionPromotion({ metrics, evaluation, thresholds });
  }
  if (forecastType !== 'price') {
    return { promoted: true, applicable: false, reasons: [], checks: { applicable: false } };
  }
  const reasons = [];
  const h = Math.max(1, Math.round(Number(horizon) || 1));

  if (!metrics || typeof metrics !== 'object') {
    reasons.push('Evaluation is incomplete.');
    return { promoted: false, applicable: true, reasons, checks };
  }

  const maeFinite = finite(metrics.mae) && finite(metrics.rmse);
  const relativeFinite = finite(metrics.relative_mae) && finite(metrics.relative_rmse);
  if (!maeFinite || !relativeFinite) {
    reasons.push('Metrics contain non-finite values.');
  }
  checks.finiteMetrics = maeFinite && relativeFinite;

  if (Number(metrics.relative_mae) >= Number(thresholds.maximumRelativeMae)) {
    reasons.push('Relative MAE did not beat persistence.');
  }
  if (Number(metrics.relative_rmse) >= Number(thresholds.maximumRelativeRmse)) {
    reasons.push('Relative RMSE did not beat persistence.');
  }
  checks.relativeMae = Number(metrics.relative_mae);
  checks.relativeRmse = Number(metrics.relative_rmse);

  const foldSummaries = Array.isArray(evaluation?.fold_summaries) ? evaluation.fold_summaries : [];
  const totalFolds = Math.round(Number(evaluation?.total_folds) || 0);
  const completeEvaluation = Boolean(evaluation?.complete) && totalFolds > 0 && (
    totalFolds === 1 ? foldSummaries.length <= 1 : foldSummaries.length === totalFolds
  );
  checks.completeEvaluation = completeEvaluation;
  checks.totalFolds = totalFolds;

  if (!completeEvaluation) {
    reasons.push('Evaluation is incomplete.');
  } else if (totalFolds > 1) {
    const validFolds = foldSummaries.every(
      (summary) => summary && finite(summary.relative_rmse ?? summary.metrics?.relative_rmse)
    );
    if (!validFolds) reasons.push('Evaluation is incomplete.');
    else {
      const foldRmses = foldSummaries.map(
        (summary) => Number(summary.relative_rmse ?? summary.metrics?.relative_rmse)
      );
      const winningFolds = foldRmses.filter((value) => value < 1).length;
      const maxFoldRelativeRmse = Math.max(...foldRmses);
      checks.winningFolds = winningFolds;
      checks.maxFoldRelativeRmse = maxFoldRelativeRmse;
      if (winningFolds < Number(thresholds.minimumWinningFolds)) {
        reasons.push(`Model won only ${winningFolds} of ${totalFolds} folds.`);
      }
      if (maxFoldRelativeRmse > Number(thresholds.maximumFoldRelativeRmse)) {
        reasons.push('A research fold exceeded the allowed degradation threshold.');
      }
    }
  }

  const perHorizon = Array.isArray(metrics.per_horizon) ? metrics.per_horizon : [];
  const selected = perHorizon.find((entry) => Number(entry.horizon) === h);
  const dayOne = perHorizon.find((entry) => Number(entry.horizon) === 1);
  const selectedRows = Number(selected?.rows ?? metrics.evaluation_rows ?? 0);
  checks.horizonRows = selectedRows;
  checks.evaluationRows = Number(metrics.evaluation_rows ?? selectedRows);

  if (selectedRows < Number(thresholds.minimumEvaluationRows)) {
    reasons.push('The selected horizon has too few evaluated observations.');
  }

  const horizonCap = Number(thresholds.hardHorizonRelativeCap);
  if (dayOne && finite(dayOne.relative_mae) && finite(dayOne.relative_rmse) &&
      (Number(dayOne.relative_mae) >= horizonCap || Number(dayOne.relative_rmse) >= horizonCap)) {
    reasons.push('The model did not beat persistence at the one-day horizon.');
  }
  if (selected && selected.horizon !== 1 && finite(selected.relative_mae) && finite(selected.relative_rmse) &&
      (Number(selected.relative_mae) >= horizonCap || Number(selected.relative_rmse) >= horizonCap)) {
    reasons.push('The model did not beat persistence at the selected horizon.');
  }
  checks.dayOneRelativeMae = dayOne ? Number(dayOne.relative_mae) : null;
  checks.dayOneRelativeRmse = dayOne ? Number(dayOne.relative_rmse) : null;
  checks.horizonRelativeMae = selected ? Number(selected.relative_mae) : null;
  checks.horizonRelativeRmse = selected ? Number(selected.relative_rmse) : null;

  const volatility = volatilityAssessment({
    closingPrices,
    horizon: h,
    predictedCumulativeReturn,
    thresholds,
  });
  checks.volatility = volatility;
  if (!volatility.plausible) {
    reasons.push('The learned forecast exceeded its historically observed volatility range.');
  }

  return { promoted: reasons.length === 0, applicable: true, reasons, checks };
}

export function buildPersistenceForecast(closingPrices, days) {
  const latest = Number((closingPrices || []).at(-1));
  if (!finite(latest) || latest <= 0) throw new Error('Persistence forecast requires positive price history.');
  return Array.from({ length: Math.max(1, Math.round(Number(days) || 1)) }, () => latest);
}

// User-facing status contract (overhaul slice 1). The decision path is what
// the UI presents as "the forecast"; the model path is always preserved
// separately so a safety fallback can never masquerade as an LSTM output.
// alpha is the blend weight toward the learned path: 1 = promoted as-is,
// 0 = pure baseline. Blending between 0 and 1 arrives in the per-horizon
// champion slice; until then the policy remains all-or-nothing.
export function describePromotionState(promotion) {
  if (!promotion || promotion.applicable === false) {
    return {
      state: 'promoted',
      decision: 'model',
      alpha: 1,
      label: 'Global-model style forecast (no promotion gate applies).',
    };
  }
  if (promotion.promoted) {
    return {
      state: 'promoted',
      decision: 'model',
      alpha: 1,
      label: 'Promoted: beat persistence on the untouched holdout.',
    };
  }
  return {
    state: 'experimental_no_demonstrated_edge',
    decision: 'persistence',
    alpha: 0,
    label:
      'Experimental model did not beat persistence on the untouched holdout; ' +
      'the forecast shown is the no-change baseline and the raw learned path is drawn for comparison.',
  };
}

export function evaluateDirectionPromotion({ metrics, evaluation, thresholds = PROMOTION_THRESHOLDS }) {
  const checks = { applicable: true };
  const reasons = [];

  if (!metrics || typeof metrics !== 'object') {
    reasons.push('Evaluation is incomplete.');
    return { promoted: false, applicable: true, reasons, checks };
  }

  const balancedAccuracy = Number(metrics.balanced_accuracy);
  const brier = Number(metrics.brier_score);
  const naiveRate = Number(metrics.naive_baseline);
  // Evidence is measured in forecast origins, not flattened horizon labels:
  // require at least minimumEvaluationRows origins so a single origin with many
  // horizons can never satisfy the minimum evidence gate by itself.
  const rowCount = Number(metrics.evaluation_origins ?? metrics.evaluation_rows ?? 0);
  checks.balancedAccuracy = balancedAccuracy;
  checks.brierScore = brier;
  checks.naiveBaseline = naiveRate;
  checks.evaluationRows = rowCount;
  checks.evaluationLabels = Number(metrics.evaluation_labels ?? metrics.evaluation_rows ?? 0);

  const metricsFinite = finite(balancedAccuracy) && finite(brier) && finite(naiveRate);
  checks.finiteMetrics = metricsFinite;
  if (!metricsFinite) {
    reasons.push('Metrics contain non-finite values.');
  } else {
    if (balancedAccuracy < Number(thresholds.minimumBalancedAccuracy)) {
      reasons.push('Balanced accuracy did not clear the minimum requirement.');
    }
    const baselineBrier = 1 - naiveRate;
    const relativeBrier = baselineBrier > 1e-9 ? brier / baselineBrier : null;
    checks.relativeBrier = relativeBrier;
    if (relativeBrier == null || relativeBrier >= Number(thresholds.maximumRelativeBrier)) {
      reasons.push('Brier score did not beat the majority-class baseline.');
    }
  }

  const foldSummaries = Array.isArray(evaluation?.fold_summaries) ? evaluation.fold_summaries : [];
  const totalFolds = Math.round(Number(evaluation?.total_folds) || 0);
  const completeEvaluation = Boolean(evaluation?.complete) && totalFolds > 0 && (
    totalFolds === 1 ? foldSummaries.length <= 1 : foldSummaries.length === totalFolds
  );
  checks.completeEvaluation = completeEvaluation;
  checks.totalFolds = totalFolds;

  if (!completeEvaluation) {
    reasons.push('Evaluation is incomplete.');
  } else if (totalFolds > 1) {
    const validFolds = foldSummaries.every(
      (summary) => summary && finite(summary.balanced_accuracy ?? summary.metrics?.balanced_accuracy) &&
        finite(summary.brier_score ?? summary.metrics?.brier_score)
    );
    if (!validFolds) reasons.push('Evaluation is incomplete.');
    else {
      const foldAccuracies = foldSummaries.map(
        (summary) => Number(summary.balanced_accuracy ?? summary.metrics?.balanced_accuracy)
      );
      const winningFolds = foldAccuracies.filter(
        (value) => value > Number(thresholds.minimumDirectionFoldAccuracy)
      ).length;
      checks.winningFolds = winningFolds;
      if (winningFolds < Number(thresholds.minimumWinningFolds)) {
        reasons.push(`Model won only ${winningFolds} of ${totalFolds} folds.`);
      }
    }
  }

  if (rowCount < Number(thresholds.minimumEvaluationRows)) {
    reasons.push('The direction model has too few evaluated observations.');
  }

  return { promoted: reasons.length === 0, applicable: true, reasons, checks };
}
