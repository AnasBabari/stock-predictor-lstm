export const PROMOTION_POLICY_VERSION = 'v2';

export const PROMOTION_THRESHOLDS = Object.freeze({
  maximumRelativeMae: 1.0,
  maximumRelativeRmse: 1.0,
  minimumWinningFolds: 4,
  maximumFoldRelativeRmse: 1.35,
  minimumEvaluationRows: 60,
  maximumVolatilityMultiple: 4,
  volatilityPercentile: 0.995,
  recentVolatilityWindow: 60,
  minimumBalancedAccuracy: 0.55,
  maximumRelativeBrier: 1.0,
  minimumDirectionFoldAccuracy: 0.5,
});

function finite(value) {
  return value != null && Number.isFinite(Number(value));
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

/**
 * Independent promotion decision for a single forecast horizon.
 */
export function evaluateHorizonPromotion({
  horizon,
  horizonMetrics,
  evaluation,
  closingPrices,
  predictedCumulativeReturn,
  thresholds = PROMOTION_THRESHOLDS,
}) {
  const h = Math.max(1, Math.round(Number(horizon) || 1));
  const reasons = [];
  const checks = { horizon: h, applicable: true };

  if (!horizonMetrics || typeof horizonMetrics !== 'object') {
    return {
      horizon: h,
      passed: false,
      state: 'unavailable',
      reasons: ['Evaluation metrics are unavailable.'],
      checks: { ...checks, complete: false },
    };
  }

  const maeFinite = finite(horizonMetrics.mae) && finite(horizonMetrics.rmse);
  const relativeFinite = finite(horizonMetrics.relative_mae) && finite(horizonMetrics.relative_rmse);
  checks.finiteMetrics = maeFinite && relativeFinite;
  if (!maeFinite || !relativeFinite) {
    reasons.push('Metrics contain non-finite values.');
  }

  const relRmse = Number(horizonMetrics.relative_rmse);
  const relMae = Number(horizonMetrics.relative_mae);
  checks.relativeRmse = relRmse;
  checks.relativeMae = relMae;

  const rows = Number(horizonMetrics.rows ?? horizonMetrics.evaluation_rows ?? 0);
  checks.rows = rows;
  if (rows < Number(thresholds.minimumEvaluationRows)) {
    reasons.push('The horizon has too few evaluated observations.');
  }

  if (relRmse >= Number(thresholds.maximumRelativeRmse)) {
    reasons.push('Relative RMSE did not beat persistence.');
  }
  if (relMae > Number(thresholds.maximumRelativeMae)) {
    reasons.push('Relative MAE did not beat persistence.');
  }

  // Fold stability checks (if multi-fold research split)
  const foldSummaries = Array.isArray(evaluation?.fold_summaries) ? evaluation.fold_summaries : [];
  const totalFolds = Math.round(Number(evaluation?.total_folds) || 0);
  let foldCheckPassed = true;

  if (totalFolds > 1) {
    const validFolds = foldSummaries.every(
      (summary) => summary && finite(summary.relative_rmse ?? summary.metrics?.relative_rmse)
    );
    if (!validFolds || foldSummaries.length !== totalFolds) {
      reasons.push('Fold evaluation is incomplete.');
      foldCheckPassed = false;
    } else {
      const foldRmses = foldSummaries.map(
        (summary) => Number(summary.relative_rmse ?? summary.metrics?.relative_rmse)
      );
      const winningFolds = foldRmses.filter((val) => val < 1.0).length;
      const maxFoldRelativeRmse = Math.max(...foldRmses);
      checks.winningFolds = winningFolds;
      checks.maxFoldRelativeRmse = maxFoldRelativeRmse;

      if (winningFolds < Number(thresholds.minimumWinningFolds)) {
        reasons.push(`Model won only ${winningFolds} of ${totalFolds} folds.`);
        foldCheckPassed = false;
      }
      if (maxFoldRelativeRmse > Number(thresholds.maximumFoldRelativeRmse)) {
        reasons.push('A validation fold exceeded the allowed degradation threshold.');
        foldCheckPassed = false;
      }
    }
  }

  // Volatility plausibility check
  if (closingPrices && predictedCumulativeReturn != null) {
    const volatility = volatilityAssessment({
      closingPrices,
      horizon: h,
      predictedCumulativeReturn,
      thresholds,
    });
    checks.volatility = volatility;
    if (!volatility.plausible) {
      reasons.push('The forecast exceeded its historically observed volatility range.');
    }
  }

  const passed = reasons.length === 0;
  let state = 'experimental';
  if (passed) {
    state = 'promoted';
  } else if (relRmse < 1.0 && foldCheckPassed === false) {
    state = 'candidate';
  } else if (!maeFinite || !relativeFinite) {
    state = 'unavailable';
  }

  return {
    horizon: h,
    passed,
    state,
    reasons,
    checks,
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
  const checks = { applicable: true, policy_version: PROMOTION_POLICY_VERSION };
  if (forecastType === 'direction') {
    return evaluateDirectionPromotion({ metrics, evaluation, thresholds });
  }
  if (forecastType !== 'price') {
    return {
      promoted: true,
      state: 'promoted',
      applicable: false,
      reasons: [],
      checks: { applicable: false },
      promoted_horizons: [],
      best_validated_horizon: null,
    };
  }

  const h = Math.max(1, Math.round(Number(horizon) || 1));
  if (!metrics || typeof metrics !== 'object') {
    return {
      promoted: false,
      state: 'unavailable',
      applicable: true,
      reasons: ['Evaluation is incomplete.'],
      checks: { ...checks, complete: false },
      promoted_horizons: [],
      best_validated_horizon: null,
    };
  }

  const topMaeFinite = finite(metrics.mae) && finite(metrics.rmse);
  const topRelFinite = finite(metrics.relative_mae) && finite(metrics.relative_rmse);
  if (!topMaeFinite || !topRelFinite) {
    return {
      promoted: false,
      state: 'unavailable',
      applicable: true,
      reasons: ['Metrics contain non-finite values.'],
      checks: { ...checks, finiteMetrics: false },
      promoted_horizons: [],
      best_validated_horizon: null,
    };
  }

  const perHorizon = Array.isArray(metrics.per_horizon) ? metrics.per_horizon : [];
  const perHorizonDecisions = perHorizon.map((entry) => {
    const dec = evaluateHorizonPromotion({
      horizon: Number(entry.horizon),
      horizonMetrics: entry,
      evaluation,
      closingPrices,
      predictedCumulativeReturn: Number(entry.horizon) === h ? predictedCumulativeReturn : null,
      thresholds,
    });
    entry.promotion = dec;
    return dec;
  });

  const promotedHorizons = perHorizonDecisions
    .filter((d) => d.passed)
    .map((d) => d.horizon);

  // Auto champion ranking across promoted horizons:
  // 1. Lowest relative RMSE
  // 2. Lowest relative MAE
  // 3. Horizon order
  const bestValidatedHorizon = promotedHorizons.length > 0
    ? [...perHorizon]
        .filter((entry) => promotedHorizons.includes(Number(entry.horizon)))
        .sort((a, b) => {
          const rmseDiff = Number(a.relative_rmse) - Number(b.relative_rmse);
          if (Math.abs(rmseDiff) > 1e-6) return rmseDiff;
          return Number(a.relative_mae) - Number(b.relative_mae);
        })[0]?.horizon ?? promotedHorizons[0]
    : null;

  // Evaluate requested horizon independently
  const selectedMetric = perHorizon.find((entry) => Number(entry.horizon) === h) || metrics;
  const requestedDecision = evaluateHorizonPromotion({
    horizon: h,
    horizonMetrics: selectedMetric,
    evaluation,
    closingPrices,
    predictedCumulativeReturn,
    thresholds,
  });

  return {
    promoted: requestedDecision.passed,
    state: requestedDecision.state,
    applicable: true,
    reasons: requestedDecision.reasons,
    checks: { ...checks, ...requestedDecision.checks },
    per_horizon_decisions: perHorizonDecisions,
    promoted_horizons: promotedHorizons,
    best_validated_horizon: bestValidatedHorizon,
    requested_horizon: h,
  };
}

export function buildPersistenceForecast(closingPrices, days) {
  const latest = Number((closingPrices || []).at(-1));
  if (!finite(latest) || latest <= 0) throw new Error('Persistence forecast requires positive price history.');
  return Array.from({ length: Math.max(1, Math.round(Number(days) || 1)) }, () => latest);
}

/**
 * User-facing promotion status descriptor.
 *
 * CRITICAL PRODUCT CONTRACT:
 * - The decision is ALWAYS 'model' (alpha: 1) for all successfully generated model forecasts.
 * - Promotion status controls the badge and scientific claim, NEVER overwriting the chart series with persistence.
 */
export function describePromotionState(promotion) {
  if (!promotion || typeof promotion !== 'object') {
    return {
      state: 'unavailable',
      decision: 'model',
      alpha: 1,
      label: 'Validation status could not be verified.',
    };
  }
  if (promotion.applicable === false) {
    return {
      state: 'unavailable',
      decision: 'model',
      alpha: 1,
      label: 'No validation gate configured for this response type.',
    };
  }
  if (promotion.promoted || promotion.state === 'promoted') {
    return {
      state: 'promoted',
      decision: 'model',
      alpha: 1,
      label: 'Validated against persistence on held-out evaluation.',
    };
  }
  if (promotion.state === 'candidate') {
    return {
      state: 'candidate',
      decision: 'model',
      alpha: 1,
      label: 'Competitive with benchmark but promotion evidence is incomplete.',
    };
  }
  if (promotion.state === 'unavailable') {
    return {
      state: 'unavailable',
      decision: 'model',
      alpha: 1,
      label: 'Validation unavailable: evaluation could not be completed.',
    };
  }
  return {
    state: 'experimental',
    decision: 'model',
    alpha: 1,
    label: 'Model forecast shown for research; validation gates were not met.',
  };
}

export function evaluateDirectionPromotion({ metrics, evaluation, thresholds = PROMOTION_THRESHOLDS }) {
  const checks = { applicable: true, policy_version: PROMOTION_POLICY_VERSION };
  const reasons = [];

  if (!metrics || typeof metrics !== 'object') {
    reasons.push('Evaluation is incomplete.');
    return {
      promoted: false,
      state: 'unavailable',
      applicable: true,
      reasons,
      checks,
      promoted_horizons: [],
      best_validated_horizon: null,
    };
  }

  const macroBalancedAccuracy = Number(metrics.macro_balanced_accuracy);
  const macroF1 = Number(metrics.macro_f1);
  const brierSkill = metrics.brier_skill == null ? null : Number(metrics.brier_skill);
  const logLoss = Number(metrics.log_loss);
  const rowCount = Number(
    metrics.evaluation_origins ?? metrics.evaluation_rows ?? 0
  );
  checks.macroBalancedAccuracy = macroBalancedAccuracy;
  checks.macroF1 = macroF1;
  checks.brierSkill = brierSkill;
  checks.logLoss = logLoss;
  checks.evaluationOrigins = rowCount;

  const metricsFinite = [macroBalancedAccuracy, macroF1, logLoss].every(finite) &&
    (brierSkill == null || finite(brierSkill));
  checks.finiteMetrics = metricsFinite;
  if (!metricsFinite) {
    reasons.push('Metrics contain non-finite values.');
  } else {
    if (macroBalancedAccuracy < Number(thresholds.minimumBalancedAccuracy)) {
      reasons.push('Macro balanced accuracy did not clear the minimum requirement.');
    }
    if (brierSkill == null || brierSkill <= 0) {
      reasons.push('Multiclass Brier skill did not beat the pre-evaluation base-rate baseline.');
    }
    if (!(logLoss > 0)) {
      reasons.push('Log loss is not positive; evaluation is degenerate.');
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
      (summary) => summary && finite(summary.macro_balanced_accuracy ?? summary.metrics?.macro_balanced_accuracy) &&
        finite(summary.multiclass_brier ?? summary.metrics?.multiclass_brier)
    );
    if (!validFolds) reasons.push('Evaluation is incomplete.');
    else {
      const foldAccuracies = foldSummaries.map(
        (summary) => Number(summary.macro_balanced_accuracy ?? summary.metrics?.macro_balanced_accuracy)
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

  const passed = reasons.length === 0;
  return {
    promoted: passed,
    state: passed ? 'promoted' : 'experimental',
    applicable: true,
    reasons,
    checks,
    promoted_horizons: passed ? [1] : [],
    best_validated_horizon: passed ? 1 : null,
  };
}
