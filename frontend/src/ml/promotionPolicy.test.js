import { describe, expect, it, test } from 'vitest';
import {
  PROMOTION_POLICY_VERSION,
  buildPersistenceForecast,
  dailyLogReturns,
  describePromotionState,
  evaluateHorizonPromotion,
  evaluatePromotion,
  observedHorizonReturns,
  percentile,
  standardDeviation,
  volatilityAssessment,
} from './promotionPolicy';

function pricesFromReturns(start, dailyReturns) {
  const prices = [start];
  for (const daily of dailyReturns) prices.push(prices.at(-1) * Math.exp(daily));
  return prices;
}

function noisePrices(start, count, drift, vol) {
  let state = 42;
  const random = () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
  const prices = [start];
  for (let index = 0; index < count; index += 1) {
    prices.push(prices.at(-1) * Math.exp(drift + (random() - 0.5) * vol));
  }
  return prices;
}

function researchEvaluation(relativeRmses, horizons = [1, 5, 7]) {
  const foldSummaries = relativeRmses.map((relative_rmse, index) => ({
    fold: index + 1,
    per_horizon: horizons.map((horizon) => ({ horizon, relative_rmse, relative_mae: relative_rmse, rows: 60 })),
  }));
  return { complete: true, completed_folds: foldSummaries.length, total_folds: foldSummaries.length, fold_summaries: foldSummaries };
}

function directionResearchEvaluation(accuracies) {
  const foldSummaries = accuracies.map((macro_balanced_accuracy, index) => ({
    fold: index + 1,
    macro_balanced_accuracy,
    multiclass_brier: 0.5,
    brier_skill: 0.1,
    best_epoch: 3,
  }));
  return { complete: true, completed_folds: foldSummaries.length, total_folds: foldSummaries.length, fold_summaries: foldSummaries };
}

function directionMetrics(macroBalancedAccuracy, brierSkill, logLoss = 0.9, rows = 600, macroF1 = 0.6) {
  return {
    metric_source: 'browser_walk_forward_out_of_fold',
    direction_classes: ['down', 'neutral', 'up'],
    macro_balanced_accuracy: macroBalancedAccuracy,
    macro_f1: macroF1,
    multiclass_brier: 0.55,
    baseline_multiclass_brier: 0.62,
    brier_skill: brierSkill,
    log_loss: logLoss,
    expected_calibration_error: 0.04,
    evaluation_rows: rows,
    evaluation_origins: rows,
  };
}

function metricsFor(relativeMae, relativeRmse, perHorizon) {
  return {
    metric_source: 'browser_walk_forward_out_of_fold',
    mae: 1, rmse: 1, relative_mae: relativeMae, relative_rmse: relativeRmse,
    evaluation_rows: 600,
    per_horizon: perHorizon,
  };
}

function horizonEntry(horizon, relativeMae, relativeRmse, rows = 300) {
  return { horizon, rows, mae: 1, rmse: 1, relative_mae: relativeMae, relative_rmse: relativeRmse, directional_accuracy: 0.5 };
}

const quiet = noisePrices(100, 400, 0.0005, 0.01);
const volatile = noisePrices(100, 400, 0.001, 0.05);

test('promotes a model that beats persistence on pooled and horizon metrics', () => {
  const metrics = metricsFor(0.8, 0.85, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.7, 0.8)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(result.promoted).toBe(true);
  expect(result.state).toBe('promoted');
  expect(result.reasons).toEqual([]);
});

test('rejects when relative MAE or RMSE do not beat persistence', () => {
  const base = {
    forecastType: 'price',
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  };
  const maeFailure = evaluatePromotion({ ...base, metrics: metricsFor(1.02, 0.8, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 1.05, 0.8)]) });
  expect(maeFailure.promoted).toBe(false);
  expect(maeFailure.reasons).toContain('Relative MAE did not beat persistence.');

  const rmseFailure = evaluatePromotion({ ...base, metrics: metricsFor(0.8, 1.04, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.7, 1.1)]) });
  expect(rmseFailure.promoted).toBe(false);
  expect(rmseFailure.reasons).toContain('Relative RMSE did not beat persistence.');
});

test('1-day failure does not automatically reject a winning 5-day or 7-day model', () => {
  // 1d has Rel-RMSE 1.09 (fails), but 5d has Rel-RMSE 0.982 and Rel-MAE 0.97 (passes)
  const metrics = metricsFor(0.99, 1.01, [
    horizonEntry(1, 1.05, 1.09),
    horizonEntry(5, 0.97, 0.982),
    horizonEntry(7, 1.01, 1.011),
  ]);
  const result5d = evaluatePromotion({
    forecastType: 'price',
    metrics,
    evaluation: researchEvaluation([0.95, 0.96, 0.98, 0.97, 0.99]),
    horizon: 5,
    predictedCumulativeReturn: 0.01,
    closingPrices: quiet,
  });
  expect(result5d.promoted).toBe(true);
  expect(result5d.state).toBe('promoted');
  expect(result5d.promoted_horizons).toContain(5);
  expect(result5d.best_validated_horizon).toBeNull();

  const result7d = evaluatePromotion({
    forecastType: 'price',
    metrics,
    evaluation: researchEvaluation([0.95, 0.96, 0.98, 0.97, 0.99]),
    horizon: 7,
    predictedCumulativeReturn: 0.01,
    closingPrices: quiet,
  });
  expect(result7d.promoted).toBe(false);
  expect(result7d.state).toBe('experimental');
  expect(result7d.promoted_horizons).toEqual([5]);
  expect(result7d.best_validated_horizon).toBeNull();
});

test('describePromotionState always returns decision: model and alpha: 1', () => {
  const promotedDesc = describePromotionState({ promoted: true, state: 'promoted' });
  expect(promotedDesc.decision).toBe('model');
  expect(promotedDesc.alpha).toBe(1);
  expect(promotedDesc.state).toBe('promoted');

  const experimentalDesc = describePromotionState({ promoted: false, state: 'experimental' });
  expect(experimentalDesc.decision).toBe('model');
  expect(experimentalDesc.alpha).toBe(1);
  expect(experimentalDesc.state).toBe('experimental');

  const candidateDesc = describePromotionState({ promoted: false, state: 'candidate' });
  expect(candidateDesc.decision).toBe('model');
  expect(candidateDesc.alpha).toBe(1);
  expect(candidateDesc.state).toBe('candidate');

  const unavailableDesc = describePromotionState({ promoted: false, state: 'unavailable' });
  expect(unavailableDesc.decision).toBe('model');
  expect(unavailableDesc.state).toBe('unavailable');
});

test('rejects when the model wins fewer than four of five folds', () => {
  const metrics = metricsFor(0.8, 0.85, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.7, 0.8)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 1.02, 1.1, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(result.promoted).toBe(false);
  expect(result.state).toBe('candidate'); // Rel-RMSE < 1.0 on horizon but failed fold stability
  expect(result.reasons).toContain('Model won only 3 of 5 folds.');
});

test('rejects when a research fold exceeds the allowed degradation threshold', () => {
  const metrics = metricsFor(0.8, 0.85, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.7, 0.8)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 1.4, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons).toContain('A validation fold exceeded the allowed degradation threshold.');
});

test('uses independent fold stability evidence for each horizon', () => {
  const foldSummaries = [0.8, 0.82, 0.85, 0.88, 0.9].map((oneDay, index) => ({
    fold: index + 1,
    per_horizon: [
      { horizon: 1, relative_rmse: oneDay, relative_mae: oneDay, rows: 60 },
      { horizon: 5, relative_rmse: index === 0 ? 0.9 : 1.2, relative_mae: 1, rows: 60 },
    ],
  }));
  const evaluation = { complete: true, completed_folds: 5, total_folds: 5, fold_summaries: foldSummaries };
  const metrics = metricsFor(0.8, 0.8, [
    horizonEntry(1, 0.8, 0.8),
    horizonEntry(5, 0.8, 0.8),
  ]);
  const oneDay = evaluatePromotion({
    forecastType: 'price', metrics, evaluation, horizon: 1,
    predictedCumulativeReturn: 0.001, closingPrices: quiet,
  });
  const fiveDay = evaluatePromotion({
    forecastType: 'price', metrics, evaluation, horizon: 5,
    predictedCumulativeReturn: 0.002, closingPrices: quiet,
  });
  expect(oneDay.checks.winningFolds).toBe(5);
  expect(oneDay.promoted).toBe(true);
  expect(fiveDay.checks.winningFolds).toBe(1);
  expect(fiveDay.promoted).toBe(false);
});

test('rejects non-finite metrics', () => {
  const metrics = metricsFor(NaN, 0.8, [horizonEntry(7, 0.7, 0.8)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(result.promoted).toBe(false);
  expect(result.state).toBe('unavailable');
  expect(result.reasons).toContain('Metrics contain non-finite values.');
});

test('rejects when the selected horizon has too few evaluated observations', () => {
  const metrics = metricsFor(0.8, 0.85, [horizonEntry(7, 0.7, 0.8, 40)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons).toContain('The horizon has too few evaluated observations.');
});

test('rejects a learned forecast that exceeds the volatility plausibility range', () => {
  const metrics = metricsFor(0.8, 0.85, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.7, 0.8)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.9, closingPrices: quiet,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons).toContain('The forecast exceeded its historically observed volatility range.');
});

test('promotes a direction model that beats the majority-class baseline', () => {
  const result = evaluatePromotion({
    forecastType: 'direction',
    metrics: directionMetrics(0.62, 0.15),
    evaluation: directionResearchEvaluation([0.61, 0.58, 0.63, 0.6, 0.62]),
    horizon: 7,
  });
  expect(result.applicable).toBe(true);
  expect(result.promoted).toBe(true);
  expect(result.state).toBe('promoted');
  expect(result.reasons).toEqual([]);
});

test('rejects a direction model with low balanced accuracy', () => {
  const result = evaluatePromotion({
    forecastType: 'direction',
    metrics: directionMetrics(0.42, 0.05),
    evaluation: directionResearchEvaluation([0.55, 0.5, 0.52, 0.54, 0.53]),
    horizon: 7,
  });
  expect(result.promoted).toBe(false);
  expect(result.state).toBe('experimental');
  expect(result.reasons).toContain('Macro balanced accuracy did not clear the minimum requirement.');
});

test('volatility helpers and persistence forecast behave correctly', () => {
  const prices = pricesFromReturns(100, [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01]);
  const returns = dailyLogReturns(prices);
  expect(returns).toHaveLength(10);
  expect(standardDeviation([1, 3])).toBeCloseTo(Math.SQRT2);
  const forecast = buildPersistenceForecast([10, 12, 11.5], 7);
  expect(forecast).toHaveLength(7);
  expect(forecast.every((v) => v === 11.5)).toBe(true);
});
