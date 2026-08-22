import {
  buildPersistenceForecast,
  dailyLogReturns,
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

function researchEvaluation(relativeRmses) {
  const foldSummaries = relativeRmses.map((relative_rmse, index) => ({ fold: index + 1, relative_rmse }));
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
  expect(result.reasons).toEqual([]);
});

test('rejects when relative MAE or RMSE do not beat persistence', () => {
  const base = {
    forecastType: 'price',
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  };
  const maeFailure = evaluatePromotion({ ...base, metrics: metricsFor(1.02, 0.8, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.9, 0.8)]) });
  expect(maeFailure.promoted).toBe(false);
  expect(maeFailure.reasons).toContain('Relative MAE did not beat persistence.');

  const rmseFailure = evaluatePromotion({ ...base, metrics: metricsFor(0.8, 1.04, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.7, 1.1)]) });
  expect(rmseFailure.promoted).toBe(false);
  expect(rmseFailure.reasons).toContain('Relative RMSE did not beat persistence.');
});

test('rejects when the model wins fewer than four of five folds', () => {
  const metrics = metricsFor(0.8, 0.85, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.7, 0.8)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 1.02, 1.1, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons).toContain('Model won only 3 of 5 folds.');
});

test('rejects when a research fold exceeds the allowed degradation threshold', () => {
  const metrics = metricsFor(0.8, 0.85, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.7, 0.8)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 1.3, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons).toContain('A research fold exceeded the allowed degradation threshold.');
});

test('rejects incomplete research evaluation', () => {
  const metrics = metricsFor(0.8, 0.85, [horizonEntry(7, 0.7, 0.8)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: { complete: false, completed_folds: 2, total_folds: 5, fold_summaries: [] },
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons).toContain('Evaluation is incomplete.');
});

test('rejects when the selected horizon has too few evaluated observations', () => {
  const metrics = metricsFor(0.8, 0.85, [horizonEntry(7, 0.7, 0.8, 40)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons).toContain('The selected horizon has too few evaluated observations.');
});

test('rejects non-finite metrics', () => {
  const metrics = metricsFor(NaN, 0.8, [horizonEntry(7, 0.7, 0.8)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons).toContain('Metrics contain non-finite values.');
});

test('rejects a learned forecast that exceeds the volatility plausibility range', () => {
  const metrics = metricsFor(0.8, 0.85, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.7, 0.8)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.9, closingPrices: quiet,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons).toContain('The learned forecast exceeded its historically observed volatility range.');
});

test('rejects an equally extreme negative forecast outside the volatility range', () => {
  const metrics = metricsFor(0.8, 0.85, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.7, 0.8)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: -0.9, closingPrices: quiet,
  });
  expect(result.promoted).toBe(false);
  expect(result.checks.volatility.exceedsMultiple).toBe(true);
  expect(result.reasons).toContain('The learned forecast exceeded its historically observed volatility range.');
});

test('rejects when the evaluation row count is missing or zero (fail closed)', () => {
  const missingRows = evaluatePromotion({
    forecastType: 'price',
      metrics: {
        metric_source: 'browser_walk_forward_out_of_fold',
        mae: 1, rmse: 1, relative_mae: 0.8, relative_rmse: 0.85,
        per_horizon: [horizonEntry(7, 0.7, 0.8, 0)],
      },
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(missingRows.promoted).toBe(false);
  expect(missingRows.reasons).toContain('The selected horizon has too few evaluated observations.');

  const zeroRows = evaluatePromotion({
    forecastType: 'price',
    metrics: { ...metricsFor(0.8, 0.85, [horizonEntry(7, 0.7, 0.8, 0)]), evaluation_rows: 0 },
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(zeroRows.promoted).toBe(false);
  expect(zeroRows.reasons).toContain('The selected horizon has too few evaluated observations.');
});

test('accepts fold summaries that nest metrics under a metrics key', () => {
  const evaluation = {
    complete: true, completed_folds: 5, total_folds: 5,
    fold_summaries: [0.8, 0.7, 0.9, 0.75, 0.85].map((relative_rmse, index) => ({
      fold: index + 1, metrics: { relative_rmse },
    })),
  };
  const result = evaluatePromotion({
    forecastType: 'price',
    metrics: metricsFor(0.8, 0.85, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.7, 0.8)]),
    evaluation,
    horizon: 7, predictedCumulativeReturn: 0.01, closingPrices: quiet,
  });
  expect(result.promoted).toBe(true);
  expect(result.checks.winningFolds).toBe(5);
});

test('creates a valid high-volatility forecast that stays inside the plausibility range', () => {
  const metrics = metricsFor(0.8, 0.85, [horizonEntry(1, 0.9, 0.9), horizonEntry(7, 0.7, 0.8)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.8, 0.7, 0.9, 0.75, 0.85]),
    horizon: 7, predictedCumulativeReturn: 0.08, closingPrices: volatile,
  });
  const volatility = result.checks.volatility;
  expect(volatility.exceedsMultiple).toBe(false);
  expect(result.promoted).toBe(true);
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
  expect(result.reasons).toContain('Macro balanced accuracy did not clear the minimum requirement.');
});

test('rejects a direction model whose Brier score fails the majority-class baseline', () => {
  const result = evaluatePromotion({
    forecastType: 'direction',
    metrics: directionMetrics(0.62, -0.1),
    evaluation: directionResearchEvaluation([0.61, 0.58, 0.63, 0.6, 0.62]),
    horizon: 7,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons).toContain('Multiclass Brier skill did not beat the pre-evaluation base-rate baseline.');
});

test('rejects a direction model with incomplete or unstable folds', () => {
  const incomplete = evaluatePromotion({
    forecastType: 'direction',
    metrics: directionMetrics(0.62, 0.15),
    evaluation: { complete: false, completed_folds: 2, total_folds: 5, fold_summaries: [] },
    horizon: 7,
  });
  expect(incomplete.promoted).toBe(false);
  expect(incomplete.reasons).toContain('Evaluation is incomplete.');

  const unstable = evaluatePromotion({
    forecastType: 'direction',
    metrics: directionMetrics(0.62, 0.15),
    evaluation: directionResearchEvaluation([0.61, 0.42, 0.63, 0.4, 0.62]),
    horizon: 7,
  });
  expect(unstable.promoted).toBe(false);
  expect(unstable.reasons).toContain('Model won only 3 of 5 folds.');
});

test('rejects a direction model with too few evaluated observations', () => {
  const result = evaluatePromotion({
    forecastType: 'direction',
    metrics: directionMetrics(0.62, 0.2, 0.55, 20),
    evaluation: directionResearchEvaluation([0.61, 0.58, 0.63, 0.6, 0.62]),
    horizon: 7,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons).toContain('The direction model has too few evaluated observations.');
});

test('asks for evaluated origins, not flattened labels, in the direction row gate', () => {
  // Flattened labels balloon past 60 while the true origin count is tiny.
  const v3 = (origins) => directionMetrics(0.62, 0.15, 0.9, origins);
  const manyLabels = evaluatePromotion({
    forecastType: 'direction',
    metrics: { ...v3(4), evaluation_origins: 4 },
    evaluation: directionResearchEvaluation([0.61, 0.58, 0.63, 0.6, 0.62]),
    horizon: 7,
  });
  expect(manyLabels.promoted).toBe(false);
  expect(manyLabels.reasons).toContain('The direction model has too few evaluated observations.');

  const enough = evaluatePromotion({
    forecastType: 'direction',
    metrics: v3(300),
    evaluation: directionResearchEvaluation([0.61, 0.58, 0.63, 0.6, 0.62]),
    horizon: 7,
  });
  expect(enough.promoted).toBe(true);
});

test('rejects a direction model with non-finite metrics', () => {
  const result = evaluatePromotion({
    forecastType: 'direction',
    metrics: directionMetrics(NaN, 0.2, 0.55, 300),
    evaluation: directionResearchEvaluation([0.61, 0.58, 0.63, 0.6, 0.62]),
    horizon: 7,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons).toContain('Metrics contain non-finite values.');
});

test('random-walk fixture: persistence wins and promotion fails', () => {
  const rng = (seed) => {
    let state = seed;
    return () => {
      state = (state * 1103515245 + 12345) % 2147483648;
      return state / 2147483648;
    };
  };
  const random = rng(42);
  const walk = pricesFromReturns(100, Array.from({ length: 600 }, () => (random() - 0.5) * 0.01));
  const horizon = 5;
  const actual = observedHorizonReturns(walk, horizon).map((value) => Math.log(1 + 0.02));
  const predicted = observedHorizonReturns(walk, horizon).map(() => 0);
  const errors = actual.map((value, index) => value - predicted[index]);
  const mae = errors.reduce((sum, value) => sum + Math.abs(value), 0) / errors.length;
  const baselineMae = actual.reduce((sum, value) => sum + Math.abs(value), 0) / actual.length;
  const rmse = Math.sqrt(errors.reduce((sum, value) => sum + value ** 2, 0) / errors.length);
  const baselineRmse = Math.sqrt(actual.reduce((sum, value) => sum + value ** 2, 0) / actual.length);
  const metrics = metricsFor(mae / baselineMae, rmse / baselineRmse, [horizonEntry(horizon, mae / baselineMae, rmse / baselineRmse)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([1.02, 0.99, 1.05, 1.0, 1.03]),
    horizon, predictedCumulativeReturn: 0.001, closingPrices: walk,
  });
  expect(result.promoted).toBe(false);
  expect(result.reasons.some((reason) => reason.includes('persistence'))).toBe(true);
});

test('linear drift fixture: drift-aligned model beats persistence', () => {
  const drift = pricesFromReturns(100, Array(600).fill(0.004));
  const horizon = 5;
  const actual = observedHorizonReturns(drift, horizon).map((value) => Math.log(1 + 0.02));
  const predicted = actual.map(() => 0.004 * horizon);
  const errors = actual.map((value, index) => value - predicted[index]);
  const mae = errors.reduce((sum, value) => sum + Math.abs(value), 0) / errors.length;
  const baselineMae = actual.reduce((sum, value) => sum + Math.abs(value), 0) / actual.length;
  const rmse = Math.sqrt(errors.reduce((sum, value) => sum + value ** 2, 0) / errors.length);
  const baselineRmse = Math.sqrt(actual.reduce((sum, value) => sum + value ** 2, 0) / actual.length);
  const metrics = metricsFor(mae / baselineMae, rmse / baselineRmse, [horizonEntry(horizon, mae / baselineMae, rmse / baselineRmse)]);
  const result = evaluatePromotion({
    forecastType: 'price', metrics,
    evaluation: researchEvaluation([0.3, 0.2, 0.4, 0.25, 0.35]),
    horizon, predictedCumulativeReturn: 0.02, closingPrices: drift,
  });
  expect(mae / baselineMae).toBeLessThan(0.98);
  expect(result.promoted).toBe(true);
});

test('volatility helpers behave on deterministic fixtures', () => {
  const prices = pricesFromReturns(100, [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01]);
  const returns = dailyLogReturns(prices);
  expect(returns).toHaveLength(10);
  expect(standardDeviation([1, 3])).toBeCloseTo(Math.SQRT2);
  const horizonReturns = observedHorizonReturns(prices, 2);
  expect(horizonReturns).toHaveLength(9);
  expect(percentile([1, 2, 3, 4], 0.5)).toBeCloseTo(2.5);
});

test('persistence forecast repeats the latest close', () => {
  const forecast = buildPersistenceForecast([10, 12, 11.5], 7);
  expect(forecast).toHaveLength(7);
  expect(forecast.every((value) => value === 11.5)).toBe(true);
  expect(() => buildPersistenceForecast([], 7)).toThrow(/positive/);
});

test('volatility assessment reports limits for a horizon', () => {
  const assessment = volatilityAssessment({ closingPrices: quiet, horizon: 7, predictedCumulativeReturn: 0.01 });
  expect(assessment.dailyVol).toBeGreaterThan(0);
  expect(assessment.horizonVolatility).toBeCloseTo(assessment.dailyVol * Math.sqrt(7), 10);
  expect(assessment.annualizedVol).toBeCloseTo(assessment.dailyVol * Math.sqrt(252), 10);
  expect(assessment.plausible).toBe(true);
});
