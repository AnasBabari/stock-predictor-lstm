export function flatten(values) {
  return values.flatMap((row) => (Array.isArray(row) ? row : [row])).map(Number);
}

// The direction baseline (majority class) must be derived from labels strictly
// before the evaluation window. This helper returns { label, rate } where
// label is the majority class (0/1) and rate the positive-class prevalence,
// both over the supplied (pre-evaluation) label rows.
export function directionMajority(targetRows) {
  const labels = flatten(targetRows).map((value) => (Number(value) > 0.5 ? 1 : 0));
  const positives = labels.filter((value) => value === 1).length;
  const rate = labels.length ? positives / labels.length : 0.5;
  return { label: rate >= 0.5 ? 1 : 0, rate };
}

export function regressionMetrics(actual, predicted, persistence, metricSource) {
  const errors = actual.map((value, index) => value - predicted[index]);
  const squared = errors.map((value) => value ** 2);
  const mae = errors.reduce((sum, value) => sum + Math.abs(value), 0) / actual.length;
  const mse = squared.reduce((sum, value) => sum + value, 0) / actual.length;
  const mean = actual.reduce((sum, value) => sum + value, 0) / actual.length;
  const ssTotal = actual.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  const baselineErrors = actual.map((value, index) => value - persistence[index]);
  const baselineMae = baselineErrors.reduce((sum, value) => sum + Math.abs(value), 0) / actual.length;
  const baselineRmse = Math.sqrt(baselineErrors.reduce((sum, value) => sum + value ** 2, 0) / actual.length);
  // MAPE over cumulative log returns explodes near zero denominators; the
  // exact-zero filter is not enough. Exclude |actual| below MAPE_EPSILON and
  // report the metric only when enough terms survive.
  const MAPE_EPSILON = 1e-4;
  const mapeTerms = actual.map((value, index) =>
    Math.abs(value) < MAPE_EPSILON ? null : Math.abs(errors[index] / value)
  );
  const finiteMape = mapeTerms.filter((value) => value !== null);
  return {
    metric_source: metricSource,
    metric_scope: metricSource === 'browser_walk_forward_out_of_fold'
      ? 'untouched_expanding_walk_forward_folds'
      : 'untouched_post_purge_holdout',
    mae,
    mse,
    rmse: Math.sqrt(mse),
    mape: finiteMape.length ? (finiteMape.reduce((sum, value) => sum + value, 0) / finiteMape.length) * 100 : null,
    r2: ssTotal === 0 ? 0 : 1 - squared.reduce((sum, value) => sum + value, 0) / ssTotal,
    relative_mae: baselineMae === 0 ? null : mae / baselineMae,
    relative_rmse: baselineRmse === 0 ? null : Math.sqrt(mse) / baselineRmse,
  };
}

export function classificationMetrics(actual, predicted, metricSource, majorityLabel) {
  const labels = flatten(actual).map((value) => Number(value) > 0.5 ? 1 : 0);
  const probabilities = flatten(predicted).map((value) => Math.min(1, Math.max(0, Number(value))));
  const predictedLabels = probabilities.map((value) => value >= 0.5 ? 1 : 0);
  const isMatrix = Array.isArray(actual) && Array.isArray(actual[0]);
  const evaluationOrigins = isMatrix ? actual.length : 1;
  let tp = 0; let tn = 0; let fp = 0; let fn = 0;
  labels.forEach((value, index) => {
    if (value === 1 && predictedLabels[index] === 1) tp += 1;
    else if (value === 0 && predictedLabels[index] === 0) tn += 1;
    else if (value === 0) fp += 1;
    else fn += 1;
  });
  const accuracy = labels.length ? (tp + tn) / labels.length : 0;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const downRecall = tn + fp ? tn / (tn + fp) : 0;
  const positives = labels.filter((value) => value === 1).length;
  const naiveBaseline = labels.length
    ? majorityLabel == null
      ? Math.max(positives, labels.length - positives) / labels.length
      : labels.filter((value) => value === majorityLabel).length / labels.length
    : 0;
  return {
    metric_source: metricSource,
    metric_scope: metricSource === 'browser_walk_forward_out_of_fold'
      ? 'untouched_expanding_walk_forward_folds'
      : 'untouched_post_purge_holdout',
    accuracy,
    directional_accuracy: accuracy,
    precision,
    recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    balanced_accuracy: (recall + downRecall) / 2,
    brier_score: labels.length
      ? probabilities.reduce((sum, value, index) => sum + (value - labels[index]) ** 2, 0) / labels.length
      : 0,
    naive_baseline: naiveBaseline,
    evaluation_origins: evaluationOrigins,
    evaluation_labels: labels.length,
    evaluation_rows: labels.length,
  };
}

export function directionalAccuracy(actualReturns, predictedReturns) {
  const actual = flatten(actualReturns);
  const predicted = flatten(predictedReturns);
  const pairs = actual.map((value, index) => [value, predicted[index]])
    .filter(([value, forecast]) => Number.isFinite(value) && Number.isFinite(forecast));
  if (!pairs.length) return null;
  const agreement = pairs.filter(([value, forecast]) =>
    (value > 0 && forecast > 0) || (value < 0 && forecast < 0) || (value === 0 && forecast === 0)).length;
  return agreement / pairs.length;
}

// Direction evidence split by forecast day: each matrix column is one horizon
// step ahead, reported with the same pooled/pre-evaluation majority label.
export function horizonClassificationMetrics(actualRows, predictedRows, metricSource, majorityLabel) {
  const steps = Math.max(1, actualRows?.[0]?.length || 1);
  const perHorizon = Array.from({ length: steps }, (_, step) => {
    const actual = actualRows.map((row) => [row[step]]);
    const predicted = predictedRows.map((row) => [row[step]]);
    const metrics = classificationMetrics(actual, predicted, metricSource, majorityLabel);
    return {
      horizon: step + 1,
      rows: actual.length,
      accuracy: metrics.accuracy,
      balanced_accuracy: metrics.balanced_accuracy,
      precision: metrics.precision,
      recall: metrics.recall,
      f1: metrics.f1,
      brier_score: metrics.brier_score,
      naive_baseline: metrics.naive_baseline,
    };
  });
  return {
    pooled: classificationMetrics(actualRows, predictedRows, metricSource, majorityLabel),
    per_horizon: perHorizon,
  };
}

export function horizonRegressionMetrics(actualRows, predictedRows, persistenceRows, horizon, metricSource) {
  const pooled = regressionMetrics(
    flatten(actualRows), flatten(predictedRows), flatten(persistenceRows), metricSource,
  );
  const perHorizon = Array.from({ length: Math.max(1, Math.round(Number(horizon) || 1)) }, (_, step) => {
    const actual = actualRows.map((row) => Number(row[step]));
    const predicted = predictedRows.map((row) => Number(row[step]));
    const persistence = persistenceRows.map((row) => Number(row[step]));
    const metrics = regressionMetrics(actual, predicted, persistence, metricSource);
    return {
      horizon: step + 1,
      rows: actual.length,
      mae: metrics.mae,
      rmse: metrics.rmse,
      mape: metrics.mape,
      relative_mae: metrics.relative_mae,
      relative_rmse: metrics.relative_rmse,
      directional_accuracy: directionalAccuracy(actual, predicted),
    };
  });
  return {
    pooled,
    per_horizon: perHorizon,
    directional_accuracy: directionalAccuracy(actualRows, predictedRows),
    evaluation_rows: actualRows.length,
  };
}

export function generateResearchSplits(sampleCount, {
  folds = 5,
  validationHorizon = 60,
  minTrainSamples = 300,
  purge = 29,
} = {}) {
  const firstValidation = sampleCount - folds * validationHorizon;
  if (firstValidation - purge < minTrainSamples) {
    throw new Error(`Research profile requires at least ${minTrainSamples + purge + folds * validationHorizon} sequence samples.`);
  }
  return Array.from({ length: folds }, (_, index) => {
    const validationStart = firstValidation + index * validationHorizon;
    return {
      fold: index + 1,
      trainEnd: validationStart - purge,
      validationStart,
      validationEnd: validationStart + validationHorizon,
    };
  });
}

export function median(values) {
  if (!values.length) return 1;
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : Math.round((ordered[middle - 1] + ordered[middle]) / 2);
}
