export function flatten(values) {
  return values.flatMap((row) => (Array.isArray(row) ? row : [row])).map(Number);
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
  const mapeTerms = actual.map((value, index) => value === 0 ? null : Math.abs(errors[index] / value));
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

export function classificationMetrics(actual, predicted, metricSource) {
  const labels = flatten(actual).map((value) => Number(value) > 0.5 ? 1 : 0);
  const probabilities = flatten(predicted).map((value) => Math.min(1, Math.max(0, Number(value))));
  const predictedLabels = probabilities.map((value) => value >= 0.5 ? 1 : 0);
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
    naive_baseline: labels.length ? Math.max(positives, labels.length - positives) / labels.length : 0,
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
