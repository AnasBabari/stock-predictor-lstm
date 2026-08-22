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

// ── Direction v2: multiclass metrics over [down, neutral, up] ───────────
// Spec §6.4: Brier skill vs the matched pre-evaluation baseline, macro
// balanced accuracy, macro F1, log loss, and calibration (ECE on max prob).

function oneHot(classIndex, classCount) {
  return Array.from({ length: classCount }, (_, i) => (i === classIndex ? 1 : 0));
}

export function classificationMetricsV3({
  actualClasses,
  probabilityRows,
  baselineProbabilities,
  classes,
  metricSource,
}) {
  const classCount = classes.length;
  const n = actualClasses.length;
  if (!n || probabilityRows.length !== n) {
    return { metric_source: metricSource, metric_scope: 'insufficient_data', evaluation_origins: 0, evaluation_labels: 0, evaluation_rows: 0 };
  }
  const rows = probabilityRows.map((row) => {
    const clipped = row.map((p) => Math.min(1, Math.max(0, Number(p))));
    const total = clipped.reduce((s, v) => s + v, 0);
    return total > 0 ? clipped.map((v) => v / total) : oneHot(1, classCount); // degenerate → neutral
  });
  const base = baselineProbabilities.map((p) => Math.min(1, Math.max(0, Number(p))));
  const baseSum = base.reduce((s, v) => s + v, 0);
  const baseline = baseSum > 0 ? base.map((v) => v / baseSum) : oneHot(1, classCount);

  // Multiclass Brier = mean over origins of Σ_c (p_c − y_c)².
  let brier = 0;
  let brierBase = 0;
  let logLoss = 0;
  let correct = 0;
  const confusion = Array.from({ length: classCount }, () => Array(classCount).fill(0));
  rows.forEach((row, i) => {
    const truth = oneHot(actualClasses[i], classCount);
    row.forEach((p, c) => {
      brier += (p - truth[c]) ** 2;
      brierBase += (baseline[c] - truth[c]) ** 2;
    });
    logLoss += -Math.log(Math.max(row[actualClasses[i]], 1e-12));
    const argmax = row.indexOf(Math.max(...row));
    confusion[actualClasses[i]][argmax] += 1;
    if (argmax === actualClasses[i]) correct += 1;
  });
  brier /= n;
  brierBase /= n;

  // Macro balanced accuracy = mean per-class recall (classes present in y).
  const perClassRecall = [];
  for (let c = 0; c < classCount; c += 1) {
    const support = confusion[c].reduce((s, v) => s + v, 0);
    if (support > 0) perClassRecall.push(confusion[c][c] / support);
  }
  const macroBalancedAccuracy = perClassRecall.length
    ? perClassRecall.reduce((s, v) => s + v, 0) / perClassRecall.length
    : 0;

  // Macro F1.
  const f1s = [];
  for (let c = 0; c < classCount; c += 1) {
    let tp = 0; let fp = 0; let fn = 0;
    rows.forEach((row, i) => {
      const predC = row.indexOf(Math.max(...row));
      if (predC === c && actualClasses[i] === c) tp += 1;
      else if (predC === c) fp += 1;
      else if (actualClasses[i] === c) fn += 1;
    });
    const denom = 2 * tp + fp + fn;
    if (denom > 0) f1s.push((2 * tp) / denom);
  }
  const macroF1 = f1s.length ? f1s.reduce((s, v) => s + v, 0) / f1s.length : 0;

  // ECE: 10 equal-width bins on max predicted probability.
  const eceBins = Array.from({ length: 10 }, () => ({ confidence: 0, correct: 0, count: 0 }));
  rows.forEach((row, i) => {
    const confidence = Math.max(...row);
    const bin = Math.min(9, Math.floor(confidence * 10));
    eceBins[bin].confidence += confidence;
    eceBins[bin].correct += row.indexOf(confidence) === actualClasses[i] ? 1 : 0;
    eceBins[bin].count += 1;
  });
  const totalBins = eceBins.reduce((s, b) => s + b.count, 0);
  const expectedCalibrationError = totalBins
    ? eceBins.reduce((s, b) => (b.count ? s + (b.count / totalBins) * Math.abs(b.correct / b.count - b.confidence / b.count) : s), 0)
    : 0;

  // Baseline argmax label distribution (for disclosure).
  const baselineArgmax = baselineProbabilitiesFromCounts(baseline);

  return {
    metric_source: metricSource,
    metric_scope: metricSource === 'browser_walk_forward_out_of_fold'
      ? 'untouched_expanding_walk_forward_folds'
      : 'untouched_post_purge_holdout',
    direction_classes: [...classes],
    accuracy: correct / n,
    macro_balanced_accuracy: macroBalancedAccuracy,
    macro_f1: macroF1,
    multiclass_brier: brier,
    baseline_multiclass_brier: brierBase,
    brier_skill: brierBase > 0 ? 1 - brier / brierBase : null,
    log_loss: logLoss / n,
    expected_calibration_error: expectedCalibrationError,
    baseline_probabilities: [...baseline],
    baseline_argmax_label: baselineArgmax,
    evaluation_origins: n,
    evaluation_labels: n,
    evaluation_rows: n,
  };
}

function baselineProbabilitiesFromCounts(baselineVector) {
  const idx = baselineVector.indexOf(Math.max(...baselineVector));
  return ['down', 'neutral', 'up'][idx] || 'neutral';
}

// Direction evidence per horizon is no longer meaningful under the v2
// contract (one three-way call per origin, not per-day signs). Use
// classificationMetricsV3.

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

// horizonClassificationMetrics was removed with direction target v2 — the
// three-way cumulative contract has no per-day decomposition. Use
// classificationMetricsV3.

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
