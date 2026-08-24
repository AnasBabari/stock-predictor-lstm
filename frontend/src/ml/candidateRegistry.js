/**
 * Development utilities for browser-side price-model research.
 *
 * Production currently exposes Balanced TF.js LSTM as its only selectable
 * generator. The tabular routines below support development-only comparisons
 * and horizon ranking; they are not silently presented as production models.
 */

export const CANDIDATE_FAMILIES = Object.freeze({
  ROLLING_MEAN: 'rolling_mean',
  RIDGE: 'ridge_regression',
  BALANCED_LSTM: 'balanced_tfjs_lstm',
});

// Only this candidate currently owns the full train/evaluate/final-forecast
// lifecycle in the production worker. Ridge and rolling mean remain explicit
// development utilities and must not be reported as selectable generators.
export const PRODUCTION_SELECTABLE_CANDIDATES = Object.freeze([
  CANDIDATE_FAMILIES.BALANCED_LSTM,
]);

/**
 * Solve Ridge Regression (L2 regularization) with exact feature centering:
 * w = (X_c^T X_c + lambda * I)^(-1) X_c^T y_c
 * bias = mean(y) - mean(X) * w
 */
export function solveRidge(X, y, lambda = 1.0) {
  const n = X.length;
  if (n === 0) return { weights: [], bias: 0, meanX: [] };
  const d = X[0].length;

  // Calculate means
  const meanX = new Float64Array(d);
  let sumY = 0;
  for (let i = 0; i < n; i += 1) {
    sumY += y[i];
    for (let j = 0; j < d; j += 1) {
      meanX[j] += X[i][j];
    }
  }
  const meanY = sumY / n;
  for (let j = 0; j < d; j += 1) meanX[j] /= n;

  // Centered Gram matrix and cross product
  const XtX = Array.from({ length: d }, () => new Float64Array(d));
  const Xty = new Float64Array(d);

  for (let i = 0; i < n; i += 1) {
    const diffY = y[i] - meanY;
    for (let j = 0; j < d; j += 1) {
      const diffXj = X[i][j] - meanX[j];
      Xty[j] += diffXj * diffY;
      for (let k = j; k < d; k += 1) {
        XtX[j][k] += diffXj * (X[i][k] - meanX[k]);
      }
    }
  }

  // Fill symmetric lower triangle and add ridge penalty
  for (let j = 0; j < d; j += 1) {
    for (let k = j; k < d; k += 1) {
      XtX[k][j] = XtX[j][k];
    }
    XtX[j][j] += Math.max(1e-9, lambda);
  }

  // Gaussian elimination with partial pivoting to solve XtX * w = Xty
  const A = XtX.map((row) => Array.from(row));
  const b = Array.from(Xty);

  for (let i = 0; i < d; i += 1) {
    let maxRow = i;
    for (let k = i + 1; k < d; k += 1) {
      if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) {
        maxRow = k;
      }
    }
    const tempA = A[i]; A[i] = A[maxRow]; A[maxRow] = tempA;
    const tempB = b[i]; b[i] = b[maxRow]; b[maxRow] = tempB;

    const diag = A[i][i];
    if (Math.abs(diag) < 1e-12) continue;

    for (let k = i + 1; k < d; k += 1) {
      const factor = A[k][i] / diag;
      b[k] -= factor * b[i];
      for (let j = i; j < d; j += 1) {
        A[k][j] -= factor * A[i][j];
      }
    }
  }

  // Back substitution
  const weights = new Array(d).fill(0);
  for (let i = d - 1; i >= 0; i -= 1) {
    let sum = 0;
    for (let j = i + 1; j < d; j += 1) {
      sum += A[i][j] * weights[j];
    }
    const diag = A[i][i];
    weights[i] = Math.abs(diag) > 1e-12 ? (b[i] - sum) / diag : 0;
  }

  let bias = meanY;
  for (let j = 0; j < d; j += 1) {
    bias -= meanX[j] * weights[j];
  }

  return { weights, bias, meanX: Array.from(meanX) };
}

/**
 * Predict using Ridge weights.
 */
export function predictRidge(X, model) {
  const { weights, bias } = model;
  return X.map((row) => {
    let dot = bias;
    for (let i = 0; i < weights.length; i += 1) {
      dot += (row[i] || 0) * weights[i];
    }
    return dot;
  });
}

/**
 * Fit and predict a rolling-mean baseline.
 */
export function fitRollingMean(y) {
  if (!y.length) return 0;
  return y.reduce((sum, val) => sum + val, 0) / y.length;
}

/**
 * Fast development comparison for tabular research candidates.
 *
 * This helper does not make a production-model selection. Production callers
 * must not claim these candidates generated a forecast unless they also fit
 * and infer with that exact candidate.
 */
export function selectDevelopmentChampions({
  trainInputs, // 3D array [sample, window, features] or 2D [sample, flattened]
  trainTargets, // 2D array [sample, horizon]
  devValidationSplits, // Array of { trainEnd, valStart, valEnd }
  horizons = [1, 3, 5, 7, 14, 30],
}) {
  const numSamples = trainInputs.length;
  if (!numSamples) throw new Error('Development competition requires non-empty training inputs.');

  // Flatten last-step features for tabular candidates (Ridge, Mean)
  const lastStepFeatures = trainInputs.map((sample) => {
    return Array.isArray(sample[0]) ? sample[sample.length - 1] : sample;
  });

  const championByHorizon = {};

  for (const h of horizons) {
    const hIdx = h - 1;
    const targetsH = trainTargets.map((row) => (row ? row[hIdx] : 0));

    let ridgeTotalLoss = 0;
    let meanTotalLoss = 0;
    let evalCount = 0;

    for (const split of devValidationSplits) {
      const trainX = lastStepFeatures.slice(0, split.trainEnd);
      const trainY = targetsH.slice(0, split.trainEnd);
      const valX = lastStepFeatures.slice(split.valStart, split.valEnd);
      const valY = targetsH.slice(split.valStart, split.valEnd);

      if (!valX.length || !trainX.length) continue;

      // 1. Ridge
      const ridgeModel = solveRidge(trainX, trainY, 1.0);
      const ridgePreds = predictRidge(valX, ridgeModel);

      // 2. Rolling Mean
      const meanVal = fitRollingMean(trainY);
      const meanPreds = valY.map(() => meanVal);

      for (let i = 0; i < valY.length; i += 1) {
        ridgeTotalLoss += (valY[i] - ridgePreds[i]) ** 2;
        meanTotalLoss += (valY[i] - meanPreds[i]) ** 2;
        evalCount += 1;
      }
    }

    const ridgeMse = evalCount > 0 ? ridgeTotalLoss / evalCount : Infinity;
    const meanMse = evalCount > 0 ? meanTotalLoss / evalCount : Infinity;

    // Default to Ridge if it improves over Rolling Mean, otherwise Rolling Mean
    championByHorizon[h] = ridgeMse <= meanMse ? CANDIDATE_FAMILIES.RIDGE : CANDIDATE_FAMILIES.ROLLING_MEAN;
  }

  return championByHorizon;
}

/**
 * Rank horizons strictly based on development cross-validation performance.
 *
 * Returns { developmentChampionHorizon, developmentRanking: [{ horizon, candidate, mse }] }
 */
export function rankDevelopmentHorizons({
  trainInputs,
  trainTargets,
  devValidationSplits,
  horizons = [1, 3, 5, 7, 14, 30],
}) {
  const lastStepFeatures = trainInputs.map((sample) => {
    return Array.isArray(sample[0]) ? sample[sample.length - 1] : sample;
  });

  const scores = [];

  for (const h of horizons) {
    const hIdx = h - 1;
    const targetsH = trainTargets.map((row) => (row ? row[hIdx] : 0));

    let totalLoss = 0;
    let totalPersistenceLoss = 0;
    let evalCount = 0;

    for (const split of devValidationSplits) {
      const trainX = lastStepFeatures.slice(0, split.trainEnd);
      const trainY = targetsH.slice(0, split.trainEnd);
      const valX = lastStepFeatures.slice(split.valStart, split.valEnd);
      const valY = targetsH.slice(split.valStart, split.valEnd);

      if (!valX.length || !trainX.length) continue;

      const ridgeModel = solveRidge(trainX, trainY, 1.0);
      const preds = predictRidge(valX, ridgeModel);

      for (let i = 0; i < valY.length; i += 1) {
        totalLoss += (valY[i] - preds[i]) ** 2;
        totalPersistenceLoss += (valY[i] - 0) ** 2; // persistence log-return = 0
        evalCount += 1;
      }
    }

    const mse = evalCount > 0 ? totalLoss / evalCount : Infinity;
    const relMse = totalPersistenceLoss > 0 ? totalLoss / totalPersistenceLoss : Infinity;
    scores.push({
      horizon: h,
      candidate: CANDIDATE_FAMILIES.RIDGE,
      mse,
      relative_mse: relMse,
    });
  }

  // Sort strictly by development relative MSE
  scores.sort((a, b) => a.relative_mse - b.relative_mse);
  const developmentChampionHorizon = scores.length ? scores[0].horizon : horizons[0];

  return {
    developmentChampionHorizon,
    developmentRanking: scores,
  };
}

/**
 * Resolve the Auto horizon recommendation.
 *
 * Invariant:
 * Candidate and horizon priorities are established STRICTLY on development folds.
 * The final holdout evaluates promotion for each horizon independently.
 *
 * The final holdout is deliberately absent from this interface. It may label
 * the frozen choice promoted or experimental, but it cannot change the choice.
 */
export function resolveAutoHorizon({
  developmentChampionHorizon,
  developmentRanking = [],
}) {
  const selected = Number(developmentChampionHorizon ?? developmentRanking[0]?.horizon);
  if (!Number.isFinite(selected)) {
    throw new Error('Auto horizon requires a development-selected champion.');
  }
  return {
    selectedHorizon: selected,
    validated: null,
    reason: `Frozen from development evidence at ${selected}d before final-holdout evaluation.`,
  };
}
