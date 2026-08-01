import * as tf from '@tensorflow/tfjs';

import {
  MODEL_VERSION,
  OUTPUT_WIDTH,
  WINDOW_SIZE,
  inverseClose,
  latestInput,
  modelKey,
  prepareDirectionData,
  preparePriceData,
  validateSnapshot,
} from './preprocessing';

const DB_NAME = 'stocklstm-browser-models';
const DB_VERSION = 1;
const STORE_NAME = 'metadata';
const MAX_MODELS = 6;
const MODEL_TTL_MS = 7 * 24 * 60 * 60 * 1000;

const cancelledIds = new Set();

function dbRequest(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('IndexedDB request failed.'));
  });
}

function openDb() {
  if (!globalThis.indexedDB) return Promise.reject(new Error('IndexedDB is unavailable.'));
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: 'key' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('IndexedDB is unavailable.'));
  });
}

async function getMetadata(key) {
  let database;
  try {
    database = await openDb();
    const transaction = database.transaction(STORE_NAME, 'readonly');
    return await dbRequest(transaction.objectStore(STORE_NAME).get(key));
  } finally {
    database?.close();
  }
}

async function listMetadata() {
  let database;
  try {
    database = await openDb();
    const transaction = database.transaction(STORE_NAME, 'readonly');
    return (await dbRequest(transaction.objectStore(STORE_NAME).getAll())) || [];
  } finally {
    database?.close();
  }
}

async function putMetadata(value) {
  let database;
  try {
    database = await openDb();
    const transaction = database.transaction(STORE_NAME, 'readwrite');
    await dbRequest(transaction.objectStore(STORE_NAME).put(value));
  } finally {
    database?.close();
  }
}

async function deleteMetadata(key) {
  let database;
  try {
    database = await openDb();
    const transaction = database.transaction(STORE_NAME, 'readwrite');
    await dbRequest(transaction.objectStore(STORE_NAME).delete(key));
  } finally {
    database?.close();
  }
}

async function pruneCache() {
  const entries = await listMetadata();
  const now = Date.now();
  const expired = entries.filter((entry) => now - entry.created_at > MODEL_TTL_MS);
  const overflow = entries.length > MAX_MODELS
    ? [...entries].sort((a, b) => a.last_used_at - b.last_used_at).slice(0, entries.length - MAX_MODELS)
    : [];
  const seen = new Set();
  for (const entry of [...expired, ...overflow]) {
    if (seen.has(entry.key)) continue;
    seen.add(entry.key);
    await tf.io.removeModel(`indexeddb://${entry.key}`).catch(() => undefined);
    await deleteMetadata(entry.key).catch(() => undefined);
  }
}

async function removeStaleEntries(snapshot, forecastType) {
  const entries = await listMetadata().catch(() => []);
  for (const entry of entries) {
    const sameModel = entry.ticker === snapshot.ticker && entry.forecast_type === forecastType;
    const compatible = entry.snapshot_id === snapshot.snapshot_id &&
      entry.schema_version === snapshot.schema_version && entry.model_version === MODEL_VERSION;
    if (!sameModel || compatible) continue;
    await tf.io.removeModel('indexeddb://' + entry.key).catch(() => undefined);
    await deleteMetadata(entry.key).catch(() => undefined);
  }
}
async function clearCache() {
  const entries = await listMetadata().catch(() => []);
  for (const entry of entries) {
    await tf.io.removeModel(`indexeddb://${entry.key}`).catch(() => undefined);
    await deleteMetadata(entry.key).catch(() => undefined);
  }
}
function emit(id, payload) {
  self.postMessage({ id, ...payload });
}

function checkCancelled(id) {
  if (cancelledIds.has(id)) throw new Error('Browser training was cancelled.');
}

function buildModel(forecastType, featureCount) {
  const outputActivation = forecastType === 'direction' ? 'sigmoid' : undefined;
  const model = tf.sequential();
  model.add(tf.layers.lstm({ units: 32, returnSequences: true, inputShape: [WINDOW_SIZE, featureCount] }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.lstm({ units: 16 }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: OUTPUT_WIDTH, activation: outputActivation }));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: forecastType === 'direction' ? 'binaryCrossentropy' : 'meanSquaredError',
  });
  return model;
}

function flatten(values) {
  return values.flatMap((row) => (Array.isArray(row) ? row : [row])).map(Number);
}

function regressionMetrics(actual, predicted, persistence) {
  const errors = actual.map((value, index) => value - predicted[index]);
  const absolute = errors.map((value) => Math.abs(value));
  const squared = errors.map((value) => value ** 2);
  const mean = actual.reduce((sum, value) => sum + value, 0) / actual.length;
  const ssTotal = actual.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  const mape = actual.reduce((sum, value, index) => sum + Math.abs(errors[index] / value), 0) / actual.length;
  const baselineErrors = actual.map((value, index) => value - persistence[index]);
  const mae = absolute.reduce((sum, value) => sum + value, 0) / actual.length;
  const mse = squared.reduce((sum, value) => sum + value, 0) / actual.length;
  const baselineMae = baselineErrors.reduce((sum, value) => sum + Math.abs(value), 0) / actual.length;
  const baselineRmse = Math.sqrt(baselineErrors.reduce((sum, value) => sum + value ** 2, 0) / actual.length);
  return {
    metric_source: 'browser_purged_holdout',
    metric_scope: 'untouched_post_purge_holdout',
    mae,
    mse,
    rmse: Math.sqrt(mse),
    mape: mape * 100,
    r2: ssTotal === 0 ? 0 : 1 - squared.reduce((sum, value) => sum + value, 0) / ssTotal,
    relative_mae: baselineMae === 0 ? null : mae / baselineMae,
    relative_rmse: baselineRmse === 0 ? null : Math.sqrt(mse) / baselineRmse,
  };
}

function classificationMetrics(actual, predicted) {
  const labels = flatten(actual).map((value) => Number(value) > 0.5 ? 1 : 0);
  const probabilities = flatten(predicted).map((value) => Math.min(1, Math.max(0, Number(value))));
  const labelsPredicted = probabilities.map((value) => value >= 0.5 ? 1 : 0);
  let tp = 0; let tn = 0; let fp = 0; let fn = 0;
  labels.forEach((value, index) => {
    if (value === 1 && labelsPredicted[index] === 1) tp += 1;
    else if (value === 0 && labelsPredicted[index] === 0) tn += 1;
    else if (value === 0) fp += 1;
    else fn += 1;
  });
  const accuracy = (tp + tn) / labels.length;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const downRecall = tn + fp ? tn / (tn + fp) : 0;
  const naiveBaseline = Math.max(labels.filter((value) => value === 1).length, labels.length - labels.filter((value) => value === 1).length) / labels.length;
  const brier = probabilities.reduce((sum, value, index) => sum + (value - labels[index]) ** 2, 0) / labels.length;
  return {
    metric_source: 'browser_purged_holdout',
    metric_scope: 'untouched_post_purge_holdout',
    accuracy,
    directional_accuracy: accuracy,
    precision,
    recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    balanced_accuracy: (recall + downRecall) / 2,
    brier_score: brier,
    naive_baseline: naiveBaseline,
  };
}

async function selectBackend() {
  try {
    const configured = await tf.setBackend('webgl');
    if (!configured) throw new Error('WebGL backend is unavailable.');
    await tf.ready();
    return 'webgl';
  } catch {
    const configured = await tf.setBackend('cpu');
    if (!configured) throw new Error('CPU backend is unavailable.');
    await tf.ready();
    return 'cpu';
  }
}

async function trainAndPredict(id, snapshot, forecastType, days) {
  validateSnapshot(snapshot);
  checkCancelled(id);
  const backend = await selectBackend();
  const key = modelKey(snapshot, forecastType);
  const cacheUrl = `indexeddb://${key}`;
  await removeStaleEntries(snapshot, forecastType);
  const prepared = forecastType === 'direction' ? prepareDirectionData(snapshot) : preparePriceData(snapshot);
  let model;
  let cacheStatus = 'miss';
  let metrics;
  const cachedMetadata = await getMetadata(key).catch(() => null);
  const cacheMatchesSnapshot = cachedMetadata?.snapshot_id === snapshot.snapshot_id;
  if (cachedMetadata && cacheMatchesSnapshot) {
    try {
      model = await tf.loadLayersModel(cacheUrl);
      cacheStatus = 'hit';
      metrics = cachedMetadata.metrics;
      await putMetadata({ ...cachedMetadata, last_used_at: Date.now() });
      emit(id, { type: 'progress', stage: 'cache_hit', message: 'Loaded your cached browser model.' });
    } catch {
      await deleteMetadata(key).catch(() => undefined);
      await tf.io.removeModel(cacheUrl).catch(() => undefined);
    }
  }

  if (!model) {
    model = buildModel(forecastType, snapshot.feature_names.length);
    const xs = tf.tensor3d(prepared.inputs);
    const ys = tf.tensor2d(prepared.targets);
    const trainXs = xs.slice([0, 0, 0], [prepared.trainCount, -1, -1]);
    const trainYs = ys.slice([0, 0], [prepared.trainCount, -1]);
    const validationXs = xs.slice([prepared.split, 0, 0], [-1, -1, -1]);
    const validationYs = ys.slice([prepared.split, 0], [-1, -1]);
    emit(id, { type: 'progress', stage: 'training', epoch: 0, total_epochs: 12, backend });
    try {
      let bestLoss = Infinity;
      let waiting = 0;
      await model.fit(trainXs, trainYs, {
        epochs: 12,
        batchSize: 32,
        shuffle: false,
        validationData: [validationXs, validationYs],
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            checkCancelled(id);
            const loss = Number(logs?.val_loss ?? logs?.loss ?? Infinity);
            if (loss < bestLoss - 1e-6) { bestLoss = loss; waiting = 0; } else { waiting += 1; }
            emit(id, { type: 'progress', stage: 'training', epoch: epoch + 1, total_epochs: 12, loss, backend });
            if (waiting >= 3) model.stopTraining = true;
            await tf.nextFrame();
          },
        },
      });
      checkCancelled(id);
      const holdoutXs = tf.tensor3d(prepared.inputs.slice(prepared.split));
      try {
        const output = model.predict(holdoutXs);
        try {
          const predictedHoldout = await output.array();
          if (forecastType === "direction") {
            metrics = classificationMetrics(prepared.targets.slice(prepared.split), predictedHoldout);
          } else {
            const actual = flatten(prepared.targets.slice(prepared.split)).map((value) => inverseClose(value, prepared.scaler, prepared.closeIndex));
            const predicted = flatten(predictedHoldout).map((value) => inverseClose(value, prepared.scaler, prepared.closeIndex));
            const persistence = prepared.origins.slice(prepared.split).flatMap((origin) => Array(OUTPUT_WIDTH).fill(inverseClose(origin, prepared.scaler, prepared.closeIndex)));
            metrics = regressionMetrics(actual, predicted, persistence);
          }
        } finally {
          output.dispose();
        }
      } finally {
        holdoutXs.dispose();
      }
      try {
        await model.save(cacheUrl);
        await putMetadata({
          key,
          ticker: snapshot.ticker,
          forecast_type: forecastType,
          snapshot_id: snapshot.snapshot_id,
          schema_version: snapshot.schema_version,
          model_version: MODEL_VERSION,
          created_at: Date.now(),
          last_used_at: Date.now(),
          feature_names: snapshot.feature_names,
          output_width: snapshot.output_width,
          metrics,
        });
        await pruneCache();
        cacheStatus = "stored";
      } catch {
        cacheStatus = "session_only";
      }
    } catch (error) {
      model.dispose();
      throw error;
    } finally {
      xs.dispose();
      ys.dispose();
      trainXs.dispose();
      trainYs.dispose();
      validationXs.dispose();
      validationYs.dispose();
    }
  }
  try {
    checkCancelled(id);
    const input = tf.tensor3d([latestInput(prepared)]);
    let predicted;
    let output;
    try {
      output = model.predict(input);
      predicted = await output.array();
    } finally {
      output?.dispose();
      input.dispose();
    }

    if (forecastType === 'direction') {
      const probabilities = predicted[0].slice(0, days).map((value) => Math.min(1, Math.max(0, Number(value))));
      return {
        forecastType,
        directions: probabilities.map((value) => value >= 0.5 ? 'Up' : 'Down'),
        probabilities,
        metrics,
        cacheStatus,
        backend,
        executionMode: cacheStatus === 'hit' ? 'browser_artifact_loaded' : 'browser_trained',
      };
    }
    return {
      forecastType,
      predictedPrices: predicted[0].slice(0, days).map((value) => inverseClose(value, prepared.scaler, prepared.closeIndex)),
      metrics,
      cacheStatus,
      backend,
      executionMode: cacheStatus === 'hit' ? 'browser_artifact_loaded' : 'browser_trained',
    };
  } finally {
    model?.dispose();
  }
}

self.onmessage = async (event) => {
  const { id, type, snapshot, forecastType, days } = event.data || {};
  if (type === 'cancel') { cancelledIds.add(id); return; }
  if (type === 'clear-cache') {
    try {
      await selectBackend();
      await clearCache();
      emit(id, { type: 'complete', result: { cleared: true } });
    } catch (error) {
      emit(id, { type: 'error', message: error instanceof Error ? error.message : 'Could not clear browser models.' });
    }
    return;
  }
  if (type !== 'forecast' || !id) return;
  cancelledIds.delete(id);
  try {
    const result = await trainAndPredict(id, snapshot, forecastType, Number(days));
    emit(id, { type: 'complete', result });
    cancelledIds.delete(id);
  } catch (error) {
    emit(id, { type: 'error', message: error instanceof Error ? error.message : 'Browser training failed.' });
    cancelledIds.delete(id);
  }
};
