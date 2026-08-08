import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

import {
  ARCHITECTURE_VERSION,
  MODEL_VERSION,
  OUTPUT_WIDTH,
  TARGET_MODE,
  TRAIN_SPLIT,
  WINDOW_SIZE,
  latestInput,
  modelKey,
  prepareDirectionData,
  preparePriceData,
  resolveHorizon,
  sequencePartition,
  validateSnapshot,
} from './preprocessing';
import {
  classificationMetrics,
  directionMajority,
  generateResearchSplits,
  horizonClassificationMetrics,
  horizonRegressionMetrics,
  median,
  regressionMetrics,
} from './evaluation';
import { buildPersistenceForecast, evaluatePromotion } from './promotionPolicy';
import { resolveTrainingProfile } from './trainingProfiles';
import { buildBrowserModel } from './modelFactory';
import { isVersionedKey } from './storageKeys';

const DB_NAME = 'stocklstm-browser-models';
const DB_VERSION = 1;
const STORE_NAME = 'metadata';
const MAX_MODELS = 8;
const MAX_MODEL_BYTES = 200 * 1024 * 1024;
const MODEL_TTL_MS = 7 * 24 * 60 * 60 * 1000;
const CHECKPOINT_TTL_MS = 24 * 60 * 60 * 1000;
const PROFILE_RANK = { quick: 1, balanced: 2, research: 3 };
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
      if (!request.result.objectStoreNames.contains(STORE_NAME)) {
        request.result.createObjectStore(STORE_NAME, { keyPath: 'key' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('IndexedDB is unavailable.'));
  });
}

async function metadataOperation(mode, operation) {
  let database;
  try {
    database = await openDb();
    const transaction = database.transaction(STORE_NAME, mode);
    return await operation(transaction.objectStore(STORE_NAME));
  } finally {
    database?.close();
  }
}

const getMetadata = (key) => metadataOperation('readonly', (store) => dbRequest(store.get(key)));
const listMetadata = () => metadataOperation('readonly', (store) => dbRequest(store.getAll()));
const putMetadata = (value) => metadataOperation('readwrite', (store) => dbRequest(store.put(value)));
const deleteMetadata = (key) => metadataOperation('readwrite', (store) => dbRequest(store.delete(key)));

async function deleteEntry(entry) {
  if (entry.kind !== 'checkpoint') {
    await tf.io.removeModel(`indexeddb://${entry.key}`).catch(() => undefined);
  }
  await deleteMetadata(entry.key).catch(() => undefined);
}

async function pruneCache() {
  const entries = await listMetadata().catch(() => []);
  const now = Date.now();
  // Reclaim entries whose key belongs to an older model/architecture/target
  // version namespace; version bumps change the key rather than invalidating
  // in place, so stale-version rows would otherwise linger until the TTL.
  for (const entry of entries.filter((item) => !isVersionedKey(item?.key))) {
    await deleteEntry(entry);
  }

  for (const entry of (await listMetadata().catch(() => [])).filter((item) =>
    item.kind === 'checkpoint'
      ? now - item.updated_at > CHECKPOINT_TTL_MS
      : now - item.created_at > MODEL_TTL_MS)) {
    await deleteEntry(entry);
  }

  const models = (await listMetadata().catch(() => []))
    .filter((entry) => entry.kind !== 'checkpoint')
    .sort((a, b) => a.last_used_at - b.last_used_at);
  let bytes = models.reduce((sum, entry) => sum + Number(entry.size_bytes || 0), 0);
  while (models.length > MAX_MODELS || bytes > MAX_MODEL_BYTES) {
    const entry = models.shift();
    bytes -= Number(entry.size_bytes || 0);
    await deleteEntry(entry);
  }
}

async function removeIncompatibleEntries({ key, ticker, forecastType, profile, backend, horizon }) {
  const entries = await listMetadata().catch(() => []);
  for (const entry of entries) {
    const sameSlot = entry.ticker === ticker && entry.forecast_type === forecastType &&
      entry.training_profile === profile && entry.horizon === horizon &&
      (entry.backend == null || entry.backend === backend);
    if (sameSlot && entry.key !== key && entry.key !== `checkpoint/${key}`) await deleteEntry(entry);
  }
}

async function pruneSuperseded(metadata) {
  const entries = await listMetadata().catch(() => []);
  const rank = PROFILE_RANK[metadata.training_profile] || 0;
  for (const entry of entries) {
    if (entry.kind === 'checkpoint' || entry.key === metadata.key) continue;
    const sameForecast = entry.ticker === metadata.ticker && entry.forecast_type === metadata.forecast_type;
    if (sameForecast && (PROFILE_RANK[entry.training_profile] || 0) < rank) await deleteEntry(entry);
  }
}

async function clearCache() {
  for (const entry of await listMetadata().catch(() => [])) await deleteEntry(entry);
}

function emit(id, payload) {
  self.postMessage({ id, ...payload });
}

function progress(id, startedAt, payload) {
  emit(id, { type: 'progress', elapsed_ms: Math.round(performance.now() - startedAt), ...payload });
}

function checkCancelled(id) {
  if (cancelledIds.has(id)) throw new Error('Browser training was cancelled.');
}

async function selectBackend() {
  for (const name of ['webgpu', 'webgl', 'cpu']) {
    try {
      if (name === 'webgpu' && !globalThis.navigator?.gpu) continue;
      if (await tf.setBackend(name)) {
        await tf.ready();
        return name;
      }
    } catch {
      // Try the next local backend.
    }
  }
  throw new Error('No supported TensorFlow.js backend is available.');
}

async function benchmarkBackend() {
  const left = tf.randomUniform([96, 96], 0, 1, 'float32', 51);
  const right = tf.randomUniform([96, 96], 0, 1, 'float32', 52);
  const started = performance.now();
  const result = tf.matMul(left, right);
  await result.data();
  const duration = performance.now() - started;
  left.dispose(); right.dispose(); result.dispose();
  return Math.round(duration);
}

function prepare(snapshot, forecastType, fitSequenceEndExclusive, horizon) {
  return forecastType === 'direction'
    ? prepareDirectionData(snapshot, fitSequenceEndExclusive, horizon)
    : preparePriceData(snapshot, fitSequenceEndExclusive, horizon);
}

async function predictRows(model, inputs) {
  const tensor = tf.tensor3d(inputs);
  let output;
  try {
    output = model.predict(tensor);
    return await output.array();
  } finally {
    output?.dispose();
    tensor.dispose();
  }
}

async function fitSelectionModel({ id, model, inputs, targets, fitSequenceEndExclusive, validationStart, validationEnd = inputs.length, profile, startedAt, stage, fold }) {
  const xs = tf.tensor3d(inputs);
  const ys = tf.tensor2d(targets);
  const trainXs = xs.slice([0, 0, 0], [fitSequenceEndExclusive, -1, -1]);
  const trainYs = ys.slice([0, 0], [fitSequenceEndExclusive, -1]);
  const validationXs = xs.slice([validationStart, 0, 0], [validationEnd - validationStart, -1, -1]);
  const validationYs = ys.slice([validationStart, 0], [validationEnd - validationStart, -1]);
  let bestLoss = Infinity;
  let bestEpoch = 1;
  let waiting = 0;
  let completedEpochs = 0;
  let bestWeights;
  try {
    await model.fit(trainXs, trainYs, {
      epochs: profile.epochs,
      batchSize: 32,
      shuffle: false,
      validationData: [validationXs, validationYs],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          checkCancelled(id);
          completedEpochs = epoch + 1;
          const loss = Number(logs?.val_loss ?? logs?.loss ?? Infinity);
          if (loss < bestLoss - 1e-6) {
            bestLoss = loss; bestEpoch = completedEpochs; waiting = 0;
            bestWeights?.forEach((weight) => weight.dispose());
            bestWeights = model.getWeights().map((weight) => weight.clone());
          } else {
            waiting += 1;
          }
          progress(id, startedAt, {
            stage, fold, folds: profile.folds, epoch: completedEpochs,
            total_epochs: profile.epochs, loss, backend: tf.getBackend(), profile: profile.id,
          });
          if (waiting >= profile.patience) model.stopTraining = true;
          await tf.nextFrame();
        },
      },
    });
    if (bestWeights) {
      model.setWeights(bestWeights);
      bestWeights.forEach((weight) => weight.dispose());
      bestWeights = undefined;
    }
    return { bestEpoch, completedEpochs, bestLoss };
  } finally {
    bestWeights?.forEach((weight) => weight.dispose());
    trainXs.dispose(); trainYs.dispose();
    validationXs.dispose(); validationYs.dispose();
  }
}

async function fitFinalModel({ id, model, prepared, epochs, profile, startedAt }) {
  const xs = tf.tensor3d(prepared.inputs);
  const ys = tf.tensor2d(prepared.targets);
  try {
    await model.fit(xs, ys, {
      epochs,
      batchSize: 32,
      shuffle: false,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          checkCancelled(id);
          progress(id, startedAt, {
            stage: 'final_fit', epoch: epoch + 1, total_epochs: epochs,
            loss: Number(logs?.loss), backend: tf.getBackend(), profile: profile.id,
          });
          await tf.nextFrame();
        },
      },
    });
  } finally {
    xs.dispose(); ys.dispose();
  }
}

function combineMetrics(returnMetrics, dollarMetrics, horizon, metricSource) {
  const pooled = returnMetrics.pooled;
  const perHorizon = returnMetrics.per_horizon.map((entry, index) => ({
    ...entry,
    dollar_mae: dollarMetrics.per_horizon[index].mae,
    dollar_rmse: dollarMetrics.per_horizon[index].rmse,
    dollar_relative_mae: dollarMetrics.per_horizon[index].relative_mae,
    dollar_relative_rmse: dollarMetrics.per_horizon[index].relative_rmse,
  }));
  return {
    metric_source: metricSource,
    metric_scope: pooled.metric_scope,
    horizon,
    target_mode: TARGET_MODE,
    mae: pooled.mae,
    mse: pooled.mse,
    rmse: pooled.rmse,
    mape: pooled.mape,
    r2: pooled.r2,
    relative_mae: pooled.relative_mae,
    relative_rmse: pooled.relative_rmse,
    dollar_mae: dollarMetrics.pooled.mae,
    dollar_rmse: dollarMetrics.pooled.rmse,
    dollar_relative_mae: dollarMetrics.pooled.relative_mae,
    dollar_relative_rmse: dollarMetrics.pooled.relative_rmse,
    directional_accuracy: returnMetrics.directional_accuracy,
    per_horizon: perHorizon,
    evaluation_rows: returnMetrics.evaluation_rows,
  };
}

async function evaluateRange(model, prepared, forecastType, start, end, metricSource, horizon, majorityLabel) {
  const predictedRows = await predictRows(model, prepared.inputs.slice(start, end));
  if (forecastType === 'direction') {
    const actual = prepared.targets.slice(start, end);
    const evidence = horizonClassificationMetrics(actual, predictedRows, metricSource, majorityLabel);
    return {
      actual,
      predicted: predictedRows,
      metrics: evidence.pooled,
      direction_per_horizon: evidence.per_horizon,
    };
  }
  const actualRows = prepared.targets.slice(start, end);
  const origins = prepared.origins.slice(start, end);
  const actualPrices = actualRows.map((row, sample) => row.map((value) => origins[sample] * Math.exp(value)));
  const predictedPrices = predictedRows.map((row, sample) => row.map((value) => origins[sample] * Math.exp(value)));
  const persistenceRows = actualRows.map((row) => row.map(() => 0));
  const persistencePrices = actualPrices.map((row, sample) => row.map(() => origins[sample]));
  const returnMetrics = horizonRegressionMetrics(actualRows, predictedRows, persistenceRows, horizon, metricSource);
  const dollarMetrics = horizonRegressionMetrics(actualPrices, predictedPrices, persistencePrices, horizon, metricSource);
  return {
    actual: actualRows,
    predicted: predictedRows,
    persistence: persistenceRows,
    actualPrices,
    predictedPrices,
    persistencePrices,
    returnMetrics,
    dollarMetrics,
  };
}

function aggregateFoldMetrics(records, forecastType, horizon, majorityLabel) {
  if (forecastType === 'direction') {
    const actuals = records.flatMap((record) => record.actual);
    const predicteds = records.flatMap((record) => record.predicted);
    const evidence = horizonClassificationMetrics(
      actuals,
      predicteds,
      'browser_walk_forward_out_of_fold',
      majorityLabel,
    );
    return {
      metrics: evidence.pooled,
      dollarMetrics: null,
      direction_per_horizon: evidence.per_horizon,
    };
  }
  const returnMetrics = horizonRegressionMetrics(
    records.flatMap((record) => record.actual),
    records.flatMap((record) => record.predicted),
    records.flatMap((record) => record.persistence),
    horizon,
    'browser_walk_forward_out_of_fold',
  );
  const dollarMetrics = horizonRegressionMetrics(
    records.flatMap((record) => record.actualPrices),
    records.flatMap((record) => record.predictedPrices),
    records.flatMap((record) => record.persistencePrices),
    horizon,
    'browser_walk_forward_out_of_fold',
  );
  return { metrics: combineMetrics(returnMetrics, dollarMetrics, horizon, 'browser_walk_forward_out_of_fold'), dollarMetrics };
}

async function trainHoldout(id, snapshot, forecastType, profile, startedAt, horizon) {
  const { sampleCount } = sequencePartition(snapshot, forecastType, horizon);
  let selection = prepare(snapshot, forecastType, undefined, horizon);
  const selectionModel = buildBrowserModel(forecastType, snapshot.feature_names.length, profile, horizon);
  let metrics;
  let dollarMetrics;
  let selectedEpochs;
  try {
    const innerValidationSize = Math.max(1, Math.floor(selection.trainCount * 0.1));
    const innerValidationStart = selection.trainCount - innerValidationSize;
    const fitSequenceEndExclusive = innerValidationStart - (horizon - 1);
    if (fitSequenceEndExclusive < 1) throw new Error('Not enough training data for a purged validation split.');
    const fit = await fitSelectionModel({
      id,
      model: selectionModel,
      inputs: selection.inputs,
      targets: selection.targets,
      fitSequenceEndExclusive,
      validationStart: innerValidationStart,
      validationEnd: selection.trainCount,
      profile,
      startedAt,
      stage: 'training',
    });
    selectedEpochs = fit.bestEpoch;
    // The majority baseline must not peek into the evaluation window: it is
    // derived from labels strictly prior to the holdout split.
    const majority = directionMajority(selection.targets.slice(0, selection.split));
    const evaluated = await evaluateRange(
      selectionModel, selection, forecastType, selection.split, selection.inputs.length, profile.metricSource, horizon, majority.label,
    );
    metrics = evaluated.metrics ?? combineMetrics(
      evaluated.returnMetrics,
      evaluated.dollarMetrics,
      horizon,
      profile.metricSource,
    );
    if (evaluated.direction_per_horizon?.length) {
      metrics.direction_per_horizon = evaluated.direction_per_horizon;
    }
    dollarMetrics = evaluated.dollarMetrics;
  } finally {
    selectionModel.dispose();
  }

  checkCancelled(id);
  selection = null;
  const finalPrepared = prepare(snapshot, forecastType, sampleCount, horizon);
  const finalModel = buildBrowserModel(forecastType, snapshot.feature_names.length, profile, horizon);
  try {
    await fitFinalModel({ id, model: finalModel, prepared: finalPrepared, epochs: selectedEpochs, profile, startedAt });
  } catch (error) {
    finalModel.dispose();
    throw error;
  }
  const foldRelativeRmse = metrics.relative_rmse;
  return {
    model: finalModel,
    prepared: finalPrepared,
    metrics,
    dollarMetrics,
    selectedEpochs,
    completedEpochs: selectedEpochs,
    evaluation: {
      completed_folds: 1,
      total_folds: 1,
      complete: true,
      fold_summaries: [{
        fold: 1,
        relative_rmse: forecastType === 'direction' ? null : metrics.relative_rmse,
        ...(forecastType === 'direction' ? {
          balanced_accuracy: metrics.balanced_accuracy,
          brier_score: metrics.brier_score,
          naive_baseline: metrics.naive_baseline,
        } : {}),
      }],
    },
  };
}

async function trainResearch(id, snapshot, forecastType, profile, startedAt, checkpointKey, horizon) {
  const { sampleCount } = sequencePartition(snapshot, forecastType, horizon);
  const splits = generateResearchSplits(sampleCount, {
    folds: profile.folds,
    validationHorizon: profile.validationHorizon,
    minTrainSamples: profile.minTrainSamples,
    purge: horizon - 1,
  });
  const checkpoint = await getMetadata(checkpointKey).catch(() => null);
  const validCheckpoint = checkpoint?.kind === 'checkpoint' &&
    Array.isArray(checkpoint.fold_records) && checkpoint.fold_records.length <= profile.folds &&
    checkpoint.fold_records.every((record, index) => record?.fold === index + 1 && Number.isFinite(record?.best_epoch));
  if (checkpoint && !validCheckpoint) await deleteMetadata(checkpointKey).catch(() => undefined);
  const records = validCheckpoint ? [...checkpoint.fold_records] : [];
  if (records.length) {
    progress(id, startedAt, {
      stage: 'checkpoint_loaded', fold: records.length, folds: profile.folds,
      profile: profile.id, backend: tf.getBackend(),
    });
  }

  for (const split of splits.slice(records.length)) {
    checkCancelled(id);
    const foldPrepared = prepare(snapshot, forecastType, split.trainEnd, horizon);
    // Every validation fold defines its own untouched evaluation window; the
    // majority class is taken from labels strictly before that window.
    const foldMajority = directionMajority(foldPrepared.targets.slice(0, split.validationStart));
    const innerValidationSize = Math.max(1, Math.floor(split.trainEnd * 0.1));
    const innerValidationStart = split.trainEnd - innerValidationSize;
    const fitSequenceEndExclusive = innerValidationStart - (horizon - 1);
    if (fitSequenceEndExclusive < 1) throw new Error('Research fold purge leaves no fitting samples.');
    const foldModel = buildBrowserModel(forecastType, snapshot.feature_names.length, profile, horizon);
    try {
      const fit = await fitSelectionModel({
        id,
        model: foldModel,
        inputs: foldPrepared.inputs.slice(0, split.trainEnd),
        targets: foldPrepared.targets.slice(0, split.trainEnd),
        fitSequenceEndExclusive,
        validationStart: innerValidationStart,
        validationEnd: split.trainEnd,
        profile,
        startedAt,
        stage: 'evaluating_fold',
        fold: split.fold,
      });
      const evaluated = await evaluateRange(
        foldModel,
        foldPrepared,
        forecastType,
        split.validationStart,
        split.validationEnd,
        profile.metricSource,
        horizon,
        foldMajority.label,
      );
      records.push({
        fold: split.fold,
        best_epoch: fit.bestEpoch,
        metrics: evaluated.metrics,
        majority: foldMajority,
        actual: evaluated.actual,
        predicted: evaluated.predicted,
        ...(evaluated.persistence ? { persistence: evaluated.persistence } : {}),
        ...(evaluated.actualPrices ? { actualPrices: evaluated.actualPrices } : {}),
        ...(evaluated.predictedPrices ? { predictedPrices: evaluated.predictedPrices } : {}),
        ...(evaluated.persistencePrices ? { persistencePrices: evaluated.persistencePrices } : {}),
        fold_relative_rmse: evaluated.returnMetrics ? evaluated.returnMetrics.pooled.relative_rmse : null,
      });
      await putMetadata({
        key: checkpointKey,
        kind: 'checkpoint',
        ticker: snapshot.ticker,
        forecast_type: forecastType,
        training_profile: profile.id,
        snapshot_id: snapshot.snapshot_id,
        horizon,
        target_mode: TARGET_MODE,
        fold_records: records,
        updated_at: Date.now(),
      }).catch(() => undefined);
    } finally {
      foldModel.dispose();
    }
    await tf.nextFrame();
  }

  const aggregated = aggregateFoldMetrics(
    records,
    forecastType,
    horizon,
    records[records.length - 1]?.majority?.label,
  );
  const metrics = aggregated.metrics;
  const dollarMetrics = aggregated.dollarMetrics;
  if (aggregated.direction_per_horizon?.length) {
    metrics.direction_per_horizon = aggregated.direction_per_horizon;
  }
  const selectedEpochs = Math.max(1, median(records.map((record) => record.best_epoch)));
  const finalPrepared = prepare(snapshot, forecastType, sampleCount, horizon);
  const finalModel = buildBrowserModel(forecastType, snapshot.feature_names.length, profile, horizon);
  try {
    await fitFinalModel({ id, model: finalModel, prepared: finalPrepared, epochs: selectedEpochs, profile, startedAt });
  } catch (error) {
    finalModel.dispose();
    throw error;
  }
  await deleteMetadata(checkpointKey).catch(() => undefined);
  return {
    model: finalModel,
    prepared: finalPrepared,
    metrics,
    dollarMetrics,
    selectedEpochs,
    completedEpochs: records.reduce((sum, record) => sum + record.best_epoch, 0) + selectedEpochs,
    evaluation: {
      completed_folds: records.length,
      total_folds: profile.folds,
      complete: records.length === profile.folds,
      fold_summaries: records.map((record) => ({
        fold: record.fold,
        best_epoch: record.best_epoch,
        relative_rmse: record.fold_relative_rmse,
        ...(record.metrics ? {
          balanced_accuracy: record.metrics.balanced_accuracy,
          brier_score: record.metrics.brier_score,
          naive_baseline: record.metrics.naive_baseline,
        } : {}),
      })),
    },
  };
}

function validCachedModel(model, metadata, snapshot, profile, backend, horizon) {
  const inputShape = model.inputs?.[0]?.shape || [];
  const outputShape = model.outputs?.[0]?.shape || [];
  return metadata.model_version === MODEL_VERSION &&
    metadata.architecture_version === ARCHITECTURE_VERSION &&
    metadata.target_mode === TARGET_MODE &&
    metadata.horizon === horizon &&
    metadata.snapshot_id === snapshot.snapshot_id &&
    metadata.training_profile === profile.id &&
    metadata.backend === backend &&
    metadata.metrics && metadata.evaluation?.complete === true &&
    Number.isFinite(metadata.selected_epochs) &&
    JSON.stringify(metadata.feature_names) === JSON.stringify(snapshot.feature_names) &&
    inputShape[1] === WINDOW_SIZE && inputShape[2] === snapshot.feature_names.length &&
    outputShape[outputShape.length - 1] === horizon;
}

async function saveModelWithTimeout(model, cacheUrl, timeoutMs = 10_000) {
  let timedOut = false;
  const timeout = new Promise((_, reject) => setTimeout(() => {
    timedOut = true;
    reject(new Error('IndexedDB model save timed out.'));
  }, timeoutMs));
  try {
    await Promise.race([model.save(cacheUrl), timeout]);
    return { saved: true, timedOut: false };
  } catch (error) {
    if (timedOut) return { saved: false, timedOut: true };
    throw error;
  }
}
async function trainAndPredict(id, snapshot, rawForecastType, days, profileName) {
  validateSnapshot(snapshot);
  const forecastType = (rawForecastType === 'trend' || rawForecastType === 'direction') ? 'direction' : 'price';
  const profile = resolveTrainingProfile(profileName);
  const horizon = resolveHorizon(days);
  const requestedDays = Math.max(1, Math.min(OUTPUT_WIDTH, Math.round(Number(days) || 1)));
  const startedAt = performance.now();
  checkCancelled(id);
  const backend = await selectBackend();
  const benchmarkMs = await benchmarkBackend();
  const key = modelKey(snapshot, forecastType, profile.id, backend, horizon);
  const cacheUrl = `indexeddb://${key}`;
  const checkpointKey = `checkpoint/${key}`;
  progress(id, startedAt, {
    stage: 'capability_check', profile: profile.id, backend, benchmark_ms: benchmarkMs,
    estimated_seconds: profile.expectedSeconds, horizon,
  });
  await pruneCache();
  await removeIncompatibleEntries({ key, ticker: snapshot.ticker, forecastType, profile: profile.id, backend, horizon });

  let model;
  let prepared;
  let metrics;
  let evaluation;
  let selectedEpochs;
  let completedEpochs;
  let cacheStatus = 'miss';
  let storageStatus = 'persistent';
  const cachedMetadata = await getMetadata(key).catch(() => null);
  if (cachedMetadata?.kind !== 'checkpoint') {
    try {
      model = await tf.loadLayersModel(cacheUrl);
      if (!validCachedModel(model, cachedMetadata, snapshot, profile, backend, horizon)) {
        throw new Error('Cached browser model is incompatible.');
      }
      prepared = prepare(snapshot, forecastType, sequencePartition(snapshot, forecastType, horizon).sampleCount, horizon);
      metrics = cachedMetadata.metrics;
      evaluation = cachedMetadata.evaluation;
      selectedEpochs = cachedMetadata.selected_epochs;
      completedEpochs = cachedMetadata.completed_epochs;
      cacheStatus = 'hit';
      await putMetadata({ ...cachedMetadata, last_used_at: Date.now() }).catch(() => undefined);
      progress(id, startedAt, { stage: 'cache_hit', profile: profile.id, backend, horizon });
    } catch {
      model?.dispose(); model = undefined;
      await tf.io.removeModel(cacheUrl).catch(() => undefined);
      if (cachedMetadata) await deleteMetadata(key).catch(() => undefined);
    }
  }

  if (!model) {
    const trained = profile.id === 'research'
      ? await trainResearch(id, snapshot, forecastType, profile, startedAt, checkpointKey, horizon)
      : await trainHoldout(id, snapshot, forecastType, profile, startedAt, horizon);
    ({ model, prepared, metrics, evaluation, selectedEpochs, completedEpochs } = trained);
    const runtime = {
      tfjs_version: tf.version.tfjs,
      backend,
      benchmark_ms: benchmarkMs,
      device_memory_gb: Number(globalThis.navigator?.deviceMemory || 0) || null,
      hardware_concurrency: Number(globalThis.navigator?.hardwareConcurrency || 0) || null,
      user_agent: globalThis.navigator?.userAgent || 'unknown',
    };
    const metadata = {
      key,
      kind: 'model',
      ticker: snapshot.ticker,
      forecast_type: forecastType,
      snapshot_id: snapshot.snapshot_id,
      schema_version: snapshot.schema_version,
      model_version: MODEL_VERSION,
      architecture_version: ARCHITECTURE_VERSION,
      target_mode: TARGET_MODE,
      horizon,
      training_profile: profile.id,
      backend,
      created_at: Date.now(),
      last_used_at: Date.now(),
      feature_names: snapshot.feature_names,
      output_width: horizon,
      scaler: prepared.scaler,
      metrics,
      evaluation,
      selected_epochs: selectedEpochs,
      completed_epochs: completedEpochs,
      training_duration_ms: Math.round(performance.now() - startedAt),
      runtime,
    };
    let saveTimedOut = false;
    try {
      const saveResult = await saveModelWithTimeout(model, cacheUrl);
      saveTimedOut = saveResult.timedOut;
      if (!saveResult.saved) {
        storageStatus = 'session_only';
        cacheStatus = 'session_only';
      } else {
        const stored = (await tf.io.listModels())[cacheUrl];
        metadata.size_bytes = Number(stored?.weightDataBytes || 0) + Number(stored?.modelTopologyBytes || 0);
        await putMetadata(metadata);
        await pruneSuperseded(metadata);
        await pruneCache();
        cacheStatus = 'stored';
      }
    } catch {
      if (!saveTimedOut) await tf.io.removeModel(cacheUrl).catch(() => undefined);
      await deleteMetadata(key).catch(() => undefined);
      storageStatus = 'session_only';
      cacheStatus = 'session_only';
    }
  }

  try {
    checkCancelled(id);
    const predicted = await predictRows(model, [latestInput(prepared)]);
    const predictedReturns = predicted[0].slice(0, requestedDays);
    if (!predictedReturns.every((value) => Number.isFinite(Number(value)))) {
      throw new Error('The local model produced invalid forecast values.');
    }
    const common = {
      forecastType,
      metrics,
      evaluation,
      cacheStatus,
      storageStatus,
      backend,
      benchmarkMs,
      trainingProfile: profile.id,
      modelVersion: MODEL_VERSION,
      architectureVersion: ARCHITECTURE_VERSION,
      targetMode: TARGET_MODE,
      horizon,
      days: requestedDays,
      selectedEpochs,
      completedEpochs,
      trainingDurationMs: Math.round(performance.now() - startedAt),
      tfjsVersion: tf.version.tfjs,
      executionMode: cacheStatus === 'hit' ? 'browser_artifact_loaded' : 'browser_trained',
    };
    if (forecastType === 'direction') {
      const clamp = (value) => Math.min(1, Math.max(0, Number(value)));
      const rawProbabilities = predictedReturns.map(clamp);
      if (!rawProbabilities.every((value) => Number.isFinite(value))) {
        throw new Error('The local direction model produced non-finite probability values.');
      }
      const promotion = evaluatePromotion({
        forecastType,
        metrics,
        evaluation,
        horizon,
      });
      // The fallback baseline must use pre-evaluation prevalence only: the
      // majority class and its positive-class rate are derived from labels
      // strictly before the evaluation window (the holdout split, or the
      // final research fold's validation start, which spans the union of
      // every fold's pre-evaluation labels).
      const sampleCount = sequencePartition(snapshot, forecastType, horizon).sampleCount;
      const preEvaluationEnd = profile.id === 'research'
        ? sampleCount - profile.validationHorizon
        : Math.floor(sampleCount * TRAIN_SPLIT);
      const majority = directionMajority(prepared.targets.slice(0, preEvaluationEnd));
      const baselineFallback = !promotion.promoted;
      const directions = baselineFallback
        ? rawProbabilities.map(() => (majority.label === 1 ? 'Up' : 'Down'))
        : rawProbabilities.map((value) => (value >= 0.5 ? 'Up' : 'Down'));
      const fallbackProbabilities = baselineFallback
        ? rawProbabilities.map(() => majority.rate)
        : rawProbabilities;
      return {
        ...common,
        directions,
        probabilities: fallbackProbabilities,
        baselineFallback,
        promotion,
      };
    }
    const latestClose = Number(snapshot.historical_prices.at(-1));
    const learnedPrices = predictedReturns.map((value) => latestClose * Math.exp(Number(value)));
    const promotion = evaluatePromotion({
      forecastType,
      metrics,
      evaluation,
      horizon,
      predictedCumulativeReturn: Number(predicted[0][horizon - 1]),
      closingPrices: snapshot.historical_prices,
    });
    const baselineFallback = !promotion.promoted;
    const predictedPrices = baselineFallback
      ? buildPersistenceForecast(snapshot.historical_prices, requestedDays)
      : learnedPrices;
    return {
      ...common,
      predictedPrices,
      learnedPrices,
      baselineFallback,
      promotion,
    };
  } finally {
    model?.dispose();
  }
}

let workQueue = Promise.resolve();

function categorizeWorkerError(error) {
  const msg = error instanceof Error ? error.message : String(error || '');
  if (msg.includes('cancelled')) return 'Training was cancelled.';
  if (msg.includes('evaluation') || msg.includes('evaluat')) return 'Model evaluation failed.';
  if (msg.includes('IndexedDB') || msg.includes('storage') || msg.includes('QuotaExceeded')) {
    return 'Local storage unavailable; using this-session-only model.';
  }
  if (msg.includes('memory') || msg.includes('GPU') || msg.includes('OOM') || msg.includes('allocation')) {
    return 'Device ran out of GPU or system memory.';
  }
  if (msg.includes('Cached') || msg.includes('cache')) {
    return 'Cached model could not be loaded and was removed.';
  }
  return 'Browser model training failed.';
}

async function processMessage(event) {
  const { id, type, snapshot, forecastType, days, profile = 'balanced' } = event.data || {};
  if (type === 'clear-cache') {
    try {
      await selectBackend();
      await clearCache();
      emit(id, { type: 'complete', result: { cleared: true } });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Could not clear browser models.';
      emit(id, { type: 'error', message, category: categorizeWorkerError(error) });
    }
    return;
  }
  if (type !== 'forecast' || !id) return;
  try {
    const result = await trainAndPredict(id, snapshot, forecastType, Number(days), profile);
    emit(id, { type: 'complete', result });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Browser training failed.';
    emit(id, { type: 'error', message, category: categorizeWorkerError(error) });
  } finally {
    cancelledIds.delete(id);
  }
}

self.onmessage = (event) => {
  const { id, type } = event.data || {};
  if (type === 'cancel') {
    cancelledIds.add(id);
    return;
  }
  if (type === 'forecast') cancelledIds.delete(id);
  workQueue = workQueue.then(() => processMessage(event), () => processMessage(event));
};

self.onerror = (event) => {
  console.error('[worker-uncaught]', event?.message || String(event));
};
self.onunhandledrejection = (event) => {
  console.error('[worker-unhandled-rejection]', event?.reason?.stack || String(event?.reason));
};
