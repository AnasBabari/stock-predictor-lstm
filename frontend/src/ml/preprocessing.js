export const MODEL_VERSION = 'tfjs-lstm-v2';
export const ARCHITECTURE_VERSION = 'local-lstm-profiles-v1';
export const WINDOW_SIZE = 60;
export const OUTPUT_WIDTH = 30;
export const TRAIN_SPLIT = 0.8;
export const VALIDATION_FRACTION = 0.2;
export const FEATURE_SCHEMA_VERSION = 3;
export const FEATURE_NAMES = [
  'Open',
  'High',
  'Low',
  'Close',
  'Volume',
  'SMA_20',
  'EMA_20',
  'RSI_14',
  'MACD',
  'MACD_Signal',
  'BB_Upper',
  'BB_Lower',
  'ATR_14',
  'OBV',
  'SPY_Return_1D',
  'QQQ_Return_1D',
  'VIX_Return_1D',
  'TNX_Return_1D',
  'Month_Sin',
  'Month_Cos',
  'Day_Sin',
  'Day_Cos',
];
function finite(value) {
  return Number.isFinite(Number(value));
}

export function validateSnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== 'object') throw new Error('Training data is unavailable.');
  if (typeof snapshot.ticker !== 'string' || !snapshot.ticker.trim()) {
    throw new Error('Training ticker is missing.');
  }
  if (typeof snapshot.snapshot_id !== 'string' || !snapshot.snapshot_id.trim()) {
    throw new Error('Training snapshot identity is missing.');
  }
  if (!Array.isArray(snapshot.feature_names) || JSON.stringify(snapshot.feature_names) !== JSON.stringify(FEATURE_NAMES)) {
    throw new Error('Training feature schema is incompatible.');
  }
  if (Number(snapshot.schema_version) !== FEATURE_SCHEMA_VERSION) {
    throw new Error('Training feature schema version is incompatible.');
  }
  if (Number(snapshot.window_size) !== WINDOW_SIZE || Number(snapshot.output_width) !== OUTPUT_WIDTH) {
    throw new Error('Training window/output contract is incompatible.');
  }
  if (Number(snapshot.close_index) !== FEATURE_NAMES.indexOf('Close')) {
    throw new Error('Training close-price index is incompatible.');
  }
  if (!Array.isArray(snapshot.features) || !Array.isArray(snapshot.dates)) {
    throw new Error('Training feature data is incomplete.');
  }
  if (snapshot.features.length !== snapshot.dates.length) {
    throw new Error('Training feature dates and rows do not match.');
  }
  if (!snapshot.dates.every((date) => typeof date === 'string' && date.length > 0)) {
    throw new Error('Training feature dates are invalid.');
  }
  if (snapshot.features.length < WINDOW_SIZE + OUTPUT_WIDTH + 1) {
    throw new Error('Not enough historical data for browser training.');
  }
  if (!snapshot.features.every((row) => Array.isArray(row) && row.length === FEATURE_NAMES.length && row.every(finite))) {
    throw new Error('Training feature data contains invalid values.');
  }
  if (!Array.isArray(snapshot.historical_prices) || snapshot.historical_prices.length !== snapshot.features.length) {
    throw new Error('Training price history is incomplete.');
  }
  if (!snapshot.historical_prices.every((value) => finite(value) && Number(value) > 0)) {
    throw new Error('Training price history contains invalid values.');
  }
  if (!Array.isArray(snapshot.future_dates) || snapshot.future_dates.length < OUTPUT_WIDTH ||
      !snapshot.future_dates.every((date) => typeof date === 'string' && date.length > 0)) {
    throw new Error('Training future dates are invalid.');
  }
}

export function fitMinMax(rows, endExclusive) {
  const limit = Math.max(1, Math.min(endExclusive, rows.length));
  const count = rows[0].length;
  const min = Array.from({ length: count }, () => Infinity);
  const max = Array.from({ length: count }, () => -Infinity);
  for (let index = 0; index < limit; index += 1) {
    rows[index].forEach((value, column) => {
      min[column] = Math.min(min[column], Number(value));
      max[column] = Math.max(max[column], Number(value));
    });
  }
  return { min, max };
}

export function scaleRows(rows, scaler) {
  return rows.map((row) => row.map((value, column) => {
    const range = scaler.max[column] - scaler.min[column];
    return range === 0 ? 0 : (Number(value) - scaler.min[column]) / range;
  }));
}

export function inverseClose(value, scaler, closeIndex) {
  const range = scaler.max[closeIndex] - scaler.min[closeIndex];
  return Number(value) * range + scaler.min[closeIndex];
}

function splitCount(rowCount) {
  const sampleCount = rowCount - WINDOW_SIZE - OUTPUT_WIDTH + 1;
  if (sampleCount <= 0) throw new Error('Not enough rows for browser training.');
  const split = Math.floor(sampleCount * TRAIN_SPLIT);
  const trainCount = split - OUTPUT_WIDTH + 1;
  if (trainCount < 1 || split >= sampleCount) throw new Error('Training split is too small.');
  return { sampleCount, split, trainCount };
}

export function preparePriceData(snapshot, scalerEnd) {
  const rows = snapshot.features.map((row) => row.map(Number));
  const { sampleCount, split, trainCount } = splitCount(rows.length);
  const scaler = fitMinMax(rows, scalerEnd ?? split + WINDOW_SIZE);
  const scaled = scaleRows(rows, scaler);
  const closeIndex = Number(snapshot.close_index);
  const inputs = [];
  const targets = [];
  const origins = [];
  for (let index = WINDOW_SIZE; index < WINDOW_SIZE + sampleCount; index += 1) {
    inputs.push(scaled.slice(index - WINDOW_SIZE, index));
    targets.push(scaled.slice(index, index + OUTPUT_WIDTH).map((row) => row[closeIndex]));
    origins.push(scaled[index - 1][closeIndex]);
  }
  return { inputs, targets, origins, scaler, split, trainCount, scaled, closeIndex };
}

export function prepareDirectionData(snapshot, scalerEnd) {
  const rawRows = snapshot.features.slice(1).map((row) => row.map(Number));
  const prices = snapshot.historical_prices.map(Number);
  const returns = prices.slice(1).map((price, index) => Math.log(price / prices[index]));
  const { sampleCount, split, trainCount } = splitCount(rawRows.length);
  const scaler = fitMinMax(rawRows, scalerEnd ?? split + WINDOW_SIZE);
  const scaled = scaleRows(rawRows, scaler);
  const inputs = [];
  const targets = [];
  const origins = [];
  for (let index = WINDOW_SIZE; index < WINDOW_SIZE + sampleCount; index += 1) {
    inputs.push(scaled.slice(index - WINDOW_SIZE, index));
    targets.push(returns.slice(index, index + OUTPUT_WIDTH).map((value) => (value > 0 ? 1 : 0)));
    origins.push(scaled[index - 1][Number(snapshot.close_index)]);
  }
  return { inputs, targets, origins, scaler, split, trainCount, scaled, closeIndex: Number(snapshot.close_index) };
}

export function featureSignature(featureNames) {
  let hash = 2166136261;
  for (const character of featureNames.join('\u001f')) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

export function modelKey(snapshot, forecastType, profile = 'balanced', backend = 'any') {
  return [
    MODEL_VERSION,
    ARCHITECTURE_VERSION,
    snapshot.schema_version,
    snapshot.ticker,
    forecastType,
    profile,
    backend,
    featureSignature(snapshot.feature_names),
    snapshot.snapshot_id,
    WINDOW_SIZE,
    OUTPUT_WIDTH,
  ].join('/');
}

export function latestInput(prepared) {
  return prepared.scaled.slice(-WINDOW_SIZE);
}
