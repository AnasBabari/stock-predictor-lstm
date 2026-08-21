export const MODEL_VERSION = 'tfjs-return-lstm-v4';
export const ARCHITECTURE_VERSION = 'local-return-lstm-v3';
export const TARGET_MODE = 'cumulative_log_return_v1';
// Direction target contract v2: ONE three-way call per forecast origin —
// the sign of the CUMULATIVE return over the requested horizon with a
// volatility/tick-aware neutral band — replacing the former h independent
// next-day signs that let "7-day direction" mean seven daily calls.
// Version-gated separately from TARGET_MODE so adopting it invalidates only
// cached direction artifacts.
export const DIRECTION_TARGET_VERSION = 'cumulative_three_way_v2';
export const WINDOW_SIZE = 60;
export const OUTPUT_WIDTH = 30;
export const HORIZONS = [1, 3, 5, 7, 14, 30];
export const TRAIN_SPLIT = 0.8;
export const VALIDATION_FRACTION = 0.2;
export const FEATURE_SCHEMA_VERSION = 4;
export const FEATURE_NAMES = [
  'Log_Open_Rel',
  'Log_High_Rel',
  'Log_Low_Rel',
  'Return_1D',
  'Volume_Log1p_Change',
  'Close_SMA_20',
  'Close_EMA_20',
  'RSI_14_Centered',
  'MACD_Close',
  'MACD_Signal_Close',
  'BB_Upper_Rel',
  'BB_Lower_Rel',
  'ATR_14_Rel',
  'OBV_Change_Z',
  'Return_5D',
  'Return_20D',
  'Realized_Vol_5D',
  'Realized_Vol_20D',
  'SPY_Return_1D',
  'QQQ_Return_1D',
  'VIX_Return_1D',
  'TNX_Return_1D',
  'Return_Rel_SPY_1D',
  'Beta_SPY_20D',
  'Month_Sin',
  'Month_Cos',
  'Day_Sin',
  'Day_Cos',
];

function finite(value) {
  return Number.isFinite(Number(value));
}

function isoTimestamp(value) {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : null;
}

function isStrictlyIncreasing(dates) {
  for (let index = 1; index < dates.length; index += 1) {
    const prev = isoTimestamp(dates[index - 1]);
    const next = isoTimestamp(dates[index]);
    if (prev == null || next == null || prev >= next) return false;
  }
  return true;
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
  if (!Array.isArray(snapshot.features) || !Array.isArray(snapshot.dates)) {
    throw new Error('Training feature data is incomplete.');
  }
  if (snapshot.features.length !== snapshot.dates.length) {
    throw new Error('Training feature dates and rows do not match.');
  }
  if (!snapshot.dates.every((date) => typeof date === 'string' && date.length > 0)) {
    throw new Error('Training feature dates are invalid.');
  }
  if (!isStrictlyIncreasing(snapshot.dates)) {
    throw new Error('Training feature dates are not in strict chronological order.');
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
  if (!isStrictlyIncreasing(snapshot.future_dates)) {
    throw new Error('Training future dates are not in strict chronological order.');
  }
  const lastTradingDay = isoTimestamp(snapshot.dates[snapshot.dates.length - 1]);
  const firstForecastDay = isoTimestamp(snapshot.future_dates[0]);
  if (lastTradingDay == null || firstForecastDay == null || firstForecastDay <= lastTradingDay) {
    throw new Error('Training future dates do not follow the last trading day.');
  }
}

export function resolveHorizon(days) {
  const requested = Math.max(1, Math.min(OUTPUT_WIDTH, Math.round(Number(days) || 1)));
  return HORIZONS.find((horizon) => horizon >= requested) ?? OUTPUT_WIDTH;
}

export function fitRobustScaler(rows, endExclusive) {
  const limit = Math.max(2, Math.min(endExclusive == null ? rows.length : endExclusive, rows.length));
  const count = rows[0].length;
  const median = [];
  const iqr = [];
  for (let column = 0; column < count; column += 1) {
    const values = rows.slice(0, limit).map((row) => Number(row[column])).sort((a, b) => a - b);
    const quantile = (position) => {
      const index = position * (values.length - 1);
      const lower = Math.floor(index);
      const upper = Math.ceil(index);
      if (lower === upper) return values[lower];
      return values[lower] + (values[upper] - values[lower]) * (index - lower);
    };
    const med = quantile(0.5);
    const spread = Math.max(quantile(0.75) - quantile(0.25), 1e-12);
    median.push(med);
    iqr.push(spread);
  }
  return { median, iqr };
}

export function scaleRows(rows, scaler) {
  return rows.map((row) => row.map((value, column) => (Number(value) - scaler.median[column]) / scaler.iqr[column]));
}

// A zero-based sequence j consumes raw rows [j, j + WINDOW_SIZE).
// A fitting sequence interval [fitSequenceStart, fitSequenceEndExclusive)
// therefore reaches raw rows from fitSequenceStart through
// (fitSequenceEndExclusive - 1) + WINDOW_SIZE, exclusive.
export function fittingScalerBounds(fitSequenceStart, fitSequenceEndExclusive, windowSize = WINDOW_SIZE) {
  const fitSequenceCount = fitSequenceEndExclusive - fitSequenceStart;
  if (fitSequenceCount <= 0) {
    return {
      fitSequenceStart,
      fitSequenceEndExclusive,
      fitSequenceCount: 0,
      scalerRawStart: fitSequenceStart,
      scalerRawEndExclusive: fitSequenceStart,
      hasFittingSequences: false,
    };
  }
  return {
    fitSequenceStart,
    fitSequenceEndExclusive,
    fitSequenceCount,
    scalerRawStart: fitSequenceStart,
    scalerRawEndExclusive: (fitSequenceEndExclusive - 1) + windowSize,
    hasFittingSequences: true,
  };
}

export function inverseRobust(value, scaler, column) {
  return Number(value) * scaler.iqr[column] + scaler.median[column];
}

// Sequence-partition arithmetic, shared by price and direction preprocessing.
// Direction consumes the shifted raw-feature matrix (features.slice(1)), so its
// sequence count is one shorter than the price coordinate system and every
// direction boundary sits exactly one raw row after the corresponding price
// boundary. The direction split is therefore DERIVED from the price split
// (split_direction = split_price - 1) instead of being floored independently;
// independent flooring diverges from the price boundary whenever
// priceSampleCount is not a multiple of 5, which would give the two forecast
// types different temporal train/holdout eras for ~20% of snapshot lengths.
export function sequencePartition(snapshot, forecastType, horizon = OUTPUT_WIDTH) {
  const h = Math.max(1, Math.min(OUTPUT_WIDTH, Math.round(Number(horizon) || OUTPUT_WIDTH)));
  const priceRowCount = snapshot.features.length;
  const priceSampleCount = priceRowCount - WINDOW_SIZE - h + 1;
  if (priceSampleCount <= 0) throw new Error('Not enough rows for browser training.');
  const isDirection = forecastType === 'direction';
  const rowCount = isDirection ? priceRowCount - 1 : priceRowCount;
  const sampleCount = rowCount - WINDOW_SIZE - h + 1;
  const split = Math.floor(priceSampleCount * TRAIN_SPLIT) - (isDirection ? 1 : 0);
  const trainCount = split - h + 1;
  if (trainCount < 1 || split >= sampleCount) throw new Error('Training split is too small.');
  return { sampleCount, split, trainCount, horizon: h };
}

export function preparePriceData(snapshot, fitSequenceEndExclusive, horizon = OUTPUT_WIDTH) {
  const partition = sequencePartition(snapshot, 'price', horizon);
  const { sampleCount, split, trainCount, horizon: h } = partition;
  const rows = snapshot.features.map((row) => row.map(Number));
  const closes = snapshot.historical_prices.map(Number);
  const bounds = fittingScalerBounds(0, fitSequenceEndExclusive ?? trainCount);
  if (!bounds.hasFittingSequences) throw new Error('No fitting sequences are available for scaler bounds.');
  const scaler = fitRobustScaler(rows, bounds.scalerRawEndExclusive);
  const scaled = scaleRows(rows, scaler);
  const inputs = [];
  const targets = [];
  const origins = [];
  for (let index = WINDOW_SIZE; index < WINDOW_SIZE + sampleCount; index += 1) {
    inputs.push(scaled.slice(index - WINDOW_SIZE, index));
    targets.push(closes.slice(index, index + h).map((close, step) => Math.log(close / closes[index - 1])));
    origins.push(closes[index - 1]);
  }
  return { inputs, targets, origins, scaler, split, trainCount, scaled, horizon: h, closes, ...bounds };
}

export function prepareDirectionData(snapshot, fitSequenceEndExclusive, horizon = OUTPUT_WIDTH) {
  const partition = sequencePartition(snapshot, 'direction', horizon);
  const { sampleCount, split, trainCount, horizon: h } = partition;
  const rawRows = snapshot.features.slice(1).map((row) => row.map(Number));
  const prices = snapshot.historical_prices.map(Number);
  const returns = prices.slice(1).map((price, index) => Math.log(price / prices[index]));
  const bounds = fittingScalerBounds(0, fitSequenceEndExclusive ?? trainCount);
  if (!bounds.hasFittingSequences) throw new Error('No fitting sequences are available for scaler bounds.');
  const scaler = fitRobustScaler(rawRows, bounds.scalerRawEndExclusive);
  const scaled = scaleRows(rawRows, scaler);
  const inputs = [];
  const targets = [];
  for (let index = WINDOW_SIZE; index < WINDOW_SIZE + sampleCount; index += 1) {
    inputs.push(scaled.slice(index - WINDOW_SIZE, index));
    targets.push(returns.slice(index, index + h).map((value) => (value > 0 ? 1 : 0)));
  }
  return { inputs, targets, scaler, split, trainCount, scaled, horizon: h, ...bounds };
}

export function featureSignature(featureNames) {
  let hash = 2166136261;
  for (const character of featureNames.join('\u001f')) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

export function modelKey(snapshot, forecastType, profile = 'balanced', backend = 'any', horizon = OUTPUT_WIDTH) {
  const parts = [
    MODEL_VERSION,
    ARCHITECTURE_VERSION,
    TARGET_MODE,
    snapshot.schema_version,
    snapshot.ticker,
    forecastType,
    profile,
    backend,
    featureSignature(snapshot.feature_names),
    snapshot.snapshot_id,
    WINDOW_SIZE,
    horizon,
  ];
  // Direction artifacts additionally carry the direction target version so
  // switching to cumulative three-way labels can never load a legacy
  // per-day-sign model (and vice versa).
  if (forecastType === 'direction') parts.push(DIRECTION_TARGET_VERSION);
  return parts.join('/');
}

export function latestInput(prepared) {
  return prepared.scaled.slice(-WINDOW_SIZE);
}

// ── Direction target contract v2 (cumulative three-way) ─────────────────
// Spec §3.2: y = Up if r_cum > τ, Down if r_cum < −τ, Neutral otherwise,
// with τ computed ONLY from information available at the forecast origin.
// τ = max(0.0005, 0.10 · σ20,t · √h, tick/P_t). The browser cannot observe
// exchange tick sizes yet, so callers pass tickPrice = null and the term is
// dropped; when the backend snapshot starts carrying minimum tick, the same
// helper consumes it without signature change.

export function trailingSigma20(prices, originIndex) {
  const endExclusive = Math.max(0, Number(originIndex)); // returns up to origin-1 vs origin-2 … causal
  const windowStart = Math.max(1, endExclusive - 20);
  const returns = [];
  for (let i = windowStart; i < endExclusive; i += 1) {
    const prev = Number(prices[i - 1]);
    const cur = Number(prices[i]);
    if (prev > 0 && cur > 0 && Number.isFinite(prev) && Number.isFinite(cur)) {
      returns.push(Math.log(cur / prev));
    }
  }
  return standardDeviationOf(returns);
}

function standardDeviationOf(values) {
  if (values.length < 2) return 0;
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  return Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1));
}

export function directionThreshold({ sigma20, horizon, price, tickPrice = null }) {
  const h = Math.max(1, Math.round(Number(horizon) || 1));
  const sigma = Math.max(0, Number(sigma20) || 0);
  const floor = 0.0005;
  const volTerm = 0.10 * sigma * Math.sqrt(h);
  let tau = Math.max(floor, volTerm);
  const p = Number(price);
  const tick = Number(tickPrice);
  if (Number.isFinite(tick) && tick > 0 && Number.isFinite(p) && p > 0) {
    tau = Math.max(tau, tick / p);
  }
  return tau;
}

/**
 * Three-way cumulative direction labels for every valid forecast origin.
 *
 * Origin t is the last row of a lookback window; its label uses the close at
 * t and the close at t+h only. Returns one entry per origin:
 *   { originIndex, horizon, sigma20, tau, cumulativeReturn, label }
 * where label ∈ 'up' | 'down' | 'neutral'. Rows before the first index with
 * enough history for σ20 are skipped (origins start at index 21).
 */
export function buildCumulativeDirectionTargets(historicalPrices, horizon, tickPrice = null) {
  const prices = (historicalPrices || []).map(Number);
  const h = Math.max(1, Math.round(Number(horizon) || 1));
  const labels = [];
  for (let origin = 21; origin + h < prices.length; origin += 1) {
    const base = prices[origin];
    const future = prices[origin + h];
    if (!(base > 0 && future > 0 && Number.isFinite(base) && Number.isFinite(future))) continue;
    const sigma20 = trailingSigma20(prices, origin);
    const tau = directionThreshold({ sigma20, horizon: h, price: base, tickPrice });
    const cumulativeReturn = Math.log(future / base);
    const label =
      cumulativeReturn > tau ? 'up' : cumulativeReturn < -tau ? 'down' : 'neutral';
    labels.push({ originIndex: origin, horizon: h, sigma20, tau, cumulativeReturn, label });
  }
  return labels;
}

// Holdout comparison series (price forecasts, single-holdout profiles only).
// For each evaluation origin, take the FINAL step of the horizon bucket and
// align actual / model / persistence prices on the true target date. Dates
// come from the backend snapshot so the chart can never invent timestamps.
// Research-profile fold aggregation is intentionally not supported here:
// overlapping windows have no single aligned timeline.
export function buildEvaluationSeries(
  snapshot,
  selection,
  evaluated,
  horizon,
  cap = 240
) {
  const h = Math.max(1, Math.min(OUTPUT_WIDTH, Math.round(Number(horizon) || OUTPUT_WIDTH)));
  const { actualPrices, predictedPrices, persistencePrices } = evaluated;
  if (!Array.isArray(actualPrices) || !Array.isArray(predictedPrices) || !Array.isArray(persistencePrices)) {
    return null;
  }
  if (!(actualPrices.length && actualPrices.length === predictedPrices.length && actualPrices.length === persistencePrices.length)) {
    return null;
  }
  const totalDates = snapshot.dates.length;
  const points = [];
  for (let sample = 0; sample < actualPrices.length; sample += 1) {
    const sequenceIndex = selection.split + sample;
    const targetRow = WINDOW_SIZE + sequenceIndex + h - 1;
    if (targetRow >= totalDates) break; // never fabricate dates beyond data
    const stepIndex = Math.min(h - 1, actualPrices[sample].length - 1);
    const actual = Number(actualPrices[sample][stepIndex]);
    const model = Number(predictedPrices[sample][stepIndex]);
    const persistence = Number(persistencePrices[sample][stepIndex]);
    if (![actual, model, persistence].every(Number.isFinite)) continue;
    points.push({ date: snapshot.dates[targetRow], actual, model, persistence });
  }
  if (!points.length) return null;
  return {
    horizon: h,
    step: h - 1,
    metric_scope: 'untouched_post_purge_holdout',
    // Keep the most recent `cap` origins for chart readability.
    ...(points.length > cap ? { truncated: points.length - cap } : {}),
    dates: points.slice(-cap).map((point) => point.date),
    actual: points.slice(-cap).map((point) => point.actual),
    model: points.slice(-cap).map((point) => point.model),
    persistence: points.slice(-cap).map((point) => point.persistence),
  };
}
