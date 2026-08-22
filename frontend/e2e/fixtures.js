export const FEATURE_NAMES_V4 = [
  'Log_Open_Rel', 'Log_High_Rel', 'Log_Low_Rel', 'Return_1D', 'Volume_Log1p_Change',
  'Close_SMA_20', 'Close_EMA_20', 'RSI_14_Centered', 'MACD_Close', 'MACD_Signal_Close',
  'BB_Upper_Rel', 'BB_Lower_Rel', 'ATR_14_Rel', 'OBV_Change_Z',
  'Return_5D', 'Return_20D', 'Realized_Vol_5D', 'Realized_Vol_20D',
  'SPY_Return_1D', 'QQQ_Return_1D', 'VIX_Return_1D', 'TNX_Return_1D',
  'Return_Rel_SPY_1D', 'Beta_SPY_20D',
  'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos',
];

const FEATURE_BASELINES = {
  Log_Open_Rel: 0.0005,
  Log_High_Rel: 0.002,
  Log_Low_Rel: -0.002,
  Return_1D: 0.002,
  Volume_Log1p_Change: 0.001,
  Close_SMA_20: 0.01,
  Close_EMA_20: 0.008,
  RSI_14_Centered: 0.1,
  MACD_Close: 0.0005,
  MACD_Signal_Close: 0.0004,
  BB_Upper_Rel: 0.03,
  BB_Lower_Rel: 0.03,
  ATR_14_Rel: 0.01,
  OBV_Change_Z: 0.1,
  Return_5D: 0.01,
  Return_20D: 0.04,
  Realized_Vol_5D: 0.001,
  Realized_Vol_20D: 0.001,
  SPY_Return_1D: 0.001,
  QQQ_Return_1D: 0.001,
  VIX_Return_1D: 0.0,
  TNX_Return_1D: 0.0,
  Return_Rel_SPY_1D: 0.001,
  Beta_SPY_20D: 1.0,
  Month_Sin: 0.0,
  Month_Cos: 1.0,
  Day_Sin: 0.0,
  Day_Cos: 1.0,
};

// Per-feature wave shape for the constant baseline columns. Return_1D is
// deliberately excluded: it must equal the price history's own 1-day log
// return (see deterministicSnapshot below), so the feature set stays
// internally consistent with the price path the model is fitness-tuned
// against. VIX_Return_1D carries a harmless oscillation so columns retain
// non-zero variance (the scaler) and rows remain distinct row-to-row.
const FEATURE_WAVES = {
  Log_Open_Rel: { amp: 0, period: 13 },
  Log_High_Rel: { amp: 0, period: 9 },
  Log_Low_Rel: { amp: 0, period: 11 },
  Return_1D: { amp: 0, period: 7 },
  Volume_Log1p_Change: { amp: 0, period: 19 },
  Close_SMA_20: { amp: 0, period: 13 },
  Close_EMA_20: { amp: 0, period: 11 },
  RSI_14_Centered: { amp: 0, period: 17 },
  MACD_Close: { amp: 0, period: 23 },
  MACD_Signal_Close: { amp: 0, period: 21 },
  BB_Upper_Rel: { amp: 0, period: 15 },
  BB_Lower_Rel: { amp: 0, period: 15 },
  ATR_14_Rel: { amp: 0, period: 13 },
  OBV_Change_Z: { amp: 0, period: 7 },
  Return_5D: { amp: 0, period: 9 },
  Return_20D: { amp: 0, period: 17 },
  Realized_Vol_5D: { amp: 0, period: 11 },
  Realized_Vol_20D: { amp: 0, period: 13 },
  SPY_Return_1D: { amp: 0, period: 19 },
  QQQ_Return_1D: { amp: 0, period: 21 },
  VIX_Return_1D: { amp: 0.004, period: 11 },
  TNX_Return_1D: { amp: 0, period: 9 },
  Return_Rel_SPY_1D: { amp: 0, period: 11 },
  Beta_SPY_20D: { amp: 0, period: 17 },
  Month_Sin: { amp: 0, period: 30 },
  Month_Cos: { amp: 0, period: 30 },
  Day_Sin: { amp: 0, period: 1 },
  Day_Cos: { amp: 0, period: 1 },
};

const ROWS = 480;
// Price history follows a deterministic upward drift (positive daily
// log-return) plus a small jitter. The model observes past realised returns
// but has no feature that directly reveals the future jitter used in its
// prediction targets, so it converges to the drift mean and clears the flat
// persistence baseline at every horizon with a wide deterministic margin. With
// the fixed fixture and seeded initialization, repeated runs keep the day-1
// relative metric around 0.3, leaving substantial promotion headroom. Every
// daily log return stays positive, so the direction/trend target is 100% Up
// and the trend gate deterministically falls back to the majority class. The
// Return_1D feature column is derived from this exact price path (not an
// arbitrary wave), keeping the fixture internally consistent, while the jitter
// widens the 99.5th percentile of observed horizon returns far above the
// learned forecast so the volatility sanity gate has headroom.
const DAILY_LOG_DRIFT = 0.008;
const JITTER_AMPLITUDE = 0.003;

// Deterministic for the supported JavaScript runtime and test environment
// (``Math.sin`` hash). It carries no trainable structure the model could pick
// up through the otherwise constant feature columns: past realised jitter is
// visible to the network, but future (target) jitter is not.
function deterministicJitter(index) {
  const x = Math.sin(index * 12.9898 + 78.2331) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

function dailyReturnAt(index) {
  return DAILY_LOG_DRIFT + deterministicJitter(index) * JITTER_AMPLITUDE;
}

/**
 * Returns `count` consecutive business days (Mon-Fri) strictly after
 * `startIso`. The generator walks the UTC calendar deterministically.
 */
export function businessDatesAfter(startIso, count) {
  const dates = [];
  const cursor = new Date(`${startIso}T00:00:00Z`);
  while (dates.length < count) {
    cursor.setUTCDate(cursor.getUTCDate() + 1);
    const weekday = cursor.getUTCDay();
    if (weekday === 0 || weekday === 6) continue;
    dates.push(cursor.toISOString().slice(0, 10));
  }
  return dates;
}

export function deterministicSnapshot(ticker = 'MSFT') {
  const featureNames = Object.keys(FEATURE_BASELINES);
  const dates = businessDatesAfter('2024-01-01', ROWS);
  const futureDates = businessDatesAfter(dates[dates.length - 1], 30);
  const historicalPrices = [100];
  for (let index = 1; index < ROWS; index += 1) {
    historicalPrices.push(
      historicalPrices[index - 1] * Math.exp(dailyReturnAt(index - 1))
    );
  }
  const features = Array.from({ length: ROWS }, (_, t) =>
    featureNames.map((name, column) => {
      if (name === 'Return_1D') {
        // Exactly the price path's own 1-day log return, so the feature column
        // is internally consistent with the prices the model is asked to
        // predict. Row 0 has no prior day (hence 0).
        return t === 0 ? 0 : dailyReturnAt(t - 1);
      }
      const wave = FEATURE_WAVES[name];
      return FEATURE_BASELINES[name] + wave.amp * Math.sin(t / wave.period + column * 0.9 + 0.5);
    })
  );
  return {
    ticker,
    schema_version: 4,
    snapshot_id: `${ticker}-quick-fixture-v6`,
    feature_names: featureNames,
    window_size: 60,
    output_width: 30,
    dates,
    future_dates: futureDates,
    historical_prices: historicalPrices,
    features,
  };
}

export function serverForecastPayload(ticker = 'MSFT', days = 7, origin = '2026-08-05') {
  const historyLength = 120;
  const originDate = new Date(`${origin}T00:00:00Z`);
  const originIndex = 479;
  const lastClose = 100 * Math.exp(0.002 * originIndex);
  const toIso = (offset) => {
    const d = new Date(originDate);
    d.setUTCDate(d.getUTCDate() + offset);
    return d.toISOString().slice(0, 10);
  };
  return {
    available: true,
    ticker,
    forecast_days: days,
    future_dates: Array.from({ length: days }, (_, index) => toIso(index + 1)),
    predicted_prices: Array.from({ length: days }, (_, index) => lastClose * (1 + 0.004 * (index + 1))),
    historical_dates: Array.from({ length: historyLength }, (_, index) => toIso(index - historyLength + 1)),
    historical_prices: Array.from({ length: historyLength }, (_, index) =>
      lastClose * (1 - 0.0025 * (historyLength - 1 - index))
    ),
    metrics: {
      metric_source: 'server_purged_walk_forward',
      metric_scope: 'forecast_origin_horizon_pairs',
      family: 'elastic_net',
      target_mode: 'cumulative_log_return_v1',
      horizon: days,
      mae: 0.004,
      mse: 0.00003,
      rmse: 0.0055,
      mape: 0.005,
      r2: 0.97,
      relative_mae: 0.45,
      relative_rmse: 0.4,
      directional_accuracy: 0.6,
      per_horizon: Array.from({ length: days }, (_, index) => ({
        horizon: index + 1,
        rows: 160,
        mae: 0.004,
        rmse: 0.0055,
        relative_mae: 0.45,
        relative_rmse: 0.4,
        directional_accuracy: 0.6,
      })),
      evaluation_rows: 160,
    },
    metadata: {
      engine: {
        role: 'server_pretrained',
        family: 'elastic_net',
        version_id: `${ticker}-price-20260805T120000Z-0123456789ab-0000abcd`,
      },
      metric_source: 'server_purged_walk_forward',
      browser_training: false,
      trained_at: `${origin}T12:00:00Z`,
      origin: { date: origin, close: lastClose },
      authenticity: 'ed25519_verified',
    },
  };
}

/**
 * Replaces the browser-training Web Worker with a deterministic stub so e2e
 * contract tests never build a real TF.js model (slow, non-deterministic).
 * The stub answers every `forecast` message with a canned canonical result
 * after a microtask, so the app's browser path renders exactly as if training
 * had run. Real-training coverage lives in browser-real-training.spec.js.
 */
export function installStubBrowserWorker(page) {
  return page.addInitScript(() => {
    if (globalThis.__STOCKLSTM_WORKER_FACTORY__) return;
    globalThis.__STOCKLSTM_WORKER_FACTORY__ = () => {
      const fake = {};
      fake.postMessage = (msg) => {
        if (!msg || typeof msg.type !== 'string') return;
        if (msg.type === 'forecast') {
          const { id, snapshot, forecastType, days, profile } = msg;
          const last = snapshot.historical_prices?.[snapshot.historical_prices.length - 1] ?? 100;
          const count = Number(days) || 7;
          const predictedPrices = Array.from(
            { length: count },
            (_, i) => last * (1 + 0.005 * (i + 1))
          );
          const isTrend = forecastType === 'trend';
          const result = isTrend
            ? {
                direction_horizon_days: count,
                direction: 'Up',
                direction_probabilities: [0.15, 0.25, 0.6],
                model_direction_probabilities: [0.15, 0.25, 0.6],
                base_rate_direction_probabilities: [0.33, 0.33, 0.34],
                forecast_status: { state: 'promoted', decision: 'model', label: 'Promoted' },
                metrics: {
                  metric_source: 'browser_purged_holdout',
                  brier_skill: 0.12,
                  log_loss: 0.85,
                  macro_f1: 0.62,
                  macro_balanced_accuracy: 0.65,
                  expected_calibration_error: 0.04,
                  baseline_probabilities: [0.33, 0.33, 0.34],
                },
                trainingProfile: profile,
                cacheStatus: 'miss',
                backend: 'stub',
                executionMode: 'trained',
                modelVersion: 'tfjs-return-lstm-v4',
                architectureVersion: 'v4',
                targetMode: 'cumulative_three_way_v2',
                trainingDurationMs: 1,
                selectedEpochs: 1,
                completedEpochs: 1,
                tfjsVersion: 'stub',
                storageStatus: 'none',
                evaluation: {},
                promotion: { promoted: true },
                horizon: count,
                forecastType,
              }
            : {
                predictedPrices,
                learnedPrices: predictedPrices,
                metrics: {
                  metric_source: 'browser_purged_holdout',
                  mae: 0.02,
                  rmse: 0.03,
                  relative_mae: 0.85,
                  relative_rmse: 0.82,
                  per_horizon: Array.from({ length: count }, (_, i) => ({
                    horizon: i + 1,
                    mae: 0.02,
                    rmse: 0.03,
                    relative_rmse: 0.82,
                    directional_accuracy: 0.6,
                    rows: 100,
                  })),
                },
                trainingProfile: profile,
                cacheStatus: 'miss',
                backend: 'stub',
                executionMode: 'trained',
                modelVersion: 'tfjs-return-lstm-v4',
                architectureVersion: 'v4',
                targetMode: 'returns',
                trainingDurationMs: 1,
                selectedEpochs: 1,
                completedEpochs: 1,
                tfjsVersion: 'stub',
                storageStatus: 'none',
                evaluation: {},
                promotion: { promoted: true },
                horizon: snapshot.output_width,
                forecastType,
              };
          setTimeout(() => fake.onmessage?.({ data: { id, type: 'complete', result } }), 0);
        } else if (msg.type === 'clear-cache') {
          setTimeout(
            () => fake.onmessage?.({ data: { id: msg.id, type: 'complete', result: { cleared: true } } }),
            0
          );
        }
      };
      fake.terminate = () => {};
      return fake;
    };
  });
}

// Unlearnable fixture: identical causal feature waves to the learnable one,
// but the close path is a seeded random walk. No architecture can beat
// persistence on it, so the trained model must fail promotion — exercising
// the fallback-labelling contract end to end.
export function rejectedForecastSnapshot(ticker = 'IGC') {
  const snapshot = deterministicSnapshot(ticker);
  let state = 987654321;
  const nextRandom = () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648 - 0.5;
  };
  const prices = [snapshot.historical_prices[0]];
  for (let i = 1; i < snapshot.historical_prices.length; i += 1) {
    prices.push(Math.max(1, prices[i - 1] * Math.exp(nextRandom() * 0.02)));
  }
  return { ...snapshot, ticker, historical_prices: prices };
}
