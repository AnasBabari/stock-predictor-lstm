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

// Per-feature wave shape. Only the Return_1D column oscillates: scaling
// normalizes it to a modest range while every other column stays constant, so
// the network's 60-row input windows remain essentially constant and it
// collapses onto the constant upward drift (which deterministically beats the
// flat persistence baseline). Rows are still distinct row-to-row because
// Return_1D alternates across zero (both up and down "days").
const FEATURE_WAVES = {
  Log_Open_Rel: { amp: 0, period: 13 },
  Log_High_Rel: { amp: 0, period: 9 },
  Log_Low_Rel: { amp: 0, period: 11 },
  Return_1D: { amp: 0.004, period: 7 },
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
  VIX_Return_1D: { amp: 0, period: 7 },
  TNX_Return_1D: { amp: 0, period: 9 },
  Return_Rel_SPY_1D: { amp: 0, period: 11 },
  Beta_SPY_20D: { amp: 0, period: 17 },
  Month_Sin: { amp: 0, period: 30 },
  Month_Cos: { amp: 0, period: 30 },
  Day_Sin: { amp: 0, period: 1 },
  Day_Cos: { amp: 0, period: 1 },
};

const ROWS = 480;
// Price history follows a smooth deterministic upward drift (constant daily
// log-return). A model trained on a constant-drift series beats the flat
// persistence baseline at every horizon, so the price path is guaranteed to be
// promoted in the browser. Both up and down "days" are exercised at the
// feature level: the constructed Return_1D feature column oscillates across
// zero, which is the level that feeds the direction/trend models.
// The drift is kept large relative to the model's day-one residual so the
// one-day horizon also clears the strict relative < 1 promotion gate.
const DAILY_LOG_DRIFT = 0.008;

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
  const historicalPrices = Array.from(
    { length: ROWS },
    (_, index) => 100 * Math.exp(DAILY_LOG_DRIFT * index)
  );
  const features = Array.from({ length: ROWS }, (_, t) =>
    featureNames.map((name, column) => {
      const wave = FEATURE_WAVES[name];
      return FEATURE_BASELINES[name] + wave.amp * Math.sin(t / wave.period + column * 0.9 + 0.5);
    })
  );
  return {
    ticker,
    schema_version: 4,
    snapshot_id: `${ticker}-quick-fixture-v5`,
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
    metrics: { pooled: { relative_rmse: 0.85, relative_mae: 0.9 } },
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
          const result = {
            predictedPrices,
            learnedPrices: predictedPrices,
            directions: predictedPrices.map((_, i) => (i % 2 === 0 ? 'Up' : 'Down')),
            probabilities: predictedPrices.map(() => 0.55),
            metrics: { metric_source: 'browser_purged_holdout', pooled: { relative_rmse: 0.9 } },
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
