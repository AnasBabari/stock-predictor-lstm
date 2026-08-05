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

export function deterministicSnapshot(ticker = 'MSFT') {
  const rows = 480;
  const featureNames = Object.keys(FEATURE_BASELINES);
  const historicalPrices = Array.from({ length: rows }, (_, index) =>
    100 * Math.exp(0.002 * index)
  );
  return {
    ticker,
    schema_version: 4,
    snapshot_id: `${ticker}-quick-fixture-v5`,
    feature_names: featureNames,
    window_size: 60,
    output_width: 30,
    dates: Array.from({ length: rows }, (_, index) => `2026-01-${String((index % 28) + 1).padStart(2, '0')}`),
    future_dates: Array.from({ length: 30 }, (_, index) => `2026-08-${String(index + 1).padStart(2, '0')}`),
    historical_prices: historicalPrices,
    features: Array.from({ length: rows }, () => featureNames.map((name) => FEATURE_BASELINES[name])),
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
 * had run. Real-training coverage lives in vercel-preview.spec.js.
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