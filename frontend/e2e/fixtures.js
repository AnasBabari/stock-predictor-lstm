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

export function serverForecastPayload(ticker = 'MSFT', days = 7) {
  const lastClose = 100 * Math.exp(0.002 * 479);
  return {
    available: true,
    ticker,
    forecast_days: days,
    future_dates: Array.from({ length: days }, (_, index) => `2026-08-${String(index + 1).padStart(2, '0')}`),
    predicted_prices: Array.from({ length: days }, (_, index) => lastClose * (1 + 0.004 * (index + 1))),
    historical_dates: Array.from({ length: 120 }, (_, index) => `2026-03-${String((index % 28) + 1).padStart(2, '0')}`),
    historical_prices: Array.from({ length: 120 }, (_, index) => 100 * Math.exp(0.002 * (359 + index))),
    metrics: { pooled: { relative_rmse: 0.85, relative_mae: 0.9 } },
    metadata: {
      engine: {
        role: 'server_pretrained',
        family: 'elastic_net',
        version_id: `${ticker}-price-20260805T120000Z-0123456789ab-0000abcd`,
      },
      metric_source: 'server_purged_walk_forward',
      browser_training: false,
      trained_at: '2026-08-05T12:00:00Z',
      origin: { date: '2026-08-05', close: lastClose },
      authenticity: 'sha256_only',
    },
  };
}