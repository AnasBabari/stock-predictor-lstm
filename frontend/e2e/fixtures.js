export function deterministicSnapshot(ticker = 'MSFT') {
  const featureNames = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'RSI_14',
    'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR_14', 'OBV',
    'SPY_Return_1D', 'QQQ_Return_1D', 'VIX_Return_1D', 'TNX_Return_1D',
    'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos',
  ];
  const rows = 240;
  return {
    ticker,
    schema_version: 3,
    snapshot_id: `${ticker}-quick-fixture-v1`,
    feature_names: featureNames,
    window_size: 60,
    output_width: 30,
    close_index: 3,
    dates: Array.from({ length: rows }, (_, index) => `2026-01-${String((index % 28) + 1).padStart(2, '0')}`),
    future_dates: Array.from({ length: 30 }, (_, index) => `2026-08-${String(index + 1).padStart(2, '0')}`),
    historical_prices: Array.from({ length: rows }, (_, index) => 100 + index * 0.35 + Math.sin(index / 4)),
    features: Array.from({ length: rows }, (_, index) => featureNames.map((_, featureIndex) => {
      if (featureIndex === 3) return 100 + index * 0.35 + Math.sin(index / 4);
      return Number((1 + featureIndex * 0.1 + index * 0.01 + Math.sin((index + featureIndex) / 7)).toFixed(6));
    })),
  };
}
