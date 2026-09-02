/**
 * Public volatility-serving contract. These horizons are the only ones
 * covered by the frozen empirical benchmark and the live forward ledger.
 */
export const VOLATILITY_HORIZONS = Object.freeze([1, 5, 10, 20]);
export const DEFAULT_VOLATILITY_HORIZON = 5;
