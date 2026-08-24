/**
 * Shared financial and metric formatters for StockLSTM.
 *
 * Implements adaptive precision:
 * - >= $100: 2 decimals ($200.00)
 * - >= $1.00: 2 decimals ($15.25)
 * - $0.01 to $1.00: 3 decimals ($0.310, $0.035)
 * - < $0.01: 4 to 6 significant decimals ($0.0042)
 */

export function formatPrice(value, { fallback = '—', prefix = '$' } = {}) {
  if (value == null || value === '') return fallback;
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  const abs = Math.abs(num);
  const sign = num < 0 ? '-' : '';

  let formatted;
  if (abs >= 1) {
    formatted = abs.toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  } else if (abs >= 0.01) {
    formatted = abs.toLocaleString('en-US', {
      minimumFractionDigits: 3,
      maximumFractionDigits: 3,
    });
  } else if (abs > 0) {
    formatted = abs.toLocaleString('en-US', {
      minimumFractionDigits: 3,
      maximumFractionDigits: 6,
    });
  } else {
    formatted = '0.00';
  }

  return `${sign}${prefix}${formatted}`;
}

export function formatPercent(value, { fallback = '—', decimals = 2, includePlus = true } = {}) {
  if (value == null || value === '') return fallback;
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  const sign = num > 0 && includePlus ? '+' : '';
  return `${sign}${num.toFixed(decimals)}%`;
}

export function formatMultiplier(value, { fallback = '—', decimals = 3 } = {}) {
  if (value == null || value === '') return fallback;
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return `${num.toFixed(decimals)}×`;
}

export function formatNumber(value, { fallback = '—', decimals = 4 } = {}) {
  if (value == null || value === '') return fallback;
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return num.toFixed(decimals);
}
