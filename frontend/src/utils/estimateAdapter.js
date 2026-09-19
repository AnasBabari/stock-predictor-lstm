/** One validated seven-session midpoint series for the chart and summary. */
const unavailable = (reason, mismatchReason = null) => ({
  isAvailable: false, isMismatch: Boolean(mismatchReason), reason, mismatchReason,
  series: [], futureDates: [], finalPrice: null, changePct: null, direction: 'flat',
});
const positive = (value) => (typeof value === 'number' || typeof value === 'string')
  && String(value).trim() !== '' && Number.isFinite(Number(value)) && Number(value) > 0;
const validDate = (value) => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)
  && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;

export function getSharedEstimatePresentation({ forecast, history, currencySymbol = '$' } = {}) {
  if (!forecast) return unavailable('no_forecast');
  if (typeof forecast.ticker !== 'string' || !forecast.ticker.trim()
      || !validDate(forecast.data_as_of) || !positive(forecast.current_price)) {
    return unavailable('invalid_origin');
  }
  const lower = forecast.lower_prices ?? forecast.historical_error_band?.lower_prices;
  const upper = forecast.upper_prices ?? forecast.historical_error_band?.upper_prices;
  const dates = forecast.future_dates;
  if (![lower, upper, dates].every((values) => Array.isArray(values) && values.length === 7)
      || (forecast.forecast_days != null && Number(forecast.forecast_days) !== 7)) {
    return unavailable('invalid_horizon');
  }
  let previous = forecast.data_as_of;
  const series = [];
  for (let i = 0; i < 7; i += 1) {
    if (!validDate(dates[i]) || dates[i] <= previous) return unavailable('invalid_dates');
    previous = dates[i];
    if (!positive(lower[i]) || !positive(upper[i]) || Number(lower[i]) > Number(upper[i])) {
      return unavailable('invalid_bounds');
    }
    const midpoint = Number(lower[i]) / 2 + Number(upper[i]) / 2;
    if (!Number.isFinite(midpoint)) return unavailable('invalid_bounds');
    series.push(midpoint);
  }
  const lastBar = Array.isArray(history?.daily) ? history.daily.at(-1) : null;
  if (typeof history?.ticker !== 'string' || !history.ticker || !lastBar || !validDate(lastBar.d) || !positive(lastBar.c)) {
    return unavailable('history_unavailable');
  }
  if (history.ticker.toUpperCase() !== forecast.ticker.toUpperCase()) {
    return unavailable('origin_mismatch', 'Ticker mismatch');
  }
  if (lastBar.d !== forecast.data_as_of || (history.asOf && history.asOf !== lastBar.d)) {
    return unavailable('origin_mismatch', 'Origin date mismatch');
  }
  const unit = (symbol) => ['p', 'GBp', 'GBX'].includes(symbol) ? 'GBp' : symbol;
  if (history.currencySymbol && unit(history.currencySymbol) !== unit(currencySymbol)) {
    return unavailable('origin_mismatch', 'Quote unit mismatch');
  }
  const precision = unit(currencySymbol) === 'GBp' ? 1 : 2;
  if (Number(lastBar.c).toFixed(precision) !== Number(forecast.current_price).toFixed(precision)) {
    return unavailable('origin_mismatch', 'Origin price mismatch');
  }
  const finalPrice = series[6];
  const changePct = (finalPrice / Number(forecast.current_price) - 1) * 100;
  if (!Number.isFinite(changePct)) return unavailable('invalid_bounds');
  return {
    isAvailable: true, isMismatch: false, mismatchReason: null, reason: null,
    series, futureDates: [...dates], finalPrice, changePct,
    direction: Math.abs(changePct) < 0.001 ? 'flat' : changePct > 0 ? 'up' : 'down',
  };
}
