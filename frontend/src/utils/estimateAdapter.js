/**
 * Pure presentation adapter for 7-day model price estimates.
 *
 * Guarantees that:
 * 1. The chart dashed line, the final Day 7 estimate, and the percentage
 *    change are strictly computed from the exact same midpoint series.
 * 2. If lower/upper bounds are missing or invalid, the estimate is marked
 *    unavailable rather than silently displaying an alternative series.
 * 3. History compatibility is strictly verified (ticker, origin date,
 *    and close price at display precision). Any mismatch suppresses the
 *    forecast overlay and summary without synthetic rescaling.
 */

export function getSharedEstimatePresentation({
  forecast,
  history,
  currencySymbol = '$',
} = {}) {
  if (!forecast) {
    return {
      isAvailable: false,
      isMismatch: false,
      reason: 'no_forecast',
      series: [],
      futureDates: [],
      finalPrice: null,
      changePct: null,
      direction: 'flat',
    };
  }

  const lower = forecast.lower_prices || forecast.historical_error_band?.lower_prices;
  const upper = forecast.upper_prices || forecast.historical_error_band?.upper_prices;
  const futureDates = forecast.future_dates;

  if (
    !Array.isArray(lower) ||
    !Array.isArray(upper) ||
    !Array.isArray(futureDates) ||
    lower.length === 0 ||
    upper.length === 0 ||
    futureDates.length === 0
  ) {
    return {
      isAvailable: false,
      isMismatch: false,
      reason: 'missing_bounds',
      series: [],
      futureDates: [],
      finalPrice: null,
      changePct: null,
      direction: 'flat',
    };
  }

  const count = Math.min(lower.length, upper.length, futureDates.length);
  const midpointSeries = [];

  for (let i = 0; i < count; i += 1) {
    const low = Number(lower[i]);
    const high = Number(upper[i]);
    if (!Number.isFinite(low) || !Number.isFinite(high)) {
      return {
        isAvailable: false,
        isMismatch: false,
        reason: 'non_finite_bounds',
        series: [],
        futureDates: [],
        finalPrice: null,
        changePct: null,
        direction: 'flat',
      };
    }
    midpointSeries.push((low + high) / 2);
  }

  const finalPrice = midpointSeries.at(-1);
  if (!Number.isFinite(finalPrice)) {
    return {
      isAvailable: false,
      isMismatch: false,
      reason: 'invalid_final_price',
      series: [],
      futureDates: [],
      finalPrice: null,
      changePct: null,
      direction: 'flat',
    };
  }

  // If history is provided, check compatibility
  if (history) {
    const historyTicker = String(history.ticker || '').toUpperCase();
    const forecastTicker = String(forecast.ticker || '').toUpperCase();

    if (historyTicker && forecastTicker && historyTicker !== forecastTicker) {
      return {
        isAvailable: false,
        isMismatch: true,
        mismatchReason: `Ticker mismatch: history is ${historyTicker} but forecast is ${forecastTicker}`,
        series: [],
        futureDates: [],
        finalPrice: null,
        changePct: null,
        direction: 'flat',
      };
    }

    const lastBar = Array.isArray(history.daily) ? history.daily.at(-1) : null;
    const historyDate = history.asOf || lastBar?.d;
    const forecastOriginDate = forecast.data_as_of;

    if (historyDate && forecastOriginDate && historyDate !== forecastOriginDate) {
      return {
        isAvailable: false,
        isMismatch: true,
        mismatchReason: `Origin date mismatch: history latest date is ${historyDate} but forecast is as of ${forecastOriginDate}`,
        series: [],
        futureDates: [],
        finalPrice: null,
        changePct: null,
        direction: 'flat',
      };
    }

    if (lastBar && Number.isFinite(Number(lastBar.c)) && forecast.current_price != null) {
      const isPence = currencySymbol === 'p' || currencySymbol === 'GBp';
      const precision = isPence ? 1 : 2;
      const historyCloseStr = Number(lastBar.c).toFixed(precision);
      const forecastCloseStr = Number(forecast.current_price).toFixed(precision);

      if (historyCloseStr !== forecastCloseStr) {
        return {
          isAvailable: false,
          isMismatch: true,
          mismatchReason: `Origin price mismatch: history close was ${historyCloseStr} but forecast origin was ${forecastCloseStr}`,
          series: [],
          futureDates: [],
          finalPrice: null,
          changePct: null,
          direction: 'flat',
        };
      }
    }
  }

  const baselinePrice = Number(forecast.current_price);
  const hasBaseline = Number.isFinite(baselinePrice) && baselinePrice > 0;
  const changePct = hasBaseline ? ((finalPrice / baselinePrice) - 1) * 100 : null;
  const direction = changePct == null || Math.abs(changePct) < 0.001
    ? 'flat'
    : changePct > 0
      ? 'up'
      : 'down';

  return {
    isAvailable: true,
    isMismatch: false,
    mismatchReason: null,
    series: midpointSeries,
    futureDates: futureDates.slice(0, count),
    finalPrice,
    changePct,
    direction,
  };
}
