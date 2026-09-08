/**
 * Pure range-tab logic for the Trading212-style price chart.
 *
 * Tabs are availability-driven: a range only appears when the loaded
 * history actually covers it (e.g. a 2024 IPO shows 1Y but never 5Y;
 * 24H appears only when intraday bars loaded). MAX always appears when
 * any daily history exists.
 */

export const CHART_RANGES = [
  { id: '24H', label: '24H', kind: 'intraday' },
  { id: '5D', label: '5D', kind: 'daily', sessions: 5 },
  { id: '1M', label: '1M', kind: 'daily', sessions: 22 },
  { id: '6M', label: '6M', kind: 'daily', sessions: 126 },
  { id: '1Y', label: '1Y', kind: 'daily', sessions: 252 },
  { id: '5Y', label: '5Y', kind: 'daily', sessions: 1260 },
  { id: 'MAX', label: 'MAX', kind: 'daily', sessions: Number.POSITIVE_INFINITY },
];

const DEFAULT_PREFERENCE = ['5D', '24H', '1M', '6M', '1Y', '5Y', 'MAX'];

export function availableRanges({ dailyCount = 0, hasIntraday = false } = {}) {
  if (!(dailyCount >= 1)) return [];
  return CHART_RANGES.filter((range) => {
    if (range.kind === 'intraday') return hasIntraday;
    if (range.sessions === Number.POSITIVE_INFINITY) return true;
    return dailyCount >= range.sessions;
  }).map((range) => range.id);
}

export function defaultRangeId(availableIds = []) {
  const set = new Set(availableIds);
  return DEFAULT_PREFERENCE.find((id) => set.has(id)) || null;
}

/**
 * Slice loaded history into render points for a range tab.
 * Returns { labels, prices, isIntraday }. Labels are date strings for
 * daily ranges and ISO timestamps for 24H.
 */
export function sliceRangePoints(history, rangeId) {
  const daily = Array.isArray(history?.daily) ? history.daily : [];
  if (rangeId === '24H') {
    const bars = Array.isArray(history?.intraday) ? history.intraday : [];
    return {
      labels: bars.map((bar) => bar.t),
      prices: bars.map((bar) => Number(bar.c)),
      isIntraday: true,
    };
  }
  const def = CHART_RANGES.find((range) => range.id === rangeId && range.kind === 'daily');
  const count = def && Number.isFinite(def.sessions) ? def.sessions : daily.length;
  const slice = daily.slice(Math.max(0, daily.length - count));
  return {
    labels: slice.map((bar) => bar.d),
    prices: slice.map((bar) => Number(bar.c)),
    isIntraday: false,
  };
}

export function periodChange(prices = []) {
  const finite = prices.filter((value) => Number.isFinite(value));
  if (finite.length < 2) return { change: 0, changePct: 0, up: true };
  const first = finite[0];
  const last = finite[finite.length - 1];
  if (!first) return { change: 0, changePct: 0, up: last >= 0 };
  const change = last - first;
  return { change, changePct: (change / Math.abs(first)) * 100, up: change >= 0 };
}
