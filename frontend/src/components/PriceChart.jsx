import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { usePriceHistory } from '../hooks/usePriceHistory';
import { useVolatilityOutlook } from '../hooks/useVolatilityOutlook';
import {
  CHART_RANGES,
  availableRanges,
  defaultRangeId,
  periodChange,
  sliceRangePoints,
} from '../utils/priceRanges';
import LazyLineChart from './LazyLineChart';

const MIN_ZOOM_POINTS = 12;

function themeIsDark() {
  if (typeof document === 'undefined') return true;
  return (document.documentElement.getAttribute('data-theme') || 'dark') !== 'light';
}

function useAppTheme() {
  const [isDark, setIsDark] = useState(themeIsDark);
  useEffect(() => {
    if (typeof document === 'undefined' || typeof MutationObserver === 'undefined') return undefined;
    const observer = new MutationObserver(() => setIsDark(themeIsDark()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);
  return isDark;
}

function formatMoneyLocal(value, currencySymbol) {
  if (!Number.isFinite(value)) return '—';
  const symbol = currencySymbol === 'p' || currencySymbol === 'GBp' ? 'p' : (currencySymbol || '$');
  const decimals = symbol === 'p' ? 1 : 2;
  return `${symbol}${Number(value).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
}

function formatAxisLabel(label, isIntraday) {
  if (!isIntraday) return String(label).slice(2);
  const parsed = Date.parse(label);
  if (!Number.isFinite(parsed)) return String(label);
  const date = new Date(parsed);
  return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
}

function padForecast(values, leadNulls) {
  // Deliberately no anchor point: the estimate must NOT join the historical
  // close seamlessly. The visible break plus the dashed style and shaded
  // future region mark this as a model estimate, not a future price.
  return [...Array(Math.max(0, leadNulls)).fill(null), ...(values || [])];
}

// Vertical crosshair through the hovered point, Trading212 style.
const crosshairPlugin = {
  id: 't212Crosshair',
  afterDraw: (chart) => {
    const active = chart.tooltip?.getActiveElements?.();
    if (!active || active.length === 0) return;
    const { ctx, chartArea } = chart;
    const x = active[0].element.x;
    if (x < chartArea.left || x > chartArea.right) return;
    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = chart.config.options?.isDark ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.25)';
    ctx.lineWidth = 1;
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    ctx.restore();
  },
};

// Dashed last-price line with a value tag on the right edge.
const lastPricePlugin = {
  id: 't212LastPrice',
  afterDraw: (chart) => {
    const price = chart.config.options?.plugins?.t212LastPrice?.value;
    if (!Number.isFinite(price) || !chart.scales?.y || !chart.chartArea) return;
    const { ctx, chartArea, scales } = chart;
    const y = scales.y.getPixelForValue(price);
    if (y < chartArea.top || y > chartArea.bottom) return;
    const color = chart.config.options?.plugins?.t212LastPrice?.color || '#8b93a7';
    const label = chart.config.options?.plugins?.t212LastPrice?.label || '';
    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.moveTo(chartArea.left, y);
    ctx.lineTo(chartArea.right, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = '600 10px Inter, sans-serif';
    const width = ctx.measureText(label).width + 12;
    const tagY = Math.min(Math.max(y - 9, chartArea.top), chartArea.bottom - 18);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(chartArea.right - width, tagY, width, 18, 4);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, chartArea.right - width / 2, tagY + 9);
    ctx.restore();
  },
};

function forecastDatasets(forecast, historyLength, colors) {
  const sets = [];
  if (!forecast || !Array.isArray(forecast.future_dates) || forecast.future_dates.length === 0) {
    return { sets, futureLabels: [] };
  }
  const futureLabels = forecast.future_dates;
  // Estimates align to future labels only: index historyLength..end. The
  // last history index stays null in these datasets, leaving a visible
  // break between actuals and estimates.
  const lead = Math.max(0, historyLength);
  if (Array.isArray(forecast.predicted_prices) && forecast.predicted_prices.length > 0) {
    sets.push({
      label: 'Average 7-day estimate',
      data: padForecast(forecast.predicted_prices, lead),
      borderColor: colors.estimate,
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      borderDash: [6, 4],
      pointRadius: 3,
      pointBackgroundColor: colors.estimate,
      pointBorderWidth: 0,
      pointHoverRadius: 6,
      tension: 0.3,
      spanGaps: false,
    });
  }
  const band = forecast.historical_error_band;
  if (band && Array.isArray(band.upper_prices) && Array.isArray(band.lower_prices)) {
    sets.push({
      label: 'Estimate range (upper)',
      data: padForecast(band.upper_prices, lead),
      borderColor: 'transparent',
      backgroundColor: colors.bandFill,
      pointRadius: 0,
      fill: '+1',
      tension: 0.3,
      spanGaps: false,
    });
    sets.push({
      label: 'Estimate range (lower)',
      data: padForecast(band.lower_prices, lead),
      borderColor: 'transparent',
      backgroundColor: 'transparent',
      pointRadius: 0,
      fill: false,
      tension: 0.3,
      spanGaps: false,
    });
  }
  return { sets, futureLabels };
}

// Shaded estimate region past the last actual: the future is a scenario,
// not a continuation of the price line.
const forecastRegionPlugin = {
  id: 't212ForecastRegion',
  beforeDraw: (chart) => {
    const { ctx, chartArea, scales } = chart;
    const splitIdx = chart.data?.forecastSplitIndex;
    if (splitIdx == null || !scales.x || !chartArea) return;
    const xPos = scales.x.getPixelForValue(splitIdx);
    if (xPos == null || xPos < chartArea.left || xPos > chartArea.right) return;
    ctx.save();
    ctx.fillStyle = chart.config.options?.isDark ? 'rgba(56,189,248,0.05)' : 'rgba(2,132,199,0.05)';
    ctx.fillRect(xPos, chartArea.top, chartArea.right - xPos, chartArea.bottom - chartArea.top);
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = chart.config.options?.isDark ? 'rgba(255,255,255,0.25)' : 'rgba(0,0,0,0.25)';
    ctx.lineWidth = 1.5;
    ctx.moveTo(xPos, chartArea.top);
    ctx.lineTo(xPos, chartArea.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = chart.config.options?.isDark ? '#38bdf8' : '#0284c7';
    ctx.font = '600 10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('ESTIMATE', xPos + 6, chartArea.top + 14);
    ctx.restore();
  },
};

export default function PriceChart({ ticker, currencySymbol = '$', forecast = null, onHistorySettled = null }) {
  const isDark = useAppTheme();
  const { history, loading, error, meta, retry } = usePriceHistory(ticker);
  // G3 expected-range band shares the outlook module cache with the
  // VolatilityOutlook card, so no second network fetch happens.
  const { outlook: g3outlook } = useVolatilityOutlook(ticker);
  const [rangeId, setRangeId] = useState(null);
  const [view, setView] = useState(null);
  const chartRef = useRef(null);
  const wrapRef = useRef(null);
  const dragRef = useRef(null);
  const reportedRef = useRef(null);

  const forecastForTicker = forecast && forecast.ticker === String(ticker || '').toUpperCase() ? forecast : null;

  // Degraded mode: when the history endpoint is unreachable, fall back to
  // the short history embedded in the forecast payload (exactly what the
  // previous chart rendered). Synthetic placeholders are refused: a chart
  // must never draw manufactured prices. No intraday legs, so 24H hides.
  const fallbackHistory = useMemo(() => {
    if (forecastForTicker?.historical_provenance === 'synthetic') return null;
    const dates = forecastForTicker?.historical_dates;
    const prices = forecastForTicker?.historical_prices;
    if (!Array.isArray(dates) || !Array.isArray(prices)) return null;
    const count = Math.min(dates.length, prices.length);
    const daily = [];
    for (let i = 0; i < count; i += 1) {
      const close = Number(prices[i]);
      if (dates[i] && Number.isFinite(close)) daily.push({ d: String(dates[i]), c: close });
    }
    if (daily.length < 2) return null;
    return {
      ticker: forecastForTicker.ticker,
      daily,
      intraday: null,
      degraded: true,
    };
  }, [forecastForTicker]);

  const effectiveHistory = history || fallbackHistory;
  const showLoading = loading && !fallbackHistory;
  const showError = !loading && error && !fallbackHistory;

  // One settlement report per ticker: chart paint (endpoint or degraded) or
  // terminal failure. Powers the user-visible timing footnote in App.
  useEffect(() => {
    if (!onHistorySettled) return;
    const key = String(ticker || '').toUpperCase();
    if (effectiveHistory && reportedRef.current !== `${key}:ok`) {
      reportedRef.current = `${key}:ok`;
      onHistorySettled({
        ok: true,
        degraded: !history && Boolean(fallbackHistory),
        fetchMs: meta?.fetchMs ?? null,
        fromCache: Boolean(meta?.fromCache),
        marketDataCache: history?.marketDataCache ?? null,
      });
    } else if (showError && reportedRef.current !== `${key}:ok` && reportedRef.current !== `${key}:error`) {
      reportedRef.current = `${key}:error`;
      onHistorySettled({ ok: false, degraded: false, fetchMs: null, fromCache: false, marketDataCache: null });
    }
  }, [effectiveHistory, showError, history, fallbackHistory, meta, ticker, onHistorySettled]);

  const hasIntraday = Array.isArray(effectiveHistory?.intraday) && effectiveHistory.intraday.length >= 5;
  const dailyCount = Array.isArray(effectiveHistory?.daily) ? effectiveHistory.daily.length : 0;
  const available = useMemo(
    () => (effectiveHistory ? availableRanges({ dailyCount, hasIntraday }) : []),
    [effectiveHistory, dailyCount, hasIntraday],
  );

  useEffect(() => {
    setRangeId((current) => {
      const avail = availableRanges({ dailyCount, hasIntraday });
      const next = effectiveHistory ? defaultRangeId(avail) : null;
      return current && avail.includes(current) ? current : next;
    });
    setView(null);
  }, [effectiveHistory, dailyCount, hasIntraday, ticker]);

  const points = useMemo(
    () => (effectiveHistory && rangeId
      ? sliceRangePoints(effectiveHistory, rangeId)
      : { labels: [], prices: [], isIntraday: false }),
    [effectiveHistory, rangeId],
  );

  const activeRange = rangeId || defaultRangeId(available);

  const totalLabels = points.labels.length + (forecastForTicker?.future_dates?.length || 0);
  const fullView = totalLabels > 0 ? { start: 0, end: totalLabels - 1 } : null;
  const effectiveView = view || fullView;
  const isZoomed = Boolean(view && fullView && (view.start > 0 || view.end < fullView.end));

  const stats = useMemo(() => {
    const change = periodChange(points.prices);
    return { ...change, last: points.prices.filter(Number.isFinite).at(-1) };
  }, [points]);

  const colors = useMemo(() => {
    const up = stats.up;
    if (isDark) {
      return {
        line: up ? '#00f5a0' : '#ff5c5c',
        areaTop: up ? 'rgba(0,245,160,0.22)' : 'rgba(255,92,92,0.22)',
        grid: 'rgba(255,255,255,0.05)',
        tick: '#8b93a7',
        tooltipBg: '#0d0d1a',
        tooltipTitle: '#e8e8f0',
        tooltipBody: '#a0a0c0',
        tooltipBorder: 'rgba(255,255,255,0.08)',
        estimate: '#38bdf8',
        bandFill: 'rgba(56,189,248,0.10)',
      };
    }
    return {
      line: up ? '#10b981' : '#ef4444',
      areaTop: up ? 'rgba(16,185,129,0.18)' : 'rgba(239,68,68,0.18)',
      grid: 'rgba(0,0,0,0.06)',
      tick: '#64748b',
      tooltipBg: '#ffffff',
      tooltipTitle: '#1e293b',
      tooltipBody: '#475569',
      tooltipBorder: 'rgba(0,0,0,0.08)',
      estimate: '#0284c7',
      bandFill: 'rgba(2,132,199,0.10)',
    };
  }, [isDark, stats.up]);

  const chartData = useMemo(() => {
    const labels = [...points.labels.map((label) => formatAxisLabel(label, points.isIntraday))];
    const historyPadded = [...points.prices];
    const { sets: forecastSets, futureLabels } = forecastForTicker
      ? forecastDatasets(forecastForTicker, points.prices.length, colors)
      : { sets: [], futureLabels: [] };
    futureLabels.forEach((_, index) => {
      historyPadded.push(null);
      labels.push(`+${index + 1}d`);
    });
    // Certified-volatility expected range (G3, 5-session): aligned strictly
    // by date against the estimate path; skipped silently on any mismatch.
    const bandSets = [];
    const five = g3outlook?.byHorizon?.[5];
    const bandUpper = five?.volatility_cone?.p95;
    const bandLower = five?.volatility_cone?.p05;
    const bandDates = five?.future_dates;
    const estimateDates = forecastForTicker?.future_dates;
    if (
      Array.isArray(bandDates) && Array.isArray(bandUpper) && Array.isArray(bandLower)
      && Array.isArray(estimateDates) && bandDates.length > 0
      && bandUpper.length === bandDates.length && bandLower.length === bandDates.length
      && estimateDates.slice(0, bandDates.length).join('|') === bandDates.join('|')
    ) {
      const head = Array(points.prices.length).fill(null);
      const tail = Array(Math.max(0, labels.length - points.prices.length - bandDates.length)).fill(null);
      bandSets.push(
        {
          label: 'Expected volatility range (upper)',
          data: [...head, ...bandUpper, ...tail],
          borderColor: 'transparent',
          backgroundColor: colors.bandFill,
          pointRadius: 0,
          fill: '+1',
          tension: 0.3,
          spanGaps: false,
        },
        {
          label: 'Expected volatility range (lower)',
          data: [...head, ...bandLower, ...tail],
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          pointRadius: 0,
          fill: false,
          tension: 0.3,
          spanGaps: false,
        },
      );
    }
    return {
      labels,
      // Index of the last actual; everything right of it is estimate.
      forecastSplitIndex: forecastSets.length > 0 ? points.prices.length - 1 : null,
      datasets: [
        {
          label: 'Price',
          data: historyPadded,
          borderColor: colors.line,
          backgroundColor: (context) => {
            const { ctx, chartArea } = context.chart;
            if (!chartArea) return colors.areaTop;
            const grad = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
            grad.addColorStop(0, colors.areaTop);
            grad.addColorStop(1, 'transparent');
            return grad;
          },
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: colors.line,
          tension: 0.25,
          fill: true,
          spanGaps: false,
        },
        ...forecastSets,
        ...bandSets,
      ],
    };
  }, [points, forecastForTicker, colors, g3outlook]);

  const yBounds = useMemo(() => {
    if (!effectiveView || chartData.labels.length === 0) return {};
    const values = [];
    for (const ds of chartData.datasets) {
      (ds.data || []).slice(effectiveView.start, effectiveView.end + 1).forEach((value) => {
        if (Number.isFinite(value)) values.push(value);
      });
    }
    if (values.length === 0) return {};
    const min = Math.min(...values);
    const max = Math.max(...values);
    const padding = Math.max((max - min) * 0.12, max * 0.015);
    return { suggestedMin: Math.max(0, min - padding), suggestedMax: max + padding };
  }, [chartData, effectiveView]);

  const clampView = useCallback((start, end) => {
    if (!fullView) return null;
    const size = fullView.end - fullView.start + 1;
    let next = { start: Math.round(start), end: Math.round(end) };
    if (next.end - next.start + 1 < Math.min(MIN_ZOOM_POINTS, size)) {
      const mid = (next.start + next.end) / 2;
      const half = Math.floor(Math.min(MIN_ZOOM_POINTS, size) / 2);
      next = { start: Math.round(mid) - half, end: Math.round(mid) + half };
    }
    if (next.start < fullView.start) {
      next = { start: fullView.start, end: Math.min(fullView.end, fullView.start + (next.end - next.start)) };
    }
    if (next.end > fullView.end) {
      next = { end: fullView.end, start: Math.max(fullView.start, fullView.end - (next.end - next.start)) };
    }
    if (next.start <= fullView.start && next.end >= fullView.end) return null;
    return next;
  }, [fullView]);

  const zoomAt = useCallback((pixelX, factor) => {
    const chart = chartRef.current;
    if (!chart || !effectiveView || factor <= 0) return;
    const area = chart.chartArea;
    const clampedX = Math.min(Math.max(pixelX, area.left), area.right);
    const cursor = chart.scales.x.getValueForPixel(clampedX);
    const size = effectiveView.end - effectiveView.start + 1;
    const nextSize = Math.max(Math.min(MIN_ZOOM_POINTS, fullView.end - fullView.start + 1), Math.round(size * factor));
    const ratio = size <= 1 ? 0.5 : (cursor - effectiveView.start) / (size - 1);
    const nextStart = cursor - ratio * (nextSize - 1);
    setView(clampView(nextStart, nextStart + nextSize - 1));
  }, [chartRef, effectiveView, fullView, clampView]);

  useEffect(() => {
    const node = wrapRef.current;
    if (!node) return undefined;
    const onWheel = (event) => {
      if (!chartRef.current) return;
      event.preventDefault();
      const rect = node.getBoundingClientRect();
      zoomAt(event.clientX - rect.left, event.deltaY > 0 ? 1.18 : 0.85);
    };
    const onPointerDown = (event) => {
      if (event.button !== 0 || !chartRef.current) return;
      const rect = node.getBoundingClientRect();
      dragRef.current = { x: event.clientX - rect.left, view: effectiveView };
      node.setPointerCapture?.(event.pointerId);
    };
    const onPointerMove = (event) => {
      const drag = dragRef.current;
      const chart = chartRef.current;
      if (!drag || !chart || !drag.view) return;
      const rect = node.getBoundingClientRect();
      const dx = event.clientX - rect.left - drag.x;
      const areaWidth = chart.chartArea.width || 1;
      const size = drag.view.end - drag.view.start + 1;
      const shift = Math.round((dx / areaWidth) * size);
      if (shift !== 0) setView(clampView(drag.view.start - shift, drag.view.end - shift));
    };
    const onPointerUp = () => { dragRef.current = null; };
    const onDoubleClick = () => setView(null);
    node.addEventListener('wheel', onWheel, { passive: false });
    node.addEventListener('pointerdown', onPointerDown);
    node.addEventListener('pointermove', onPointerMove);
    node.addEventListener('pointerup', onPointerUp);
    node.addEventListener('dblclick', onDoubleClick);
    return () => {
      node.removeEventListener('wheel', onWheel);
      node.removeEventListener('pointerdown', onPointerDown);
      node.removeEventListener('pointermove', onPointerMove);
      node.removeEventListener('pointerup', onPointerUp);
      node.removeEventListener('dblclick', onDoubleClick);
    };
  }, [zoomAt, effectiveView, clampView]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    isDark,
    animation: { duration: 250 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      t212LastPrice: forecastForTicker
        ? undefined
        : {
          value: stats.last,
          color: stats.up ? (isDark ? '#00f5a0' : '#10b981') : (isDark ? '#ff5c5c' : '#ef4444'),
          label: formatMoneyLocal(stats.last, currencySymbol),
        },
      tooltip: {
        backgroundColor: colors.tooltipBg,
        titleColor: colors.tooltipTitle,
        bodyColor: colors.tooltipBody,
        borderColor: colors.tooltipBorder,
        borderWidth: 1,
        cornerRadius: 10,
        padding: 12,
        displayColors: false,
        titleFont: { family: 'Inter, sans-serif', size: 12, weight: '600' },
        bodyFont: { family: 'JetBrains Mono, monospace', size: 12 },
        callbacks: {
          title: (items) => (items.length > 0 ? String(chartData.labels[items[0].dataIndex] ?? '') : ''),
          label: (ctx) => {
            const val = Number(ctx.parsed.y);
            if (!Number.isFinite(val)) return null;
            const first = chartData.datasets[0].data.filter(Number.isFinite)[0];
            const rel = Number.isFinite(first) && first ? ((val / first - 1) * 100) : 0;
            return ` ${ctx.dataset.label}: ${formatMoneyLocal(val, currencySymbol)} (${rel >= 0 ? '+' : ''}${rel.toFixed(2)}%)`;
          },
        },
      },
    },
    scales: {
      x: {
        min: effectiveView?.start,
        max: effectiveView?.end,
        ticks: { color: colors.tick, maxTicksLimit: 8, maxRotation: 0, font: { family: 'JetBrains Mono, monospace', size: 10 } },
        grid: { display: false },
      },
      y: {
        ...yBounds,
        position: 'right',
        ticks: {
          color: colors.tick,
          font: { family: 'JetBrains Mono, monospace', size: 10 },
          callback: (v) => formatMoneyLocal(Number(v), currencySymbol),
        },
        grid: { color: colors.grid },
      },
    },
  }), [isDark, colors, chartData, effectiveView, yBounds, stats.last, stats.up, currencySymbol, forecastForTicker]);

  const selectRange = useCallback((id) => {
    setRangeId(id);
    setView(null);
  }, []);

  const changeClass = stats.up ? 'up' : 'down';
  const sign = stats.change >= 0 ? '+' : '';

  return (
    <section id="chartContainer" className="t212-chart-section" aria-label={`${ticker} price chart`}>
      <div className="t212-chart-head">
        <div className="t212-price-block">
          <span className="t212-ticker">{ticker}</span>
          <strong className="t212-price mono">{formatMoneyLocal(stats.last, currencySymbol)}</strong>
          <span className={`t212-change ${changeClass}`}>
            {sign}{formatMoneyLocal(stats.change, currencySymbol)} ({sign}{stats.changePct.toFixed(2)}%)
          </span>
          <span className="t212-range-name">{activeRange === 'MAX' ? 'All time' : activeRange}</span>
        </div>
        <div className="t212-range-tabs" role="tablist" aria-label="Chart time range">
          {CHART_RANGES.filter((range) => available.includes(range.id)).map((range) => (
            <button
              key={range.id}
              type="button"
              role="tab"
              aria-selected={activeRange === range.id}
              className={`range-pill ${activeRange === range.id ? 'active' : ''}`}
              onClick={() => selectRange(range.id)}
            >
              {range.label}
            </button>
          ))}
          {isZoomed && (
            <button type="button" className="range-pill reset-pill" onClick={() => setView(null)}>
              Reset
            </button>
          )}
        </div>
      </div>
      <div
        className="t212-chart-wrap"
        ref={wrapRef}
        style={{ height: 'clamp(340px, 48vh, 520px)', position: 'relative', cursor: 'crosshair', touchAction: 'none' }}
      >
        {showLoading && <div className="loading-text" role="status">Loading price history…</div>}
        {showError && (
          <div className="t212-chart-error" role="alert">
            <p>{error}</p>
            <button type="button" className="retry-button" onClick={retry}>Retry chart</button>
          </div>
        )}
        {!showLoading && !showError && effectiveHistory && points.labels.length > 0 && (
          <React.Suspense fallback={<div className="loading-text">Loading Chart…</div>}>
            <LazyLineChart
              ref={chartRef}
              data={chartData}
              options={chartOptions}
              plugins={[crosshairPlugin, lastPricePlugin, forecastRegionPlugin]}
            />
          </React.Suspense>
        )}
        {!showLoading && !showError && effectiveHistory && points.labels.length === 0 && (
          <div className="empty-copy">Not enough price history for this view.</div>
        )}
      </div>
      <p className="t212-chart-hint">Scroll to zoom · drag to pan · double-click to reset. Intraday 24H appears when live session bars load.</p>
    </section>
  );
}

