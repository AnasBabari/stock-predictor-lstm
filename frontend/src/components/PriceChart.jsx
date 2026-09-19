import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { usePriceHistory } from '../hooks/usePriceHistory';
import {
  CHART_RANGES,
  availableRanges,
  defaultRangeId,
  periodChange,
  sliceRangePoints,
} from '../utils/priceRanges';
import { getSharedEstimatePresentation } from '../utils/estimateAdapter';
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
  if (value == null || value === '' || !Number.isFinite(Number(value))) return '—';
  const isPence = currencySymbol === 'p' || currencySymbol === 'GBp';
  const symbol = isPence ? 'p' : (currencySymbol || '$');
  const decimals = isPence ? 1 : 2;
  const numeric = Number(value);
  const magnitude = Math.abs(numeric).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  if (isPence) {
    return `${numeric < 0 ? '-' : ''}${magnitude}p`;
  }
  return `${numeric < 0 ? '-' : ''}${symbol}${magnitude}`;
}

function formatAxisLabel(label, isIntraday) {
  if (!isIntraday) {
    // If it's a date string like '2026-09-09', show '09-09' or full label
    if (typeof label === 'string' && label.length >= 10 && label.includes('-')) {
      return label.slice(5);
    }
    return String(label);
  }
  const parsed = Date.parse(label);
  if (!Number.isFinite(parsed)) return String(label);
  const date = new Date(parsed);
  return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
}

function padForecast(values, leadNulls) {
  return [...Array(Math.max(0, leadNulls)).fill(null), ...(values || [])];
}

// Vertical crosshair through the hovered point
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

// Dashed last-price line with a value tag on the right edge
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

// Shaded estimate region past the last actual
const forecastRegionPlugin = {
  id: 't212ForecastRegion',
  beforeDraw: (chart) => {
    const { ctx, chartArea, scales } = chart;
    const splitIdx = chart.options.plugins?.t212ForecastRegion?.splitIndex;
    if (splitIdx == null || !scales.x || !chartArea) return;
    const xPos = scales.x.getPixelForValue(splitIdx);
    if (xPos == null || xPos < chartArea.left || xPos > chartArea.right) return;
    ctx.save();
    ctx.fillStyle = chart.config.options?.isDark ? 'rgba(56,189,248,0.06)' : 'rgba(2,132,199,0.06)';
    ctx.fillRect(xPos, chartArea.top, chartArea.right - xPos, chartArea.bottom - chartArea.top);
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = chart.config.options?.isDark ? 'rgba(56,189,248,0.35)' : 'rgba(2,132,199,0.35)';
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

export default function PriceChart({
  ticker,
  currencySymbol = '$',
  forecast = null,
  companyName = null,
  onHistorySettled = null,
  onRetryForecast = null,
}) {
  const isDark = useAppTheme();
  const { history, loading, error, meta, retry } = usePriceHistory(ticker);
  const [rangeId, setRangeId] = useState(null);
  const [view, setView] = useState(null);
  const [showForecast, setShowForecast] = useState(true);
  const [isExpanded, setIsExpanded] = useState(false);
  const [showEstimateInfo, setShowEstimateInfo] = useState(false);

  const chartRef = useRef(null);
  const wrapRef = useRef(null);
  const dragRef = useRef(null);
  const reportedRef = useRef(null);
  const expandBtnRef = useRef(null);

  const forecastForTicker = forecast && forecast.ticker === String(ticker || '').toUpperCase() ? forecast : null;

  // Degraded mode: fallback history embedded in forecast payload if primary endpoint fails
  const fallbackHistory = useMemo(() => {
    if (forecastForTicker?.historical_provenance === 'synthetic') return null;
    const dates = forecastForTicker?.historical_dates;
    const prices = forecastForTicker?.historical_prices;
    if (!Array.isArray(dates) || !Array.isArray(prices)) return null;
    const count = Math.min(dates.length, prices.length);
    const daily = [];
    for (let i = 0; i < count; i += 1) {
      if (prices[i] == null || prices[i] === '') continue;
      const close = Number(prices[i]);
      if (dates[i] && Number.isFinite(close) && close > 0) daily.push({ d: String(dates[i]), c: close });
    }
    if (daily.length < 2) return null;
    return {
      ticker: forecastForTicker.ticker,
      asOf: forecastForTicker.data_as_of,
      daily,
      intraday: null,
      degraded: true,
    };
  }, [forecastForTicker]);

  const effectiveHistory = history || fallbackHistory;
  const showLoading = loading && !fallbackHistory;
  const showError = !loading && error && !fallbackHistory;

  // Single presentation adapter verification
  const estimate = useMemo(() => {
    if (!forecastForTicker) return null;
    return getSharedEstimatePresentation({
      forecast: forecastForTicker,
      history: effectiveHistory,
      currencySymbol,
    });
  }, [forecastForTicker, effectiveHistory, currencySymbol]);

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
  const isIntradayRange = activeRange === '24H' || points.isIntraday;

  // Overlay forecast only when enabled, not intraday, and estimate is available without mismatch
  const canShowForecastOverlay = showForecast && !isIntradayRange && estimate?.isAvailable;

  const totalLabels = points.labels.length + (canShowForecastOverlay ? estimate.futureDates.length : 0);
  const fullView = totalLabels > 0 ? { start: 0, end: totalLabels - 1 } : null;
  const effectiveView = view || fullView;
  const isZoomed = Boolean(view && fullView && (view.start > 0 || view.end < fullView.end));

  const stats = useMemo(() => {
    const change = periodChange(points.prices);
    return { ...change, last: points.prices.filter(Number.isFinite).at(-1) };
  }, [points]);

  const hasData = points.labels.length > 0 && Number.isFinite(stats.last);

  const colors = useMemo(() => {
    const up = stats.up;
    if (isDark) {
      return {
        line: up ? '#00f5a0' : '#ff5c5c',
        areaTop: up ? 'rgba(0,245,160,0.18)' : 'rgba(255,92,92,0.18)',
        grid: 'rgba(255,255,255,0.05)',
        tick: '#8b93a7',
        tooltipBg: '#0d0d1a',
        tooltipTitle: '#e8e8f0',
        tooltipBody: '#a0a0c0',
        tooltipBorder: 'rgba(255,255,255,0.08)',
        estimate: '#38bdf8',
      };
    }
    return {
      line: up ? '#10b981' : '#ef4444',
      areaTop: up ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
      grid: 'rgba(0,0,0,0.06)',
      tick: '#64748b',
      tooltipBg: '#ffffff',
      tooltipTitle: '#1e293b',
      tooltipBody: '#475569',
      tooltipBorder: 'rgba(0,0,0,0.08)',
      estimate: '#0284c7',
    };
  }, [isDark, stats.up]);

  const chartData = useMemo(() => {
    const labels = [...points.labels.map((label) => formatAxisLabel(label, points.isIntraday))];
    const historyPadded = [...points.prices];
    const datasets = [];

    // Main historical price dataset
    datasets.push({
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
    });

    let forecastSplitIndex = null;

    if (canShowForecastOverlay && estimate?.series?.length > 0) {
      estimate.futureDates.forEach((futureDate) => {
        historyPadded.push(null);
        labels.push(formatAxisLabel(futureDate, false));
      });

      forecastSplitIndex = points.prices.length - 1;

      datasets.push({
        label: '7-day estimate',
        data: padForecast(estimate.series, points.prices.length),
        borderColor: colors.estimate,
        backgroundColor: 'transparent',
        borderWidth: 2.2,
        borderDash: [6, 4],
        pointRadius: 0,
        pointHoverRadius: 6,
        pointHoverBackgroundColor: colors.estimate,
        tension: 0.25,
        spanGaps: false,
      });
    }

    return {
      labels,
      forecastSplitIndex,
      datasets,
    };
  }, [points, canShowForecastOverlay, estimate, colors]);

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
      if (event.pointerType === 'touch') return;
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

  // Modal Escape listener & focus restoration
  useEffect(() => {
    if (!isExpanded) return undefined;
    const dialog = wrapRef.current;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    // Make every sibling outside the dialog inert, preserving prior values.
    const blocked = [];
    for (let node = dialog; node?.parentElement; node = node.parentElement) {
      for (const sibling of node.parentElement.children) {
        if (sibling !== node) {
          blocked.push([sibling, sibling.inert]);
          sibling.inert = true;
        }
      }
      if (node.parentElement === document.body) break;
    }
    const focusable = () => [...dialog.querySelectorAll('button:not(:disabled), [href], [tabindex="0"]')];
    focusable()[0]?.focus();
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        setIsExpanded(false);
      }
      if (e.key === 'Tab') {
        const elements = focusable();
        const first = elements[0];
        const last = elements.at(-1);
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault(); last?.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault(); first?.focus();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      blocked.forEach(([element, inert]) => { element.inert = inert; });
      document.body.style.overflow = previousOverflow;
      expandBtnRef.current?.focus();
    };
  }, [isExpanded]);

  const toggleExpand = useCallback(() => {
    setIsExpanded((prev) => {
      if (prev) {
        setTimeout(() => expandBtnRef.current?.focus(), 0);
      }
      return !prev;
    });
  }, []);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    isDark,
    animation: { duration: 250 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      t212ForecastRegion: { splitIndex: chartData.forecastSplitIndex },
      t212LastPrice: canShowForecastOverlay
        ? undefined
        : {
          value: stats.last,
          color: stats.change > 0 ? (isDark ? '#00f5a0' : '#10b981') : stats.change < 0 ? (isDark ? '#ff5c5c' : '#ef4444') : '#8b93a7',
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
  }), [isDark, colors, chartData, effectiveView, yBounds, stats.last, stats.change, currencySymbol, canShowForecastOverlay]);

  const selectRange = useCallback((id) => {
    setRangeId(id);
    setView(null);
  }, []);

  const changeClass = stats.change > 0 ? 'up' : stats.change < 0 ? 'down' : 'flat';
  const sign = stats.change > 0 ? '+' : '';
  const dataDate = effectiveHistory?.asOf || (Array.isArray(effectiveHistory?.daily) ? effectiveHistory.daily.at(-1)?.d : null);
  const resolvedCompanyName = companyName || forecastForTicker?.ticker_name || null;

  return (
    <section
      id="chartContainer"
      className={`t212-chart-section ${isExpanded ? 'expanded-modal-open' : ''}`}
      aria-label={`${ticker} price chart`}
    >
      {/* Chart Header */}
      <div className="t212-chart-head">
        <div className="t212-price-block">
          <div className="t212-title-row">
            {resolvedCompanyName && <span className="t212-company-name">{resolvedCompanyName}</span>}
            <span className="t212-ticker">{ticker}</span>
          </div>

          {hasData ? (
            <div className="t212-metrics-row">
              <strong className="t212-price mono">
                {formatMoneyLocal(stats.last, currencySymbol)}
              </strong>
              {points.prices.filter(Number.isFinite).length >= 2 && (
                <span className={`t212-change ${changeClass}`}>
                  {sign}{formatMoneyLocal(stats.change, currencySymbol)} ({sign}{stats.changePct.toFixed(2)}%)
                </span>
              )}
              <span className="t212-range-name">
                {activeRange === 'MAX' ? 'All time' : activeRange}
              </span>
              {dataDate && !points.isIntraday && (
                <span className="t212-data-date" title={`Market data through ${dataDate}`}>
                  Data through {dataDate}
                </span>
              )}
            </div>
          ) : (
            <span className="t212-price-skeleton" aria-hidden="true" />
          )}
        </div>

        {/* Restrained Toolbar */}
        <div className="t212-restrained-toolbar">
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
          </div>

          <div className="t212-toolbar-actions">
            {/* 7-day estimate toggle */}
            <div className="estimate-toggle-wrapper">
              <button
                type="button"
                className={`toolbar-btn estimate-toggle-btn ${showForecast && !isIntradayRange ? 'active' : ''}`}
                disabled={isIntradayRange}
                onClick={() => setShowForecast((prev) => !prev)}
                title={isIntradayRange ? '7-day estimate is hidden on 24H intraday range' : 'Toggle 7-day estimate line'}
                aria-pressed={showForecast && !isIntradayRange}
              >
                <span className="estimate-indicator-dash" aria-hidden="true" />
                7-day estimate
              </button>
              {isIntradayRange && (
                <span className="intraday-estimate-note">Hidden for 24H</span>
              )}
            </div>

            {/* Reset control */}
            {isZoomed && (
              <button
                type="button"
                className="toolbar-btn reset-btn"
                onClick={() => setView(null)}
                aria-label="Reset"
              >
                Reset
              </button>
            )}

            {/* Expand chart control */}
            <button
              ref={expandBtnRef}
              type="button"
              className="toolbar-btn expand-btn"
              onClick={toggleExpand}
              aria-label={isExpanded ? 'Close expanded chart' : 'Expand chart'}
              title={isExpanded ? 'Close (Esc)' : 'Expand chart'}
            >
              {isExpanded ? 'Close' : 'Expand'}
            </button>
          </div>
        </div>
      </div>

      {/* Chart Canvas Wrap */}
      <div
        className={`t212-chart-wrap ${isExpanded ? 'expanded' : ''}`}
        ref={wrapRef}
        role={isExpanded ? 'dialog' : undefined}
        aria-modal={isExpanded ? 'true' : undefined}
        aria-label={isExpanded ? 'Expanded price chart' : undefined}
      >
        {isExpanded && (
          <div className="expanded-modal-bar">
            <span className="expanded-title" id="expanded-chart-title">{ticker} · Interactive chart</span>
            <button
              type="button"
              className="expanded-close-btn"
              onClick={toggleExpand}
              aria-label="Close expanded chart"
            >
              ✕ Close (Esc)
            </button>
          </div>
        )}

        {showLoading && (
          <div className="t212-chart-skeleton" role="status" aria-live="polite">
            <span className="t212-skeleton-bars" aria-hidden="true">
              {Array.from({ length: 24 }, (_, index) => (
                <i key={index} style={{ height: `${28 + ((index * 37) % 62)}%` }} />
              ))}
            </span>
            <span className="t212-skeleton-label">Loading price history…</span>
          </div>
        )}

        {showError && (
          <div className="t212-chart-error" role="alert">
            <p>{error}</p>
            <button type="button" className="retry-button" onClick={retry}>Retry chart</button>
          </div>
        )}

        <div className="t212-canvas-viewport">
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
      </div>

      {/* In-Chart Forecast Summary Line or Mismatch Alert */}
      {estimate?.isAvailable && (
        <div className="chart-estimate-bar">
          <div className="estimate-summary-text">
            <span className="estimate-dot" aria-hidden="true" />
            <span>
              <strong>Day 7 estimate:</strong> {formatMoneyLocal(estimate.finalPrice, currencySymbol)} ·{' '}
              <span className={`estimate-delta ${estimate.direction}`}>
                {estimate.changePct != null && Number.isFinite(estimate.changePct)
                  ? `${estimate.changePct > 0 ? '+' : ''}${estimate.changePct.toFixed(1)}%`
                  : '—'}
              </span>{' '}
              from the latest close.
            </span>
            <button
              type="button"
              className="info-icon-btn"
              onClick={() => setShowEstimateInfo((prev) => !prev)}
              aria-label="How this estimate works"
              title="How this estimate works"
            >
              ℹ
            </button>
          </div>

          {showEstimateInfo && (
            <p className="estimate-inline-explanation">
              Model estimate re-fitted on daily closing prices. Past performance does not guarantee future results.
            </p>
          )}
        </div>
      )}

      {estimate?.isMismatch && (
        <div className="chart-mismatch-alert" role="alert">
          <span className="mismatch-icon" aria-hidden="true">⚠️</span>
          <span className="mismatch-msg">
            Forecast data does not match the latest market history ({estimate.mismatchReason}).
          </span>
          {onRetryForecast && (
            <button type="button" className="retry-action-btn" onClick={onRetryForecast}>
              Retry forecast
            </button>
          )}
        </div>
      )}
      {forecastForTicker && estimate && !estimate.isAvailable && !estimate.isMismatch && (
        <p className="chart-mismatch-alert" role="status">
          Seven-day estimate unavailable: forecast data is incomplete or invalid.
          {onRetryForecast && <button type="button" className="retry-action-btn" onClick={onRetryForecast}>Retry forecast</button>}
        </p>
      )}
    </section>
  );
}
