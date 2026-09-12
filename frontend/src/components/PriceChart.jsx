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
  if (!label) return '';
  if (isIntraday) {
    const parsed = Date.parse(label);
    if (!Number.isFinite(parsed)) return String(label);
    const date = new Date(parsed);
    return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
  }
  const str = String(label);
  const parts = str.split('T')[0].split('-');
  if (parts.length === 3) {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const mIdx = parseInt(parts[1], 10) - 1;
    const day = parseInt(parts[2], 10);
    if (mIdx >= 0 && mIdx < 12 && Number.isFinite(day)) {
      return `${months[mIdx]} ${String(day).padStart(2, '0')}`;
    }
  }
  return str.length > 5 ? str.slice(5) : str;
}

function drawPill(ctx, x, y, width, height, radius = 4) {
  ctx.beginPath();
  if (typeof ctx.roundRect === 'function') {
    ctx.roundRect(x, y, width, height, radius);
  } else {
    ctx.rect(x, y, width, height);
  }
}

// Dual crosshairs (vertical + horizontal) with floating axis badges (TradingView & Trading 212 style)
const crosshairPlugin = {
  id: 't212Crosshair',
  afterDraw: (chart) => {
    const active = chart.tooltip?.getActiveElements?.();
    if (!active || active.length === 0) return;
    const { ctx, chartArea, scales } = chart;
    const activeElem = active[0].element;
    const x = activeElem.x;
    const y = activeElem.y;
    if (x < chartArea.left || x > chartArea.right || y < chartArea.top || y > chartArea.bottom) return;

    const isDark = chart.config.options?.isDark ?? true;
    const lineStroke = isDark ? 'rgba(255, 255, 255, 0.28)' : 'rgba(0, 0, 0, 0.25)';
    const badgeBg = isDark ? '#1e222d' : '#ffffff';
    const badgeBorder = isDark ? '#363c4e' : '#cbd5e1';
    const textColor = isDark ? '#f8fafc' : '#0f172a';

    ctx.save();

    // 1. Dotted Vertical Crosshair Line
    ctx.beginPath();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = lineStroke;
    ctx.lineWidth = 1;
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();

    // 2. Dotted Horizontal Crosshair Line
    ctx.beginPath();
    ctx.moveTo(chartArea.left, y);
    ctx.lineTo(chartArea.right, y);
    ctx.stroke();
    ctx.setLineDash([]);

    // 3. Floating Date Badge on X-Axis (bottom)
    const dataIndex = active[0].index;
    const dateLabel = chart.data?.labels?.[dataIndex] || '';
    if (dateLabel) {
      ctx.font = '600 10px JetBrains Mono, monospace';
      const textWidth = ctx.measureText(dateLabel).width;
      const bWidth = textWidth + 12;
      const bHeight = 18;
      const bX = Math.min(Math.max(x - bWidth / 2, chartArea.left), chartArea.right - bWidth);
      const bY = chartArea.bottom + 2;

      ctx.fillStyle = badgeBg;
      ctx.strokeStyle = badgeBorder;
      ctx.lineWidth = 1;
      drawPill(ctx, bX, bY, bWidth, bHeight, 4);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = textColor;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(dateLabel, bX + bWidth / 2, bY + bHeight / 2);
    }

    // 4. Floating Price Badge on Y-Axis (right)
    const rawPrice = scales.y ? scales.y.getValueForPixel(y) : null;
    if (Number.isFinite(rawPrice)) {
      const priceLabel = chart.config.options?.plugins?.t212Crosshair?.formatPrice
        ? chart.config.options.plugins.t212Crosshair.formatPrice(rawPrice)
        : rawPrice.toFixed(2);
      ctx.font = '600 10px JetBrains Mono, monospace';
      const textWidth = ctx.measureText(priceLabel).width;
      const bWidth = textWidth + 12;
      const bHeight = 18;
      const bX = chartArea.right - bWidth;
      const bY = Math.min(Math.max(y - bHeight / 2, chartArea.top), chartArea.bottom - bHeight);

      ctx.fillStyle = badgeBg;
      ctx.strokeStyle = badgeBorder;
      ctx.lineWidth = 1;
      drawPill(ctx, bX, bY, bWidth, bHeight, 4);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = textColor;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(priceLabel, bX + bWidth / 2, bY + bHeight / 2);
    }

    ctx.restore();
  },
};

// Dashed last-price line and right-edge terminal corridor tags (p95, last, p05)
const lastPricePlugin = {
  id: 't212LastPrice',
  afterDraw: (chart) => {
    const config = chart.config.options?.plugins?.t212LastPrice;
    const price = config?.value;
    if (!Number.isFinite(price) || !chart.scales?.y || !chart.chartArea) return;
    const { ctx, chartArea, scales } = chart;
    const y = scales.y.getPixelForValue(price);
    if (y < chartArea.top || y > chartArea.bottom) return;

    const color = config?.color || '#8b93a7';
    const label = config?.label || '';
    const splitIdx = chart.data?.forecastSplitIndex;

    // Draw horizontal dashed line across the historical region
    let lineEndX = chartArea.right;
    if (splitIdx != null && scales.x) {
      const splitPixel = scales.x.getPixelForValue(splitIdx);
      if (splitPixel != null && splitPixel > chartArea.left && splitPixel < chartArea.right) {
        lineEndX = splitPixel;
      }
    }

    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.moveTo(chartArea.left, y);
    ctx.lineTo(lineEndX, y);
    ctx.stroke();
    ctx.setLineDash([]);

    // 1. Current Price Badge Pill
    ctx.font = '700 10px JetBrains Mono, monospace';
    const width = ctx.measureText(label).width + 12;
    const tagY = Math.min(Math.max(y - 9, chartArea.top), chartArea.bottom - 18);
    ctx.fillStyle = color;
    drawPill(ctx, chartArea.right - width, tagY, width, 18, 4);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, chartArea.right - width / 2, tagY + 9);

    // 2. Terminal Upper Bound Badge (p95) on Right Y-Scale
    const isDark = chart.config.options?.isDark ?? true;
    const cyanTagBg = isDark ? '#0284c7' : '#0369a1';
    const upperVal = config?.upperTerminal;
    const lowerVal = config?.lowerTerminal;

    if (Number.isFinite(upperVal)) {
      const yUpper = scales.y.getPixelForValue(upperVal);
      if (yUpper >= chartArea.top && yUpper <= chartArea.bottom && Math.abs(yUpper - y) > 16) {
        const uLabel = config?.upperLabel || upperVal.toFixed(2);
        const uWidth = ctx.measureText(uLabel).width + 10;
        const uY = Math.min(Math.max(yUpper - 8, chartArea.top), chartArea.bottom - 16);
        ctx.fillStyle = cyanTagBg;
        drawPill(ctx, chartArea.right - uWidth, uY, uWidth, 16, 3);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(uLabel, chartArea.right - uWidth / 2, uY + 8);
      }
    }

    // 3. Terminal Lower Bound Badge (p05) on Right Y-Scale
    if (Number.isFinite(lowerVal)) {
      const yLower = scales.y.getPixelForValue(lowerVal);
      if (yLower >= chartArea.top && yLower <= chartArea.bottom && Math.abs(yLower - y) > 16) {
        const lLabel = config?.lowerLabel || lowerVal.toFixed(2);
        const lWidth = ctx.measureText(lLabel).width + 10;
        const lY = Math.min(Math.max(yLower - 8, chartArea.top), chartArea.bottom - 16);
        ctx.fillStyle = cyanTagBg;
        drawPill(ctx, chartArea.right - lWidth, lY, lWidth, 16, 3);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(lLabel, chartArea.right - lWidth / 2, lY + 8);
      }
    }

    ctx.restore();
  },
};

// Shaded expected volatility cone past the last actual close price with milestone marker.
const forecastRegionPlugin = {
  id: 't212ForecastRegion',
  beforeDraw: (chart) => {
    const { ctx, chartArea, scales } = chart;
    const splitIdx = chart.data?.forecastSplitIndex;
    if (splitIdx == null || !scales.x || !chartArea) return;
    const xPos = scales.x.getPixelForValue(splitIdx);
    if (xPos == null || xPos < chartArea.left || xPos > chartArea.right) return;

    const isDark = chart.config.options?.isDark ?? true;
    ctx.save();

    // Subtle gradient background shading the forward projection horizon
    const grad = ctx.createLinearGradient(xPos, 0, chartArea.right, 0);
    if (isDark) {
      grad.addColorStop(0, 'rgba(56, 189, 248, 0.08)');
      grad.addColorStop(1, 'rgba(56, 189, 248, 0.02)');
    } else {
      grad.addColorStop(0, 'rgba(2, 132, 199, 0.08)');
      grad.addColorStop(1, 'rgba(2, 132, 199, 0.02)');
    }
    ctx.fillStyle = grad;
    ctx.fillRect(xPos, chartArea.top, chartArea.right - xPos, chartArea.bottom - chartArea.top);

    // Vertical milestone demarcation line
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = isDark ? 'rgba(56, 189, 248, 0.45)' : 'rgba(2, 132, 199, 0.45)';
    ctx.lineWidth = 1.5;
    ctx.moveTo(xPos, chartArea.top);
    ctx.lineTo(xPos, chartArea.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Sleek TradingView style milestone badge
    const badgeText = `${chart.data?.horizonLabel || '5D'} FORECAST CONE`;
    ctx.font = '700 9px JetBrains Mono, monospace';
    const textWidth = ctx.measureText(badgeText).width;
    const bW = textWidth + 14;
    const bH = 18;
    const bX = xPos + 8;
    const bY = chartArea.top + 10;

    ctx.fillStyle = isDark ? 'rgba(15, 23, 42, 0.85)' : 'rgba(255, 255, 255, 0.9)';
    ctx.strokeStyle = isDark ? 'rgba(56, 189, 248, 0.35)' : 'rgba(2, 132, 199, 0.35)';
    ctx.lineWidth = 1;
    drawPill(ctx, bX, bY, bW, bH, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = isDark ? '#38bdf8' : '#0284c7';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(badgeText, bX + bW / 2, bY + bH / 2);

    ctx.restore();
  },
};

function volatilityConeDatasets(g3outlook, historyPrices, colors, horizon = 5, isDark = true) {
  const sets = [];
  const entry = g3outlook?.byHorizon?.[horizon] || g3outlook?.byHorizon?.[5];
  const bandUpper95 = entry?.volatility_cone?.p95;
  const bandUpper75 = entry?.volatility_cone?.p75;
  const bandMedian = entry?.volatility_cone?.p50;
  const bandLower25 = entry?.volatility_cone?.p25;
  const bandLower05 = entry?.volatility_cone?.p05;
  const bandDates = entry?.future_dates;

  if (
    !Array.isArray(bandDates) ||
    !Array.isArray(bandUpper95) ||
    !Array.isArray(bandLower05) ||
    bandDates.length === 0 ||
    !Array.isArray(historyPrices) ||
    historyPrices.length === 0
  ) {
    return { sets, futureLabels: [] };
  }

  const historyLength = historyPrices.length;
  const lastPrice = historyPrices[historyLength - 1];
  // Anchor cleanly at the last historical price point
  const prefix = Array(Math.max(0, historyLength - 1)).fill(null);
  const futureCount = bandDates.length;

  const cyanBorder = isDark ? '#38bdf8' : '#0284c7';
  const outerCorridorBg = isDark ? 'rgba(56, 189, 248, 0.08)' : 'rgba(2, 132, 199, 0.07)';
  const outerBoundary = isDark ? 'rgba(56, 189, 248, 0.50)' : 'rgba(2, 132, 199, 0.50)';
  const innerCorridorBg = isDark ? 'rgba(56, 189, 248, 0.14)' : 'rgba(2, 132, 199, 0.12)';
  const innerBoundary = isDark ? 'rgba(56, 189, 248, 0.30)' : 'rgba(2, 132, 199, 0.30)';

  const hasInnerBands = Array.isArray(bandUpper75) && Array.isArray(bandLower25);

  // 1. Upper Bound (p95)
  sets.push({
    label: 'Expected volatility range (upper)',
    data: [...prefix, lastPrice, ...bandUpper95.slice(0, futureCount)],
    borderColor: outerBoundary,
    borderWidth: 1.5,
    borderDash: [4, 4],
    backgroundColor: outerCorridorBg,
    pointRadius: 0,
    pointHoverRadius: 4,
    fill: hasInnerBands ? '+4' : '+2',
    tension: 0.2,
    spanGaps: false,
  });

  // 2. Inner Upper Bound (p75) - when available
  if (hasInnerBands) {
    sets.push({
      label: 'Expected 50% corridor (upper)',
      data: [...prefix, lastPrice, ...bandUpper75.slice(0, futureCount)],
      borderColor: innerBoundary,
      borderWidth: 1,
      borderDash: [2, 2],
      backgroundColor: innerCorridorBg,
      pointRadius: 0,
      pointHoverRadius: 3,
      fill: '+2',
      tension: 0.2,
      spanGaps: false,
    });
  }

  // 3. Expected Median Path (p50)
  const totalPoints = prefix.length + 1 + futureCount;
  sets.push({
    label: `${horizon}D Expected Price (median p50)`,
    data: [
      ...prefix,
      lastPrice,
      ...(bandMedian ? bandMedian.slice(0, futureCount) : Array(futureCount).fill(lastPrice)),
    ],
    borderColor: cyanBorder,
    borderWidth: 2.2,
    borderDash: [5, 4],
    backgroundColor: 'transparent',
    pointRadius: (ctx) => (ctx.dataIndex === totalPoints - 1 ? 5 : 0),
    pointBackgroundColor: cyanBorder,
    pointBorderColor: '#ffffff',
    pointBorderWidth: 2,
    pointHoverRadius: 6,
    pointHoverBackgroundColor: cyanBorder,
    fill: false,
    tension: 0.2,
    spanGaps: false,
  });

  // 4. Inner Lower Bound (p25) - when available
  if (hasInnerBands) {
    sets.push({
      label: 'Expected 50% corridor (lower)',
      data: [...prefix, lastPrice, ...bandLower25.slice(0, futureCount)],
      borderColor: innerBoundary,
      borderWidth: 1,
      borderDash: [2, 2],
      backgroundColor: 'transparent',
      pointRadius: 0,
      pointHoverRadius: 3,
      fill: false,
      tension: 0.2,
      spanGaps: false,
    });
  }

  // 5. Lower Bound (p05)
  sets.push({
    label: 'Expected volatility range (lower)',
    data: [...prefix, lastPrice, ...bandLower05.slice(0, futureCount)],
    borderColor: outerBoundary,
    borderWidth: 1.5,
    borderDash: [4, 4],
    backgroundColor: 'transparent',
    pointRadius: 0,
    pointHoverRadius: 4,
    fill: false,
    tension: 0.2,
    spanGaps: false,
  });

  return { sets, futureLabels: bandDates.slice(0, futureCount) };
}

export default function PriceChart({ ticker, currencySymbol = '$', onHistorySettled = null }) {
  const isDark = useAppTheme();
  const { history, loading, error, meta, retry } = usePriceHistory(ticker);
  const { outlook: g3outlook } = useVolatilityOutlook(ticker);
  const [rangeId, setRangeId] = useState(null);
  const [forecastHorizon, setForecastHorizon] = useState(5);
  const [view, setView] = useState(null);
  const chartRef = useRef(null);
  const wrapRef = useRef(null);
  const reportedRef = useRef(null);

  const effectiveHistory = history;
  const showLoading = loading;
  const showError = !loading && error;

  // One settlement report per ticker: chart paint or failure.
  useEffect(() => {
    if (!onHistorySettled) return;
    const key = String(ticker || '').toUpperCase();
    if (effectiveHistory && reportedRef.current !== `${key}:ok`) {
      reportedRef.current = `${key}:ok`;
      onHistorySettled({
        ok: true,
        degraded: false,
        fetchMs: meta?.fetchMs ?? null,
        fromCache: Boolean(meta?.fromCache),
        marketDataCache: history?.marketDataCache ?? null,
      });
    } else if (showError && reportedRef.current !== `${key}:ok` && reportedRef.current !== `${key}:error`) {
      reportedRef.current = `${key}:error`;
      onHistorySettled({ ok: false, degraded: false, fetchMs: null, fromCache: false, marketDataCache: null });
    }
  }, [effectiveHistory, showError, history, meta, ticker, onHistorySettled]);

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
        grid: 'rgba(255,255,255,0.06)',
        tick: '#8b93a7',
        tooltipBg: '#0d131f',
        tooltipTitle: '#e2e8f0',
        tooltipBody: '#94a3b8',
        tooltipBorder: 'rgba(255,255,255,0.12)',
        estimate: '#38bdf8',
        bandFill: 'rgba(56,189,248,0.12)',
      };
    }
    return {
      line: up ? '#059669' : '#dc2626',
      areaTop: up ? 'rgba(5,150,105,0.18)' : 'rgba(220,38,38,0.18)',
      grid: 'rgba(0,0,0,0.06)',
      tick: '#64748b',
      tooltipBg: '#ffffff',
      tooltipTitle: '#0f172a',
      tooltipBody: '#475569',
      tooltipBorder: 'rgba(0,0,0,0.12)',
      estimate: '#0284c7',
      bandFill: 'rgba(2,132,199,0.10)',
    };
  }, [isDark, stats.up]);

  const chartData = useMemo(() => {
    const labels = points.labels.map((label) => formatAxisLabel(label, points.isIntraday));
    const historyPadded = [...points.prices];
    const { sets: coneSets, futureLabels } = volatilityConeDatasets(
      g3outlook,
      points.prices,
      colors,
      forecastHorizon,
      isDark
    );

    futureLabels.forEach((futureDate, index) => {
      historyPadded.push(null);
      labels.push(`+${index + 1}d (${formatAxisLabel(futureDate, false)})`);
    });

    return {
      labels,
      horizonLabel: `${forecastHorizon}D`,
      forecastSplitIndex: coneSets.length > 0 ? points.prices.length - 1 : null,
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
          tension: 0.2,
          fill: true,
          spanGaps: false,
        },
        ...coneSets,
      ],
    };
  }, [points, colors, g3outlook, forecastHorizon, isDark]);

  const futureCount = g3outlook?.byHorizon?.[forecastHorizon]?.future_dates?.length || 0;
  const totalLabels = points.labels.length + futureCount;
  const fullView = totalLabels > 0 ? { start: 0, end: totalLabels - 1 } : null;
  const effectiveView = view || fullView;
  const isZoomed = Boolean(view && fullView && (view.start > 0 || view.end < fullView.end));

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
    const wrap = wrapRef.current;
    if (!wrap) return undefined;
    const onWheel = (event) => {
      event.preventDefault();
      const rect = wrap.getBoundingClientRect();
      const factor = event.deltaY < 0 ? 0.85 : 1.18;
      zoomAt(event.clientX - rect.left, factor);
    };
    const onDblClick = () => setView(null);
    wrap.addEventListener('wheel', onWheel, { passive: false });
    wrap.addEventListener('dblclick', onDblClick);
    return () => {
      wrap.removeEventListener('wheel', onWheel);
      wrap.removeEventListener('dblclick', onDblClick);
    };
  }, [zoomAt]);

  const [hoveredIndex, setHoveredIndex] = useState(null);

  // Active horizon forecast metadata
  const activeForecast = g3outlook?.byHorizon?.[forecastHorizon];
  const activeVolAnnual = activeForecast?.forecast?.expected_annualized_volatility || activeForecast?.forecast?.predicted_volatility;
  const p05Terminal = activeForecast?.volatility_cone?.p05?.at(-1);
  const p95Terminal = activeForecast?.volatility_cone?.p95?.at(-1);
  const riskLevel = activeForecast?.evidence?.risk_level || 'Moderate';

  const hoveredDetails = useMemo(() => {
    if (hoveredIndex == null || !effectiveHistory) return null;
    const historyLen = points.prices.length;
    if (hoveredIndex < historyLen) {
      const price = points.prices[hoveredIndex];
      const prevPrice = hoveredIndex > 0 ? points.prices[hoveredIndex - 1] : points.prices[0];
      const change = Number.isFinite(price) && Number.isFinite(prevPrice) ? price - prevPrice : 0;
      const changePct = Number.isFinite(prevPrice) && prevPrice > 0 ? (change / prevPrice) * 100 : 0;
      const sign = change >= 0 ? '+' : '';
      return {
        isForecast: false,
        date: points.labels[hoveredIndex] || '',
        price: formatMoneyLocal(price, currencySymbol),
        change: `${sign}${formatMoneyLocal(change, currencySymbol)}`,
        changePct: `${sign}${changePct.toFixed(2)}%`,
        up: change >= 0,
      };
    }
    const futureOffset = hoveredIndex - historyLen;
    const futureDates = activeForecast?.future_dates || [];
    const p50Arr = activeForecast?.volatility_cone?.p50 || [];
    const p05Arr = activeForecast?.volatility_cone?.p05 || [];
    const p95Arr = activeForecast?.volatility_cone?.p95 || [];

    const date = futureDates[futureOffset] || `+${futureOffset + 1}d`;
    const p50 = p50Arr[futureOffset] ?? stats.last;
    const p05 = p05Arr[futureOffset];
    const p95 = p95Arr[futureOffset];
    const spread = Number.isFinite(p95) && Number.isFinite(p05) ? p95 - p05 : null;

    return {
      isForecast: true,
      day: futureOffset + 1,
      date: formatAxisLabel(date, false),
      median: formatMoneyLocal(p50, currencySymbol),
      range: `${formatMoneyLocal(p05, currencySymbol)} – ${formatMoneyLocal(p95, currencySymbol)}`,
      spread: spread != null ? `±${formatMoneyLocal(spread / 2, currencySymbol)}` : '—',
    };
  }, [hoveredIndex, points, effectiveHistory, activeForecast, stats.last, currencySymbol]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    isDark,
    interaction: { mode: 'index', intersect: false },
    onHover: (event, activeElements) => {
      if (!activeElements || activeElements.length === 0) {
        if (hoveredIndex !== null) setHoveredIndex(null);
        return;
      }
      const idx = activeElements[0].index;
      if (idx !== hoveredIndex) {
        setHoveredIndex(idx);
      }
    },
    plugins: {
      legend: { display: false },
      t212LastPrice: {
        value: stats.last,
        color: stats.up ? (isDark ? '#00f5a0' : '#10b981') : (isDark ? '#ff5c5c' : '#ef4444'),
        label: formatMoneyLocal(stats.last, currencySymbol),
        upperTerminal: p95Terminal,
        lowerTerminal: p05Terminal,
        upperLabel: `${formatMoneyLocal(p95Terminal, currencySymbol)} p95`,
        lowerLabel: `${formatMoneyLocal(p05Terminal, currencySymbol)} p05`,
      },
      t212Crosshair: {
        formatPrice: (v) => formatMoneyLocal(v, currencySymbol),
      },
      tooltip: {
        backgroundColor: colors.tooltipBg,
        titleColor: colors.tooltipTitle,
        bodyColor: colors.tooltipBody,
        borderColor: colors.tooltipBorder,
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
        displayColors: true,
        titleFont: { family: 'Inter, sans-serif', size: 12, weight: '700' },
        bodyFont: { family: 'JetBrains Mono, monospace', size: 11 },
        callbacks: {
          title: (items) => (items.length > 0 ? String(chartData.labels[items[0].dataIndex] ?? '') : ''),
          label: (ctx) => {
            const val = Number(ctx.parsed.y);
            if (!Number.isFinite(val)) return null;
            return ` ${ctx.dataset.label}: ${formatMoneyLocal(val, currencySymbol)}`;
          },
        },
      },
    },
    scales: {
      x: {
        min: effectiveView?.start,
        max: effectiveView?.end,
        ticks: {
          color: colors.tick,
          maxTicksLimit: 10,
          maxRotation: 0,
          font: { family: 'JetBrains Mono, monospace', size: 10 },
        },
        grid: {
          display: true,
          color: colors.grid,
          borderDash: [2, 2],
          drawTicks: false,
        },
        border: {
          display: false,
        },
      },
      y: {
        ...yBounds,
        position: 'right',
        ticks: {
          color: colors.tick,
          font: { family: 'JetBrains Mono, monospace', size: 10 },
          callback: (v) => formatMoneyLocal(Number(v), currencySymbol),
        },
        grid: {
          display: true,
          color: colors.grid,
          borderDash: [2, 2],
          drawTicks: false,
        },
        border: {
          display: false,
        },
      },
    },
  }), [isDark, colors, chartData, effectiveView, yBounds, stats.last, stats.up, currencySymbol, p95Terminal, p05Terminal, hoveredIndex]);

  const selectRange = useCallback((id) => {
    setRangeId(id);
    setView(null);
  }, []);

  const changeClass = stats.up ? 'up' : 'down';
  const sign = stats.change >= 0 ? '+' : '';

  return (
    <section id="chartContainer" className="t212-chart-section" aria-label={`${ticker} price chart`}>
      {/* Chart Header */}
      <div className="t212-chart-head">
        <div className="t212-price-block">
          <span className="t212-ticker">{ticker}</span>
          <strong className="t212-price mono">{formatMoneyLocal(stats.last, currencySymbol)}</strong>
          <span className={`t212-change ${changeClass}`}>
            {sign}{formatMoneyLocal(stats.change, currencySymbol)} ({sign}{stats.changePct.toFixed(2)}%)
          </span>
          <span className="t212-range-name">{activeRange === 'MAX' ? 'All time' : activeRange}</span>
        </div>

        <div className="t212-controls-group">
          {/* Time Range Selector */}
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

          {/* Forecast Horizon Selector */}
          <div className="t212-horizon-tabs" role="group" aria-label="Forecast horizon">
            <span className="horizon-label">Forecast:</span>
            {[5, 10, 20].map((h) => (
              <button
                key={h}
                type="button"
                className={`horizon-pill ${forecastHorizon === h ? 'active' : ''}`}
                onClick={() => setForecastHorizon(h)}
              >
                {h}D
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Dynamic TradingView OHLC / Hover Strip */}
      <div className="t212-ohlc-bar" aria-live="polite">
        {hoveredDetails ? (
          hoveredDetails.isForecast ? (
            <div className="t212-ohlc-inner forecast-mode">
              <span className="t212-ohlc-tag forecast-badge">FORECAST +{hoveredDetails.day}D</span>
              <span className="t212-ohlc-item">
                <span className="label">Date:</span>
                <strong className="val">{hoveredDetails.date}</strong>
              </span>
              <span className="t212-ohlc-item">
                <span className="label">Expected Median (p50):</span>
                <strong className="val cyan">{hoveredDetails.median}</strong>
              </span>
              <span className="t212-ohlc-item">
                <span className="label">90% Range:</span>
                <strong className="val">{hoveredDetails.range}</strong>
              </span>
              <span className="t212-ohlc-item">
                <span className="label">Spread:</span>
                <strong className="val highlight">{hoveredDetails.spread}</strong>
              </span>
            </div>
          ) : (
            <div className="t212-ohlc-inner">
              <span className="t212-ohlc-tag">HISTORICAL</span>
              <span className="t212-ohlc-item">
                <span className="label">Date:</span>
                <strong className="val">{hoveredDetails.date}</strong>
              </span>
              <span className="t212-ohlc-item">
                <span className="label">Close:</span>
                <strong className="val">{hoveredDetails.price}</strong>
              </span>
              <span className="t212-ohlc-item">
                <span className="label">Session:</span>
                <strong className={`val ${hoveredDetails.up ? 'up' : 'down'}`}>
                  {hoveredDetails.change} ({hoveredDetails.changePct})
                </strong>
              </span>
            </div>
          )
        ) : (
          <div className="t212-ohlc-inner idle-mode">
            <span className="t212-ohlc-tag idle-pulse">LIVE</span>
            <span className="t212-ohlc-item">
              <span className="label">Latest Close:</span>
              <strong className="val">{formatMoneyLocal(stats.last, currencySymbol)}</strong>
            </span>
            <span className="t212-ohlc-item">
              <span className="label">{activeRange} Movement:</span>
              <strong className={`val ${stats.up ? 'up' : 'down'}`}>
                {stats.up ? '▲' : '▼'} {formatMoneyLocal(Math.abs(stats.change), currencySymbol)} ({stats.changePct.toFixed(2)}% net)
              </strong>
            </span>
            <span className="t212-ohlc-item">
              <span className="label">Horizon:</span>
              <strong className="val cyan">{forecastHorizon}D Zero-Drift Volatility Cone</strong>
            </span>
          </div>
        )}
      </div>

      {/* Quick Forecast Stats Strip */}
      {activeForecast && Number.isFinite(p05Terminal) && Number.isFinite(p95Terminal) && (
        <div className="chart-quick-stats">
          <div className="chart-stat-item">
            <span className="chart-stat-label">{forecastHorizon}-Day Expected Corridor</span>
            <span className="chart-stat-value highlight">
              {formatMoneyLocal(p05Terminal, currencySymbol)} – {formatMoneyLocal(p95Terminal, currencySymbol)}
            </span>
          </div>
          <div className="chart-stat-item">
            <span className="chart-stat-label">Annualized Volatility</span>
            <span className="chart-stat-value">
              {activeVolAnnual != null ? `${(activeVolAnnual * 100).toFixed(1)}%` : '—'}
            </span>
          </div>
          <div className="chart-stat-item">
            <span className="chart-stat-label">Risk Rating</span>
            <span className="chart-stat-value">
              <span className={`risk-pill risk-${riskLevel.toLowerCase()}`}>{riskLevel}</span>
            </span>
          </div>
          <div className="chart-stat-item">
            <span className="chart-stat-label">Horizon Target</span>
            <span className="chart-stat-value">
              Next {forecastHorizon} sessions ({activeForecast.future_dates?.at(-1) || '—'})
            </span>
          </div>
        </div>
      )}

      {/* Chart Canvas */}
      <div
        className="t212-chart-wrap"
        ref={wrapRef}
        onMouseLeave={() => setHoveredIndex(null)}
        style={{ height: 'clamp(390px, 54vh, 580px)', position: 'relative' }}
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

      <p className="t212-chart-hint">
        Scroll to zoom · drag to pan · double-click to reset. Past price line seamlessly connects into the {forecastHorizon}-day expected zero-drift volatility cone.
      </p>
    </section>
  );
}
