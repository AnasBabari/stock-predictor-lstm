/* ===================================================================
   StockLSTM — Frontend Application v2
   =================================================================== */

const API_BASE = 'http://127.0.0.1:8000';

// ── DOM refs ────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const tickerInput    = $('tickerInput');
const predictBtn     = $('predictBtn');
const errorMsg       = $('errorMsg');
const loading        = $('loading');
const statsBar       = $('statsBar');
const chartContainer = $('chartContainer');
const chartTitle     = $('chartTitle');
const themeToggle    = $('themeToggle');
const searchDropdown = $('searchDropdown');
const forecastDays   = $('forecastDays');
const stockInfoEl    = $('stockInfo');
const infoGrid       = $('infoGrid');
const metricsCard    = $('metricsCard');
const watchlistItems = $('watchlistItems');
const historyItems   = $('historyItems');

// ── State ───────────────────────────────────────────────────────────
let stockChart       = null;
let currentStockData = null;
let currentDaysView  = 21;
let currentTheme     = localStorage.getItem('stocklstm-theme') || 'dark';

// Apply persisted theme on load
document.documentElement.setAttribute('data-theme', currentTheme);
themeToggle.textContent = currentTheme === 'dark' ? '🌙' : '☀️';

// ── Splash Screen ───────────────────────────────────────────────────
window.addEventListener('load', () => {
    setTimeout(() => $('splashScreen').classList.add('fade-out'), 1000);
    renderWatchlist();
    renderHistory();
});

// ── Theme Toggle ────────────────────────────────────────────────────
themeToggle.addEventListener('click', () => {
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', currentTheme);
    themeToggle.textContent = currentTheme === 'dark' ? '🌙' : '☀️';
    localStorage.setItem('stocklstm-theme', currentTheme);
    if (currentStockData) renderChart(currentStockData);
});

// ── Predict Button ──────────────────────────────────────────────────
predictBtn.addEventListener('click', () => {
    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) { showError('Please enter a ticker symbol.'); return; }
    fetchPrediction(ticker);
});

tickerInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') predictBtn.click();
});

// ── Search Autocomplete ─────────────────────────────────────────────
let searchTimeout = null;

tickerInput.addEventListener('input', () => {
    const q = tickerInput.value.trim();
    clearTimeout(searchTimeout);
    if (q.length < 2) { closeDropdown(); return; }
    searchTimeout = setTimeout(() => fetchSuggestions(q), 250);
});

document.addEventListener('click', (e) => {
    if (!e.target.closest('.input-wrapper')) closeDropdown();
});

async function fetchSuggestions(query) {
    try {
        const res = await fetch(`${API_BASE}/api/v1/search?query=${encodeURIComponent(query)}`);
        if (!res.ok) return;
        const data = await res.json();
        renderDropdown(data.results);
    } catch { /* silent */ }
}

function renderDropdown(results) {
    if (!results.length) { closeDropdown(); return; }
    searchDropdown.innerHTML = results.map(r => `
        <div class="dropdown-item" data-ticker="${r.ticker}">
            <div>
                <div class="dropdown-name">${r.name}</div>
                <div class="dropdown-type">${r.type}</div>
            </div>
            <div class="dropdown-ticker">${r.ticker}</div>
        </div>
    `).join('');

    searchDropdown.querySelectorAll('.dropdown-item').forEach(item => {
        item.addEventListener('click', () => {
            tickerInput.value = item.dataset.ticker;
            closeDropdown();
            fetchPrediction(item.dataset.ticker);
        });
    });
    searchDropdown.classList.remove('hidden');
}

function closeDropdown() {
    searchDropdown.classList.add('hidden');
    searchDropdown.innerHTML = '';
}

// ── Timeframe Filters ───────────────────────────────────────────────
document.querySelectorAll('.time-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        currentDaysView = parseInt(e.target.dataset.days);
        if (currentStockData) renderChart(currentStockData);
    });
});

// ── Fetch Prediction ────────────────────────────────────────────────
async function fetchPrediction(ticker) {
    resetUI();
    showLoading(true);
    predictBtn.disabled = true;

    const days = parseInt(forecastDays.value);

    try {
        // Fetch prediction + stock info in parallel
        const [predRes, infoRes] = await Promise.allSettled([
            fetch(`${API_BASE}/api/v1/predict?ticker=${ticker}&days=${days}`),
            fetch(`${API_BASE}/api/v1/info?ticker=${ticker}`),
        ]);

        // Handle prediction
        if (predRes.status === 'fulfilled' && predRes.value.ok) {
            currentStockData = await predRes.value.json();
            renderStats(currentStockData);
            renderChart(currentStockData);
            renderMetrics(currentStockData);
            addToHistory(currentStockData);
            toast('success', `Forecast ready for ${currentStockData.ticker}`);
        } else {
            const errData = predRes.status === 'fulfilled'
                ? await predRes.value.json().catch(() => ({}))
                : {};
            throw new Error(errData.detail || `Prediction failed (${predRes.value?.status || 'network error'})`);
        }

        // Handle stock info (non-critical)
        if (infoRes.status === 'fulfilled' && infoRes.value.ok) {
            const info = await infoRes.value.json();
            renderStockInfo(info);
        }

    } catch (err) {
        console.error(err);
        const msg = err.message.includes('Failed to fetch')
            ? 'Could not connect to the backend. Make sure the server is running.'
            : err.message.includes('400')
            ? 'Invalid ticker or not enough data. Try a different symbol.'
            : err.message;
        showError(msg);
    } finally {
        showLoading(false);
        predictBtn.disabled = false;
    }
}

// ── Render Stock Info Cards ─────────────────────────────────────────
function renderStockInfo(info) {
    const cards = [
        { label: 'Company', value: info.name || '—' },
        { label: 'Sector', value: info.sector || '—' },
        { label: 'Market Cap', value: formatLargeNum(info.marketCap) },
        { label: 'P/E Ratio', value: info.peRatio ? info.peRatio.toFixed(2) : '—' },
        { label: '52W High', value: info.fiftyTwoWeekHigh ? `$${info.fiftyTwoWeekHigh.toFixed(2)}` : '—' },
        { label: '52W Low', value: info.fiftyTwoWeekLow ? `$${info.fiftyTwoWeekLow.toFixed(2)}` : '—' },
        { label: 'Avg Volume', value: formatLargeNum(info.avgVolume) },
        { label: 'Prev Close', value: info.previousClose ? `$${info.previousClose.toFixed(2)}` : '—' },
    ];

    infoGrid.innerHTML = cards.map(c => `
        <div class="info-card">
            <span class="info-card-label">${c.label}</span>
            <span class="info-card-value">${c.value}</span>
        </div>
    `).join('');

    stockInfoEl.classList.remove('hidden');
}

// ── Render Stats Bar ────────────────────────────────────────────────
function renderStats(data) {
    const lastClose = data.historical_prices.at(-1);
    const forecast  = data.predicted_prices[0];
    const isUp      = forecast > lastClose;
    const change    = forecast - lastClose;
    const changePct = ((change / lastClose) * 100).toFixed(2);

    const changeEl = $('statChange');
    const trendEl  = $('statTrend');

    $('statTicker').textContent    = data.ticker;
    $('statLastClose').textContent = `$${lastClose.toFixed(2)}`;
    $('statForecast').textContent  = `$${forecast.toFixed(2)}`;

    changeEl.textContent = `${isUp ? '+' : ''}${changePct}%`;
    changeEl.style.color = isUp ? 'var(--bullish)' : 'var(--bearish)';

    trendEl.textContent  = isUp ? '▲ Bullish' : '▼ Bearish';
    trendEl.style.color  = isUp ? 'var(--bullish)' : 'var(--bearish)';

    statsBar.classList.remove('hidden');
    reflow(statsBar);
}

// ── Render Chart ────────────────────────────────────────────────────
function renderChart(data) {
    chartTitle.textContent = `${data.ticker} — Historical vs Predicted`;
    chartContainer.classList.remove('hidden');

    const isDark       = currentTheme === 'dark';
    const gridColor    = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.05)';
    const tickColor    = isDark ? '#5a5a7a' : '#94a3b8';
    const tooltipBg    = isDark ? '#0d0d1a' : '#ffffff';
    const tooltipTitle = isDark ? '#e8e8f0' : '#1e293b';
    const tooltipBody  = isDark ? '#a0a0c0' : '#475569';
    const tooltipBorder= isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
    const legendColor  = isDark ? '#a0a0c0' : '#475569';
    const histColor    = isDark ? '#58a6ff' : '#3b82f6';
    const predColor    = isDark ? '#00f5a0' : '#10b981';

    const total      = data.historical_prices.length;
    const sliceIdx   = Math.max(0, total - currentDaysView);
    const sliceDates = data.historical_dates.slice(sliceIdx);
    const slicePrices= data.historical_prices.slice(sliceIdx);
    const allDates   = [...sliceDates, ...data.future_dates];

    const historicalPadded = [
        ...slicePrices,
        ...Array(data.future_dates.length).fill(null),
    ];
    const predictedPadded = [
        ...Array(slicePrices.length - 1).fill(null),
        slicePrices.at(-1),
        ...data.predicted_prices,
    ];

    if (stockChart) stockChart.destroy();

    const ctx = document.getElementById('stockChart').getContext('2d');

    // Gradient fills
    const histGrad = ctx.createLinearGradient(0, 0, 0, 400);
    histGrad.addColorStop(0, isDark ? 'rgba(88,166,255,0.12)' : 'rgba(59,130,246,0.08)');
    histGrad.addColorStop(1, 'transparent');

    const predGrad = ctx.createLinearGradient(0, 0, 0, 400);
    predGrad.addColorStop(0, isDark ? 'rgba(0,245,160,0.12)' : 'rgba(16,185,129,0.08)');
    predGrad.addColorStop(1, 'transparent');

    stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allDates,
            datasets: [
                {
                    label: 'Historical Price',
                    data: historicalPadded,
                    borderColor: histColor,
                    backgroundColor: histGrad,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: histColor,
                    tension: 0.35,
                    fill: true,
                    spanGaps: false,
                },
                {
                    label: 'Predicted Price',
                    data: predictedPadded,
                    borderColor: predColor,
                    backgroundColor: predGrad,
                    borderWidth: 2.5,
                    pointRadius: 4,
                    pointBackgroundColor: predColor,
                    pointHoverRadius: 6,
                    borderDash: [6, 3],
                    tension: 0.35,
                    fill: true,
                    spanGaps: false,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: {
                    labels: {
                        color: legendColor,
                        font: { size: 12, family: 'Inter' },
                        usePointStyle: true,
                        padding: 20,
                    },
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: tooltipTitle,
                    bodyColor: tooltipBody,
                    borderColor: tooltipBorder,
                    borderWidth: 1,
                    cornerRadius: 10,
                    padding: 12,
                    titleFont: { family: 'Inter', weight: '600' },
                    bodyFont: { family: 'Inter' },
                    callbacks: {
                        label: (ctx) => {
                            const val = ctx.parsed.y;
                            return val != null ? ` $${val.toFixed(2)}` : null;
                        },
                    },
                },
            },
            scales: {
                x: {
                    ticks: { color: tickColor, maxTicksLimit: 10, font: { family: 'Inter', size: 11 } },
                    grid: { color: gridColor },
                },
                y: {
                    ticks: {
                        color: tickColor,
                        font: { family: 'Inter', size: 11 },
                        callback: (v) => `$${v.toFixed(0)}`,
                    },
                    grid: { color: gridColor },
                },
            },
        },
    });

    reflow(chartContainer);
}

// ── Render Metrics ──────────────────────────────────────────────────
function renderMetrics(data) {
    const m = data.metrics;
    $('metricRMSE').textContent = m && m.rmse != null ? `$${m.rmse.toFixed(2)}` : '—';
    $('metricMAE').textContent  = m && m.mae != null  ? `$${m.mae.toFixed(2)}`  : '—';
    $('metricDays').textContent = `${data.forecast_days} days`;
    metricsCard.classList.remove('hidden');
}

// ── Export: PNG ──────────────────────────────────────────────────────
$('exportPng').addEventListener('click', () => {
    if (!stockChart) return;
    const link = document.createElement('a');
    link.download = `${currentStockData?.ticker || 'chart'}_forecast.png`;
    link.href = stockChart.toBase64Image();
    link.click();
    toast('success', 'Chart exported as PNG');
});

// ── Export: CSV ──────────────────────────────────────────────────────
$('exportCsv').addEventListener('click', () => {
    if (!currentStockData) return;
    const d = currentStockData;
    let csv = 'Date,Price,Type\n';

    d.historical_dates.forEach((dt, i) => {
        csv += `${dt},${d.historical_prices[i].toFixed(2)},Historical\n`;
    });
    d.future_dates.forEach((dt, i) => {
        csv += `${dt},${d.predicted_prices[i].toFixed(2)},Predicted\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const link = document.createElement('a');
    link.download = `${d.ticker}_forecast.csv`;
    link.href = URL.createObjectURL(blob);
    link.click();
    URL.revokeObjectURL(link.href);
    toast('success', 'Data exported as CSV');
});

// ══════════════════════════════════════════════════════════════════════
//  WATCHLIST — localStorage powered
// ══════════════════════════════════════════════════════════════════════
const WL_KEY = 'stocklstm-watchlist';

function getWatchlist() {
    try { return JSON.parse(localStorage.getItem(WL_KEY)) || []; }
    catch { return []; }
}
function saveWatchlist(list) {
    localStorage.setItem(WL_KEY, JSON.stringify(list));
}

$('addWatchlist').addEventListener('click', () => {
    if (!currentStockData) return;
    const d = currentStockData;
    let list = getWatchlist();

    if (list.find(w => w.ticker === d.ticker)) {
        toast('info', `${d.ticker} is already in your watchlist`);
        return;
    }

    list.unshift({
        ticker: d.ticker,
        name: '',  // will be filled from info if available
        lastPrice: d.historical_prices.at(-1),
    });
    // Try to get the name from the info grid
    const nameCard = infoGrid.querySelector('.info-card-value');
    if (nameCard) list[0].name = nameCard.textContent;

    saveWatchlist(list);
    renderWatchlist();
    toast('success', `${d.ticker} added to watchlist`);
});

$('clearWatchlist').addEventListener('click', () => {
    saveWatchlist([]);
    renderWatchlist();
    toast('info', 'Watchlist cleared');
});

function renderWatchlist() {
    const list = getWatchlist();
    if (!list.length) {
        watchlistItems.innerHTML = '<p class="empty-state">No tickers saved yet. Click ⭐ after predicting to add one.</p>';
        return;
    }

    watchlistItems.innerHTML = list.map((w, i) => `
        <div class="watchlist-item" data-ticker="${w.ticker}" data-index="${i}">
            <span class="wl-ticker">${w.ticker}</span>
            <span class="wl-name">${w.name || ''}</span>
            <span class="wl-price">$${w.lastPrice?.toFixed(2) || '—'}</span>
            <button class="wl-remove" data-index="${i}" title="Remove">✕</button>
        </div>
    `).join('');

    // Click to predict
    watchlistItems.querySelectorAll('.watchlist-item').forEach(item => {
        item.addEventListener('click', (e) => {
            if (e.target.classList.contains('wl-remove')) return;
            tickerInput.value = item.dataset.ticker;
            fetchPrediction(item.dataset.ticker);
        });
    });

    // Remove button
    watchlistItems.querySelectorAll('.wl-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const list = getWatchlist();
            list.splice(parseInt(btn.dataset.index), 1);
            saveWatchlist(list);
            renderWatchlist();
            toast('info', 'Removed from watchlist');
        });
    });
}

// ══════════════════════════════════════════════════════════════════════
//  PREDICTION HISTORY — localStorage powered
// ══════════════════════════════════════════════════════════════════════
const HIST_KEY = 'stocklstm-history';
const MAX_HISTORY = 15;

function getHistory() {
    try { return JSON.parse(localStorage.getItem(HIST_KEY)) || []; }
    catch { return []; }
}
function saveHistory(list) {
    localStorage.setItem(HIST_KEY, JSON.stringify(list));
}

function addToHistory(data) {
    const lastClose = data.historical_prices.at(-1);
    const forecast  = data.predicted_prices[0];
    const change    = ((forecast - lastClose) / lastClose * 100).toFixed(2);

    let list = getHistory();
    // Avoid exact duplicates (same ticker & same minute)
    const now = new Date().toISOString().slice(0, 16);
    list = list.filter(h => !(h.ticker === data.ticker && h.date?.startsWith(now)));

    list.unshift({
        ticker: data.ticker,
        lastClose: lastClose,
        forecast: forecast,
        change: change,
        days: data.forecast_days,
        date: new Date().toISOString(),
    });

    if (list.length > MAX_HISTORY) list = list.slice(0, MAX_HISTORY);
    saveHistory(list);
    renderHistory();
}

$('clearHistory').addEventListener('click', () => {
    saveHistory([]);
    renderHistory();
    toast('info', 'History cleared');
});

function renderHistory() {
    const list = getHistory();
    if (!list.length) {
        historyItems.innerHTML = '<p class="empty-state">No predictions yet. Search a ticker to get started.</p>';
        return;
    }

    historyItems.innerHTML = list.map(h => {
        const isUp = parseFloat(h.change) >= 0;
        const color = isUp ? 'var(--bullish)' : 'var(--bearish)';
        const arrow = isUp ? '▲' : '▼';
        const dateStr = new Date(h.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        return `
            <div class="history-item" data-ticker="${h.ticker}">
                <span class="hi-ticker">${h.ticker}</span>
                <span class="hi-detail">$${h.lastClose?.toFixed(2)} → $${h.forecast?.toFixed(2)} · ${h.days}d</span>
                <span class="hi-change" style="color:${color}">${arrow} ${isUp ? '+' : ''}${h.change}%</span>
                <span class="hi-date">${dateStr}</span>
            </div>
        `;
    }).join('');

    historyItems.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
            tickerInput.value = item.dataset.ticker;
            fetchPrediction(item.dataset.ticker);
        });
    });
}

// ══════════════════════════════════════════════════════════════════════
//  TOAST NOTIFICATIONS
// ══════════════════════════════════════════════════════════════════════
const TOAST_ICONS = { success: '✅', error: '❌', info: 'ℹ️' };

function toast(type, message) {
    const container = $('toastContainer');
    const el = document.createElement('div');
    el.className = 'toast';
    el.innerHTML = `
        <span class="toast-icon">${TOAST_ICONS[type] || 'ℹ️'}</span>
        <span class="toast-msg">${message}</span>
    `;
    container.appendChild(el);

    setTimeout(() => {
        el.classList.add('fade-out');
        el.addEventListener('animationend', () => el.remove());
    }, 3000);
}

// ══════════════════════════════════════════════════════════════════════
//  UTILITIES
// ══════════════════════════════════════════════════════════════════════
function formatLargeNum(n) {
    if (n == null) return '—';
    if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
    if (n >= 1e9)  return `$${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e6)  return `$${(n / 1e6).toFixed(2)}M`;
    if (n >= 1e3)  return `${(n / 1e3).toFixed(1)}K`;
    return n.toLocaleString();
}

function reflow(el) {
    el.style.animation = 'none';
    el.offsetHeight;
    el.style.animation = '';
}

function resetUI() {
    errorMsg.classList.add('hidden');
    statsBar.classList.add('hidden');
    chartContainer.classList.add('hidden');
    stockInfoEl.classList.add('hidden');
    metricsCard.classList.add('hidden');
    errorMsg.textContent = '';
}

function showLoading(state) {
    loading.classList.toggle('hidden', !state);
}

function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.remove('hidden');
    toast('error', msg);
}
