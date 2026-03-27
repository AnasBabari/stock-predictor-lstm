/* ===================================================================
   StockLSTM — Frontend Application v3
   =================================================================== */

// 5.1 Configurable API_BASE
const API_BASE = window.STOCKLSTM_API_BASE || 'http://127.0.0.1:8000';

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

// Provide safe HTML escaping to prevent XSS (1.2)
function escapeHtml(str) {
    if (str == null) return '';
    return String(str).replace(/[&<>'"]/g, tag => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    }[tag] || tag));
}

// Apply persisted theme on load
document.documentElement.setAttribute('data-theme', currentTheme);
themeToggle.querySelector('.theme-icon').textContent = currentTheme === 'dark' ? '🌙' : '☀️';

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
    themeToggle.querySelector('.theme-icon').textContent = currentTheme === 'dark' ? '🌙' : '☀️';
    localStorage.setItem('stocklstm-theme', currentTheme);
    if (currentStockData) renderChart(currentStockData);
});

// ── Predict Button ──────────────────────────────────────────────────
predictBtn.addEventListener('click', () => {
    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) { showError('Please enter a ticker symbol.'); return; }
    fetchPrediction(ticker);
});

// ── Search Autocomplete & Keyboard Nav (5.3) ───────────────────────
let searchTimeout = null;
let highlightedIndex = -1;

tickerInput.addEventListener('input', () => {
    const q = tickerInput.value.trim();
    clearTimeout(searchTimeout);
    if (q.length < 2) { closeDropdown(); return; }
    searchTimeout = setTimeout(() => fetchSuggestions(q), 250);
});

tickerInput.addEventListener('keydown', (e) => {
    const items = searchDropdown.querySelectorAll('.dropdown-item');
    if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (!searchDropdown.classList.contains('hidden') && items.length > 0) {
            highlightedIndex = Math.min(items.length - 1, highlightedIndex + 1);
            updateHighlight();
        }
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (!searchDropdown.classList.contains('hidden') && items.length > 0) {
            highlightedIndex = Math.max(-1, highlightedIndex - 1);
            updateHighlight();
        }
    } else if (e.key === 'Enter') {
        e.preventDefault();
        if (highlightedIndex >= 0 && items.length > highlightedIndex) {
            items[highlightedIndex].click();
        } else {
            predictBtn.click();
        }
    } else if (e.key === 'Escape') {
        closeDropdown();
    }
});

function updateHighlight() {
    const items = searchDropdown.querySelectorAll('.dropdown-item');
    items.forEach((item, i) => {
        if (i === highlightedIndex) {
            item.classList.add('highlighted');
            item.style.background = 'var(--accent-glow-subtle)'; // simple inline highlight
            tickerInput.setAttribute('aria-activedescendant', 'dropdown-item-' + i);
        } else {
            item.classList.remove('highlighted');
            item.style.background = '';
        }
    });
}

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
    
    // XSS Fix: using manual DOM structure creation
    searchDropdown.innerHTML = '';
    const fragment = document.createDocumentFragment();
    results.forEach((r, i) => {
        const item = document.createElement('div');
        item.className = 'dropdown-item';
        item.id = `dropdown-item-${i}`;
        item.dataset.ticker = r.ticker;
        item.setAttribute('role', 'option');

        const leftDiv = document.createElement('div');
        const nameDiv = document.createElement('div');
        nameDiv.className = 'dropdown-name';
        nameDiv.textContent = r.name;
        const typeDiv = document.createElement('div');
        typeDiv.className = 'dropdown-type';
        typeDiv.textContent = r.type;
        leftDiv.appendChild(nameDiv);
        leftDiv.appendChild(typeDiv);

        const rightDiv = document.createElement('div');
        rightDiv.className = 'dropdown-ticker';
        rightDiv.textContent = r.ticker;

        item.appendChild(leftDiv);
        item.appendChild(rightDiv);
        fragment.appendChild(item);
    });
    searchDropdown.appendChild(fragment);

    searchDropdown.querySelectorAll('.dropdown-item').forEach(item => {
        item.addEventListener('click', () => {
            tickerInput.value = item.dataset.ticker;
            closeDropdown();
            fetchPrediction(item.dataset.ticker);
        });
    });
    
    searchDropdown.classList.remove('hidden');
    tickerInput.setAttribute('aria-expanded', 'true');
    highlightedIndex = -1;
}

function closeDropdown() {
    searchDropdown.classList.add('hidden');
    searchDropdown.innerHTML = '';
    tickerInput.setAttribute('aria-expanded', 'false');
    tickerInput.setAttribute('aria-activedescendant', '');
    highlightedIndex = -1;
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

// ── Fetch Prediction (5.2 AbortController added) ────────────────────
let currentController = null;
let lastInfo = null; // Stored to accurately populate watchlist name (5.5)

async function fetchPrediction(ticker) {
    if (currentController) currentController.abort();
    currentController = new AbortController();
    const { signal } = currentController;
    
    resetUI();
    showLoading(true);
    predictBtn.disabled = true;

    const days = parseInt(forecastDays.value);

    try {
        // Fetch prediction + stock info in parallel
        const [predRes, infoRes] = await Promise.allSettled([
            fetch(`${API_BASE}/api/v1/predict?ticker=${ticker}&days=${days}`, { signal }),
            fetch(`${API_BASE}/api/v1/info?ticker=${ticker}`, { signal }),
        ]);

        // Handle prediction
        if (predRes.status === 'fulfilled' && predRes.value.ok) {
            currentStockData = await predRes.value.json();
            renderStats(currentStockData);
            renderChart(currentStockData);
            renderMetrics(currentStockData);
            addToHistory(currentStockData);
            toast('success', `Forecast ready for ${escapeHtml(currentStockData.ticker)}`);
        } else if (predRes.status === 'fulfilled') {
            const errData = await predRes.value.json().catch(() => ({}));
            throw new Error(errData.detail || `Prediction failed (${predRes.value.status})`);
        } else if (predRes.reason.name === 'AbortError') {
            return; // Silently ignore aborts
        } else {
            throw new Error('Network error. Failed to fetch prediction.');
        }

        // Handle stock info
        if (infoRes.status === 'fulfilled' && infoRes.value.ok) {
            lastInfo = await infoRes.value.json();
            renderStockInfo(lastInfo);
        } else {
            lastInfo = null;
        }

    } catch (err) {
        if (err.name === 'AbortError') return;
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
    const icons = {
        'Company':   '<svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path fill-rule="evenodd" d="M4 4a2 2 0 012-2h8a2 2 0 012 2v12a1 1 0 110 2h-3a1 1 0 01-1-1v-2a1 1 0 00-1-1H9a1 1 0 00-1 1v2a1 1 0 01-1 1H4a1 1 0 110-2V4zm3 1h2v2H7V5zm2 4H7v2h2V9zm2-4h2v2h-2V5zm2 4h-2v2h2V9z"/></svg>',
        'Sector':    '<svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path fill-rule="evenodd" d="M2 5a2 2 0 012-2h8a2 2 0 012 2v10a2 2 0 002 2H4a2 2 0 01-2-2V5zm3 1h6v4H5V6zm6 6H5v2h6v-2z"/><path d="M15 7h1a2 2 0 012 2v5.5a1.5 1.5 0 01-3 0V7z"/></svg>',
        'Market Cap':'<svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path d="M8.433 7.418c.155-.103.346-.196.567-.267v1.698a2.305 2.305 0 01-.567-.267C8.07 8.34 8 8.114 8 8c0-.114.07-.34.433-.582zM11 12.849v-1.698c.22.071.412.164.567.267.364.243.433.468.433.582 0 .114-.07.34-.433.582a2.305 2.305 0 01-.567.267z"/><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-13a1 1 0 10-2 0v.092a4.535 4.535 0 00-1.676.662C6.602 6.234 6 7.009 6 8c0 .99.602 1.765 1.324 2.246.48.32 1.054.545 1.676.662v1.941c-.391-.127-.68-.317-.843-.504a1 1 0 10-1.51 1.31c.562.649 1.413 1.076 2.353 1.253V15a1 1 0 102 0v-.092a4.535 4.535 0 001.676-.662C13.398 13.766 14 12.991 14 12c0-.99-.602-1.765-1.324-2.246A4.535 4.535 0 0011 9.092V7.151c.391.127.68.317.843.504a1 1 0 101.511-1.31c-.563-.649-1.413-1.076-2.354-1.253V5z"/></svg>',
        'P/E Ratio': '<svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path fill-rule="evenodd" d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 0l-2 2a1 1 0 101.414 1.414L8 10.414l1.293 1.293a1 1 0 001.414 0l4-4z"/></svg>',
        '52W High':  '<svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path fill-rule="evenodd" d="M12 7a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0V8.414l-4.293 4.293a1 1 0 01-1.414 0L8 10.414l-4.293 4.293a1 1 0 01-1.414-1.414l5-5a1 1 0 011.414 0L11 10.586 14.586 7H12z"/></svg>',
        '52W Low':   '<svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path fill-rule="evenodd" d="M12 13a1 1 0 100 2h5a1 1 0 001-1V9a1 1 0 10-2 0v2.586l-4.293-4.293a1 1 0 00-1.414 0L8 9.586 3.707 5.293a1 1 0 00-1.414 1.414l5 5a1 1 0 001.414 0L11 9.414 14.586 13H12z"/></svg>',
        'Avg Volume':'<svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z"/></svg>',
        'Prev Close':'<svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z"/></svg>',
    };

    const cards = [
        { label: 'Company', value: escapeHtml(info.name) || '—', mono: false },
        { label: 'Sector', value: escapeHtml(info.sector) || '—', mono: false },
        { label: 'Market Cap', value: formatLargeNum(info.marketCap), mono: true },
        { label: 'P/E Ratio', value: info.peRatio ? info.peRatio.toFixed(2) : '—', mono: true },
        { label: '52W High', value: info.fiftyTwoWeekHigh ? `$${info.fiftyTwoWeekHigh.toFixed(2)}` : '—', mono: true },
        { label: '52W Low', value: info.fiftyTwoWeekLow ? `$${info.fiftyTwoWeekLow.toFixed(2)}` : '—', mono: true },
        { label: 'Avg Volume', value: formatLargeNum(info.avgVolume), mono: true },
        { label: 'Prev Close', value: info.previousClose ? `$${info.previousClose.toFixed(2)}` : '—', mono: true },
    ];

    infoGrid.innerHTML = cards.map((c, i) => `
        <div class="info-card" style="animation: fadeUp 0.3s ${0.05 * i}s ease both">
            <span class="info-card-label">${icons[c.label] || ''} ${c.label}</span>
            <span class="info-card-value${c.mono ? ' mono' : ''}">${c.value}</span>
        </div>
    `).join('');

    stockInfoEl.classList.remove('hidden');
}

// ── Render Stats Bar ────────────────────────────────────────────────
function renderStats(data) {
    const lastClose = data.historical_prices.at(-1);
    // 5.8 Ensure we show end-of-forecast price
    const forecast  = data.predicted_prices.at(-1);
    const isUp      = forecast > lastClose;
    const change    = forecast - lastClose;
    const changePct = ((change / lastClose) * 100).toFixed(2);

    const changeEl = $('statChange');
    const trendEl  = $('statTrend');

    $('statTicker').textContent    = escapeHtml(data.ticker);
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
    chartTitle.textContent = `${escapeHtml(data.ticker)} — Historical vs Predicted`;
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
                    labels: { color: legendColor, font: { size: 12, family: 'Inter' }, usePointStyle: true, padding: 20 },
                },
                tooltip: {
                    backgroundColor: tooltipBg, titleColor: tooltipTitle, bodyColor: tooltipBody,
                    borderColor: tooltipBorder, borderWidth: 1, cornerRadius: 10, padding: 12,
                    titleFont: { family: 'Inter', weight: '600' }, bodyFont: { family: 'Inter' },
                    callbacks: {
                        label: (ctx) => {
                            const val = ctx.parsed.y;
                            return val != null ? ` $${val.toFixed(2)}` : null;
                        },
                    },
                },
            },
            scales: {
                x: { ticks: { color: tickColor, maxTicksLimit: 10, font: { family: 'Inter', size: 11 } }, grid: { color: gridColor } },
                y: { ticks: { color: tickColor, font: { family: 'Inter', size: 11 }, callback: (v) => `$${v.toFixed(0)}` }, grid: { color: gridColor } },
            },
        },
    });

    reflow(chartContainer);
}

// ── Render Metrics ──────────────────────────────────────────────────
function renderMetrics(data) {
    const m = data.metrics;
    
    // Support rendering full list of model metrics if available
    $('metricRMSE').textContent = m && m.rmse != null ? m.rmse.toFixed(2) : '—';
    $('metricMAE').textContent  = m && m.mae != null  ? m.mae.toFixed(2)  : '—';
    
    const metricR2 = $('metricR2');
    if (metricR2) metricR2.textContent = m && m.r2 != null ? m.r2.toFixed(4) : '—';
    
    const metricMAPE = $('metricMAPE');
    const metricDA = $('metricDA');
    if (metricMAPE) metricMAPE.textContent = m && m.mape != null ? `${m.mape.toFixed(2)}%` : '—';
    if (metricDA) metricDA.textContent = m && m.directional_accuracy != null ? `${(m.directional_accuracy * 100).toFixed(1)}%` : '—';
    
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

    d.historical_dates.forEach((dt, i) => csv += `${dt},${d.historical_prices[i].toFixed(2)},Historical\n`);
    d.future_dates.forEach((dt, i) => csv += `${dt},${d.predicted_prices[i].toFixed(2)},Predicted\n`);

    const blob = new Blob([csv], { type: 'text/csv' });
    const link = document.createElement('a');
    link.download = `${d.ticker}_forecast.csv`;
    link.href = URL.createObjectURL(blob);
    link.click();
    // 5.4 Object URL Revoked Before Download Completed
    setTimeout(() => URL.revokeObjectURL(link.href), 500);
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
    try {
        localStorage.setItem(WL_KEY, JSON.stringify(list));
    } catch (e) {
        toast('error', 'Storage full. Please clear space to save your watchlist.');
    }
}

$('addWatchlist').addEventListener('click', () => {
    if (!currentStockData) return;
    const d = currentStockData;
    let list = getWatchlist();

    if (list.find(w => w.ticker === d.ticker)) {
        toast('info', `${escapeHtml(d.ticker)} is already in your watchlist`);
        return;
    }

    // 5.5 Watchlist name retrieved via reliable API response instead of fragile DOM element
    const safeName = lastInfo && lastInfo.ticker === d.ticker ? lastInfo.name : '';

    list.unshift({
        ticker: d.ticker,
        name: safeName,
        lastPrice: d.historical_prices.at(-1),
    });

    saveWatchlist(list);
    renderWatchlist();
    toast('success', `${escapeHtml(d.ticker)} added to watchlist`);
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

    watchlistItems.innerHTML = '';
    const fragment = document.createDocumentFragment();
    list.forEach((w, i) => {
        const item = document.createElement('div');
        item.className = 'watchlist-item';
        item.dataset.ticker = w.ticker;
        item.dataset.index = i;

        const tSpan = document.createElement('span');
        tSpan.className = 'wl-ticker';
        tSpan.textContent = w.ticker;

        const nSpan = document.createElement('span');
        nSpan.className = 'wl-name';
        nSpan.textContent = w.name || '';

        const pSpan = document.createElement('span');
        pSpan.className = 'wl-price';
        pSpan.textContent = w.lastPrice ? `$${w.lastPrice.toFixed(2)}` : '—';

        const removeBtn = document.createElement('button');
        removeBtn.className = 'wl-remove';
        removeBtn.dataset.index = i;
        removeBtn.title = 'Remove';
        removeBtn.textContent = '✕';

        item.appendChild(tSpan);
        item.appendChild(nSpan);
        item.appendChild(pSpan);
        item.appendChild(removeBtn);
        fragment.appendChild(item);
    });
    watchlistItems.appendChild(fragment);

    watchlistItems.querySelectorAll('.watchlist-item').forEach(item => {
        item.addEventListener('click', (e) => {
            if (e.target.classList.contains('wl-remove')) return;
            tickerInput.value = item.dataset.ticker;
            fetchPrediction(item.dataset.ticker);
        });
    });

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
    try {
        localStorage.setItem(HIST_KEY, JSON.stringify(list));
    } catch (e) {
        toast('error', 'Storage full. Cannot update history.');
    }
}

function addToHistory(data) {
    const lastClose = data.historical_prices.at(-1);
    const forecast  = data.predicted_prices.at(-1);
    const change    = ((forecast - lastClose) / lastClose * 100).toFixed(2);

    let list = getHistory();
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

    historyItems.innerHTML = '';
    const fragment = document.createDocumentFragment();
    list.forEach(h => {
        const isUp = parseFloat(h.change) >= 0;
        const color = isUp ? 'var(--bullish)' : 'var(--bearish)';
        const arrow = isUp ? '▲' : '▼';
        const dateStr = new Date(h.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

        const item = document.createElement('div');
        item.className = 'history-item';
        item.dataset.ticker = h.ticker;

        const tSpan = document.createElement('span');
        tSpan.className = 'hi-ticker';
        tSpan.textContent = h.ticker;

        const dSpan = document.createElement('span');
        dSpan.className = 'hi-detail';
        dSpan.textContent = `$${h.lastClose?.toFixed(2)} → $${h.forecast?.toFixed(2)} · ${h.days}d`;

        const cSpan = document.createElement('span');
        cSpan.className = 'hi-change';
        cSpan.style.color = color;
        cSpan.textContent = `${arrow} ${isUp ? '+' : ''}${h.change}%`;

        const dateSpan = document.createElement('span');
        dateSpan.className = 'hi-date';
        dateSpan.textContent = dateStr;

        item.appendChild(tSpan);
        item.appendChild(dSpan);
        item.appendChild(cSpan);
        item.appendChild(dateSpan);
        fragment.appendChild(item);
    });
    historyItems.appendChild(fragment);

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
    
    const iconSpan = document.createElement('span');
    iconSpan.className = 'toast-icon';
    iconSpan.textContent = TOAST_ICONS[type] || 'ℹ️';

    const msgSpan = document.createElement('span');
    msgSpan.className = 'toast-msg';
    msgSpan.textContent = message;

    el.appendChild(iconSpan);
    el.appendChild(msgSpan);
    
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
    errorMsg.textContent = msg; // TextContent implicitly escapes HTML
    errorMsg.classList.remove('hidden');
    toast('error', msg);
}
