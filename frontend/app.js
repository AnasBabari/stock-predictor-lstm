
const tickerInput    = document.getElementById('tickerInput');
const predictBtn     = document.getElementById('predictBtn');
const errorMsg       = document.getElementById('errorMsg');
const loading        = document.getElementById('loading');
const statsBar       = document.getElementById('statsBar');
const chartContainer = document.getElementById('chartContainer');
const chartTitle     = document.getElementById('chartTitle');
const themeToggle    = document.getElementById('themeToggle');

let stockChart       = null;
let currentStockData = null;
let currentDaysView  = 21;
let currentTheme     = 'dark';

//Splash screen
window.addEventListener('load', () => {
    const splashScreen = document.getElementById('splashScreen');
    setTimeout(() => {
        splashScreen.classList.add('fade-out');
    }, 1200); 
});

//dark/light mode toggle
themeToggle.addEventListener('click', () => {
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', currentTheme);
    themeToggle.textContent = currentTheme === 'dark' ? '🌙' : '☀️';
    if (currentStockData) renderChart(currentStockData);
});
 
predictBtn.addEventListener('click', () => {
    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) { showError('Please enter a ticker symbol.'); return; }
    fetchPrediction(ticker);
});

tickerInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') predictBtn.click();
});
// Search Dropdown 
const searchDropdown = document.getElementById('searchDropdown');
let searchTimeout = null;

tickerInput.addEventListener('input', () => {
    const query = tickerInput.value.trim();
    clearTimeout(searchTimeout);

    if (query.length < 2) {
        closeDropdown();
        return;
    }

    searchTimeout = setTimeout(() => fetchSuggestions(query), 200);
});

document.addEventListener('click', (e) => {
    if (!e.target.closest('.input-wrapper')) closeDropdown();
});

async function fetchSuggestions(query) {
    try {
        const response = await fetch(`http://127.0.0.1:8000/api/v1/search?query=${encodeURIComponent(query)}`);
        if (!response.ok) return;
        const data = await response.json();
        renderDropdown(data.results);
    } catch (err) {
        console.error('Search failed:', err);
    }
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
            const ticker = item.dataset.ticker;
            tickerInput.value = ticker;
            closeDropdown();
            fetchPrediction(ticker);
        });
    });

    searchDropdown.classList.remove('hidden');
}

function closeDropdown() {
    searchDropdown.classList.add('hidden');
    searchDropdown.innerHTML = '';
}

document.querySelectorAll('.time-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        currentDaysView = parseInt(e.target.dataset.days);
        if (currentStockData) renderChart(currentStockData);
    });
});

//Fetch prediction from backend
async function fetchPrediction(ticker) {
    resetUI();
    showLoading(true);
    predictBtn.disabled = true;

    try {
        const response = await fetch(`http://127.0.0.1:8000/api/v1/predict?ticker=${ticker}`);
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || `Server error: ${response.status}`);
        }
        currentStockData = await response.json();
        renderStats(currentStockData);
        renderChart(currentStockData);
    } catch (err) {
        console.error(err);
        const friendlyError = err.message.includes('Failed to fetch')
            ? 'Could not connect to the backend. Make sure the server is running.'
            : err.message.includes('500')
            ? 'Something went wrong on the server. The ticker may be invalid or unsupported.'
            : err.message.includes('404')
            ? 'Endpoint not found. Check the backend is running correctly.'
            : 'Something went wrong. Please try again with a valid ticker symbol.';
        showError(friendlyError);
    } finally {
        showLoading(false);
        predictBtn.disabled = false;
    }
}

//Render Stats 
function renderStats(data) {
    const lastClose = data.historical_prices.at(-1);
    const forecast  = data.predicted_prices[0];
    const isUp      = forecast > lastClose;
    const change    = forecast - lastClose;
    const changePct = ((change / lastClose) * 100).toFixed(2);

    const changeEl = document.getElementById('statChange');
    const trendEl  = document.getElementById('statTrend');

    document.getElementById('statTicker').textContent    = data.ticker;
    document.getElementById('statLastClose').textContent = `$${lastClose.toFixed(2)}`;
    document.getElementById('statForecast').textContent  = `$${forecast.toFixed(2)}`;

    changeEl.textContent = `${isUp ? '+' : ''}${changePct}%`;
    changeEl.style.color = isUp ? 'var(--accent)' : '#ff4d6d';

    trendEl.textContent  = isUp ? '▲ Bullish' : '▼ Bearish';
    trendEl.style.color  = isUp ? 'var(--accent)' : '#ff4d6d';

    statsBar.classList.remove('hidden');
    statsBar.style.animation = 'none';
    statsBar.offsetHeight;
    statsBar.style.animation = '';
}

//Render Chart 
function renderChart(data) {
    chartTitle.textContent = `${data.ticker} — Historical vs Predicted Closing Price`;
    chartContainer.classList.remove('hidden');

    const isDark       = currentTheme === 'dark';
    const gridColor    = isDark ? '#1e1e2e' : '#e0e4ea';
    const tickColor    = isDark ? '#555577' : '#8888aa';
    const tooltipBg    = isDark ? '#0f0f1a' : '#fffefb';
    const tooltipTitle = isDark ? '#e0e0e0' : '#1a1a2e';
    const tooltipBody  = isDark ? '#888899' : '#555577';
    const tooltipBorder = isDark ? '#1e1e2e' : '#e0dbd0';
    const legendColor  = isDark ? '#888899' : '#555577';

    const totalHistorical = data.historical_prices.length;
    const sliceIndex      = Math.max(0, totalHistorical - currentDaysView);
    const slicedDates     = data.historical_dates.slice(sliceIndex);
    const slicedPrices    = data.historical_prices.slice(sliceIndex);
    const allDates        = [...slicedDates, ...data.future_dates];

    const historicalPadded = [
        ...slicedPrices,
        ...Array(data.future_dates.length).fill(null)
    ];

    const predictedPadded = [
        ...Array(slicedPrices.length - 1).fill(null),
        slicedPrices.at(-1),
        ...data.predicted_prices
    ];

    if (stockChart) stockChart.destroy();

    const ctx = document.getElementById('stockChart').getContext('2d');
    stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allDates,
            datasets: [
                {
                    label: 'Historical Price',
                    data: historicalPadded,
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                    fill: true,
                    spanGaps: false,
                },
                {
                    label: 'Predicted Price',
                    data: predictedPadded,
                    borderColor: '#00f5a0',
                    backgroundColor: 'rgba(0, 245, 160, 0.05)',
                    borderWidth: 2,
                    pointRadius: 4,
                    pointBackgroundColor: '#00f5a0',
                    borderDash: [6, 3],
                    tension: 0.3,
                    fill: true,
                    spanGaps: false,
                }
            ]
        },
        options: {
            responsive: true,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: {
                    labels: { color: legendColor, font: { size: 12 } }
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: tooltipTitle,
                    bodyColor: tooltipBody,
                    borderColor: tooltipBorder,
                    borderWidth: 1,
                    callbacks: {
                        label: (ctx) => {
                            const val = ctx.parsed.y;
                            return val != null ? ` $${val.toFixed(2)}` : null;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: tickColor, maxTicksLimit: 10 },
                    grid:  { color: gridColor }
                },
                y: {
                    ticks: {
                        color: tickColor,
                        callback: (val) => `$${val.toFixed(0)}`
                    },
                    grid: { color: gridColor }
                }
            }
        }
    });

    chartContainer.style.animation = 'none';
    chartContainer.offsetHeight;
    chartContainer.style.animation = '';
}

function resetUI() {
    errorMsg.classList.add('hidden');
    statsBar.classList.add('hidden');
    chartContainer.classList.add('hidden');
    errorMsg.textContent = '';
}

function showLoading(state) {
    loading.classList.toggle('hidden', !state);
}

function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.remove('hidden');
}
