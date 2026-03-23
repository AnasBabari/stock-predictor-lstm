//Fetches predictions and renders graph

const tickerInput = document.getElementById('tickerInput');
const predictBtn = document.getElementById('predictBtn');
const errorMsg = document.getElementById('errorMsg');
const loading = document.getElementById('loading');
const statsBar = document.getElementById('statsBar');
const chartContainer = document.getElementById('chartContainer');
const chartTitle = document.getElementById('chartTitle');

let stockChart = null;

// ── Event Listener ────────────────────────────────
predictBtn.addEventListener('click', () => {
    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) {
        showError('Please enter a ticker symbol.');
        return;
    }
    fetchPrediction(ticker);
});

tickerInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') predictBtn.click();
});

// ── Fetch Prediction ──────────────────────────────
async function fetchPrediction(ticker) {
    resetUI();
    showLoading(true);

    try {
        const response = await fetch(`http://127.0.0.1:8000/api/v1/predict?ticker=${ticker}`);
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        const data = await response.json();
        renderStats(data);
        renderChart(data);
    } catch (err) {
        showError(`Failed to fetch prediction.`);
    } finally {
        showLoading(false);
    }
}

// ── Render Stats Bar ──────────────────────────────
function renderStats(data) {
    const lastClose = data.historical_prices.at(-1).toFixed(2);
    const forecast = data.predicted_prices[0].toFixed(2);
    const trend = forecast > lastClose ? '▲ Bullish' : '▼ Bearish';
    const trendEl = document.getElementById('statTrend');

    document.getElementById('statTicker').textContent = data.ticker;
    document.getElementById('statLastClose').textContent = `$${lastClose}`;
    document.getElementById('statForecast').textContent = `$${forecast}`;
    trendEl.textContent = trend;
    trendEl.style.color = forecast > lastClose ? '#00f5a0' : '#ff4d6d';

    statsBar.classList.remove('hidden');
}

// ── Render Chart ──────────────────────────────────
function renderChart(data) {
    chartTitle.textContent = `${data.ticker} — Historical vs Predicted Closing Price`;
    chartContainer.classList.remove('hidden');

    const allDates = [...data.historical_dates, ...data.future_dates];
    const historicalPadded = [
        ...data.historical_prices,
        ...Array(data.future_dates.length).fill(null)
    ];
    const predictedPadded = [
        ...Array(data.historical_prices.length - 1).fill(null),
        data.historical_prices.at(-1), // connect lines at last historical point
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
                }
            ]
        },
        options: {
            responsive: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    labels: { color: '#888899', font: { size: 12 } }
                },
                tooltip: {
                    backgroundColor: '#0f0f1a',
                    titleColor: '#e0e0e0',
                    bodyColor: '#888899',
                    borderColor: '#1e1e2e',
                    borderWidth: 1,
                    callbacks: {
                        label: (ctx) => ` $${ctx.parsed.y?.toFixed(2) ?? '—'}`
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#555577', maxTicksLimit: 10 },
                    grid: { color: '#1e1e2e' }
                },
                y: {
                    ticks: {
                        color: '#555577',
                        callback: (val) => `$${val.toFixed(0)}`
                    },
                    grid: { color: '#1e1e2e' }
                }
            }
        }
    });
}

// ── UI Helpers ────────────────────────────────────
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