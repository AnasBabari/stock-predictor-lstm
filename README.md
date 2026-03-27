# StockLSTM

StockLSTM is a full-stack stock forecasting app that uses an LSTM neural network to predict the next 3–30 trading days of closing prices and visualize results in an interactive chart.

## Preview

![Live App Preview Top](assets/screenshot-top.png)
*Modern glassmorphism interface with ambient background orbs*

![Live App Preview Bottom](assets/screenshot-bottom.png)
*Interactive chart, detailed stock info, metrics, and prediction history*

## Features

### Core
- FastAPI backend for model inference
- Two-layer LSTM model trained on historical close prices from Yahoo Finance (`yfinance`)
- Configurable forecast horizon: **3, 7, 14, or 30 days**
- Saved model cache per ticker with **automatic staleness detection** (retrains after 7 days)
- **EarlyStopping** callback to prevent overfitting
- **Model evaluation metrics** (RMSE, MAE) returned with every prediction

### Frontend
- Interactive Chart.js line chart with gradient fills
- Timeframe filters: `1W`, `1M`, `3M`, `6M`, `1Y`
- Dark / light theme toggle (persisted in localStorage)
- Company-name search with live ticker autocomplete
- **Stock info dashboard** — market cap, P/E ratio, 52-week range, volume, sector
- **Watchlist** — save favourite tickers (localStorage), one-click re-predict
- **Prediction history** — log of recent forecasts with change % and date
- **Export** — download chart as PNG or data as CSV
- **Toast notifications** for success, error, and info events
- Premium glassmorphism design with Inter font

### API
- `GET /api/v1/predict?ticker=AAPL&days=7` — forecast with configurable horizon
- `GET /api/v1/search?query=apple` — ticker autocomplete
- `GET /api/v1/info?ticker=AAPL` — rich stock metadata
- **In-memory caching** (5 min TTL) for predictions and info

## Project Structure

```text
stock-predictor-lstm/
	backend/
		api.py              # FastAPI app (3 endpoints + caching)
		config.py           # Hyperparameters & settings
		data_pipeline.py    # Data fetching & preprocessing
		model.py            # LSTM build, train, evaluate, predict
		requirements.txt
		saved_models/
	frontend/
		index.html          # Single-page layout
		app.js              # Application logic
		styles.css          # Design system
	assets/
		screenshot.png
	README.md
```

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **ML:** TensorFlow/Keras (2-layer LSTM), scikit-learn
- **Data:** yfinance, NumPy, Pandas
- **Frontend:** HTML, CSS (Inter font, CSS variables), Vanilla JavaScript, Chart.js

## How It Works

1. Historical stock data is downloaded from Yahoo Finance.
2. Close prices are scaled with `MinMaxScaler`.
3. Sliding windows of 60 timesteps are created for training.
4. A per-ticker LSTM model is loaded from disk (if fresh) or trained with EarlyStopping.
5. The model is evaluated on the test set (RMSE, MAE are returned to the frontend).
6. The backend predicts the requested number of future values recursively.
7. The frontend displays historical data, forecasted points, stock info, and metrics.

## Local Setup

### 1. Clone Repository

```bash
git clone https://github.com/AnasBabari/stock-predictor-lstm.git
cd stock-predictor-lstm
```

### 2. Create and Activate Virtual Environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Run Backend API

```bash
cd backend
uvicorn api:app --reload
```

Backend URL: `http://127.0.0.1:8000`

Interactive docs: `http://127.0.0.1:8000/docs`

### 5. Run Frontend

Use a lightweight local server from the project root:

```bash
python -m http.server 5500
```

Then open: `http://127.0.0.1:5500/frontend/`

## API Reference

### `GET /api/v1/predict`

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `ticker` | string | `AAPL` | Stock ticker symbol |
| `days` | int | `7` | Forecast horizon (1–30) |

Returns historical dates/prices, predicted prices, and model metrics.

### `GET /api/v1/search`

| Param | Type | Description |
|-------|------|-------------|
| `query` | string | Search term (ticker or company name) |

Returns up to 8 matching EQUITY/ETF results.

### `GET /api/v1/info`

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `ticker` | string | `AAPL` | Stock ticker symbol |

Returns name, sector, market cap, P/E, 52-week range, volume, etc.

## Configuration

Tune training/inference settings in `backend/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `HISTORICAL_YEARS` | 3 | Years of training data |
| `WINDOW_SIZE` | 60 | Sliding window timesteps |
| `TRAIN_SPLIT` | 0.80 | Train/test split ratio |
| `LSTM_UNITS` | 64 | Units in first LSTM layer |
| `EPOCHS` | 25 | Max training epochs |
| `BATCH_SIZE` | 32 | Training batch size |
| `MODEL_MAX_AGE_DAYS` | 7 | Retrain cached models older than this |
| `DEFAULT_FORECAST_DAYS` | 7 | Default forecast horizon |
| `MAX_FORECAST_DAYS` | 30 | Maximum allowed forecast days |

## Notes and Limitations

- This project is for educational and experimentation purposes only.
- Forecasts are based on historical price patterns and are not financial advice.
- Initial request for a new ticker may take longer because model training runs first.
- Models are automatically retrained when they become stale (older than `MODEL_MAX_AGE_DAYS`).

## Troubleshooting

- **"Not enough data for training."**
	Use a ticker with sufficient historical data and keep `WINDOW_SIZE` reasonable.
- **Slow first prediction for a ticker.**
	Expected behavior: model is being trained and then cached.
- **Frontend cannot connect to backend.**
	Ensure backend is running on `http://127.0.0.1:8000` and CORS is enabled.
- **Old cached model gives poor results.**
	Delete the `.keras` file from `backend/saved_models/` or wait for auto-retrain.

## License

MIT