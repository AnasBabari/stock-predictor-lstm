#type: ignore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import timedelta
from config import HISTORICAL_YEARS
from data_pipeline import get_pipeline
from model import load_or_train, predict_future

# Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/predict")
def predict(ticker: str = "AAPL"):
    ticker = ticker.upper()
    try:
        # Fetch data & preprocess
        pipeline_data, closing_prices = get_pipeline(ticker)
        X_train, X_test, y_train, y_test, scaler = pipeline_data

        # Load existing model or train a new one
        model = load_or_train(ticker, X_train, y_train)

        # Predict the next 7 days
        predictions = predict_future(model, closing_prices, scaler, days=7)

        # Generate Dates for the Chart
        raw_data = yf.download(ticker, period=f"{HISTORICAL_YEARS}y", progress=False)
        raw_data = raw_data.dropna()
        historical_dates = raw_data.index.strftime('%Y-%m-%d').tolist()
        
        # Calculate the next 7 days (skipping weekends)
        future_dates = []
        current_date = raw_data.index[-1]
        days_added = 0
        while days_added < 7:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:  # 0-4 are Monday-Friday
                future_dates.append(current_date.strftime('%Y-%m-%d'))
                days_added += 1

        # Clean up numpy arrays into standard Python lists for JSON format
        historical_prices = [float(p[0]) for p in closing_prices]

        
        return {
            "ticker": ticker,
            "historical_dates": historical_dates,
            "historical_prices": historical_prices,
            "future_dates": future_dates,
            "predicted_prices": [float(p) for p in predictions]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))