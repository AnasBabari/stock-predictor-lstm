"""Application configuration with environment variable support (4.2)."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All settings can be overridden via environment variables or a .env file."""

    default_ticker: str = "AAPL"
    historical_years: int = 3
    window_size: int = 60
    train_split: float = 0.80
    lstm_units: int = 64
    epochs: int = 25
    batch_size: int = 32
    model_dir: str = "saved_models"
    model_max_age_days: int = 7
    default_forecast_days: int = 7
    max_forecast_days: int = 30
    allowed_origins: list[str] = [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ]
    cache_ttl: int = 300
    cache_max_size: int = 256

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

# Backward-compatible module-level constants
DEFAULT_TICKER = settings.default_ticker
HISTORICAL_YEARS = settings.historical_years
WINDOW_SIZE = settings.window_size
TRAIN_SPLIT = settings.train_split
LSTM_UNITS = settings.lstm_units
EPOCHS = settings.epochs
BATCH_SIZE = settings.batch_size
MODEL_DIR = settings.model_dir
MODEL_MAX_AGE_DAYS = settings.model_max_age_days
DEFAULT_FORECAST_DAYS = settings.default_forecast_days
MAX_FORECAST_DAYS = settings.max_forecast_days