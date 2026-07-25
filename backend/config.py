"""Application configuration with environment variable support (4.2)."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def get_app_version() -> str:
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                return data.get("project", {}).get("version", "1.1.0")
        except Exception:
            pass
    return "1.1.0"


APP_VERSION = get_app_version()


class ValidationConfig(BaseModel):
    """Configurable walk-forward validation strategy.

    Supports 'expanding', 'rolling', and 'anchored' methods so future
    strategy comparisons require no code changes — only config changes.
    """

    method: str = "expanding"  # "expanding" | "rolling" | "anchored"
    folds: int = 5
    min_train_size: int = 500  # minimum rows required in any training fold
    horizon: int = 30  # validation window size per fold (days)
    seed: int = 42  # global random seed for reproducibility
    test_size: float = 0.2


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

    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ]
    )

    cors_origin: str | None = None

    cache_ttl: int = 300
    cache_max_size: int = 256
    model_type: str = "bilstm_attention_direction"
    validation: ValidationConfig = ValidationConfig()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
if settings.cors_origin:
    settings.allowed_origins.append(settings.cors_origin)

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
MODEL_TYPE = settings.model_type
VALIDATION_CONFIG = settings.validation
VALIDATION_SEED = settings.validation.seed

# Feature Schema Versioning & Centralized Config
SCHEMA_VERSION = 2

FEATURE_CONFIG = {
    "base": ["Open", "High", "Low", "Close", "Volume"],
    "technical": [
        "SMA_20",
        "EMA_20",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "BB_Upper",
        "BB_Lower",
        "ATR_14",
        "OBV",
    ],
    "market": ["SPY_Return_1D", "QQQ_Return_1D", "VIX_Return_1D", "TNX_Return_1D"],
    "calendar": ["Month_Sin", "Month_Cos", "Day_Sin", "Day_Cos"],
}

FEATURES: list[str] = (
    FEATURE_CONFIG["base"]
    + FEATURE_CONFIG["technical"]
    + FEATURE_CONFIG["market"]
    + FEATURE_CONFIG["calendar"]
)
