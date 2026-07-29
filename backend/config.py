"""Application configuration with environment variable support (4.2)."""

import tomllib
from ipaddress import ip_address
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
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
    """Supported, executable walk-forward validation settings."""

    method: Literal["expanding", "rolling"] = "expanding"
    folds: int = Field(default=5, ge=1, le=20)
    min_train_size: int = Field(default=300, ge=100)
    horizon: int = Field(default=60, ge=30)
    gap: int = Field(default=0, ge=0)
    seed: int = 42
    deterministic: bool = True

    @model_validator(mode="after")
    def validate_strategy(self):
        if self.method == "rolling" and self.min_train_size < 100:
            raise ValueError("rolling validation requires min_train_size >= 100")
        return self


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
    info_cache_ttl: int = 3600
    cache_max_size: int = 256
    model_type: str = "bilstm_attention_direction"
    validation: ValidationConfig = ValidationConfig()
    prediction_workers: int = Field(default=2, ge=1, le=8)
    prediction_queue_size: int = Field(default=4, ge=0, le=64)
    prediction_timeout_seconds: int = Field(default=600, ge=30, le=3600)
    upstream_circuit_cooldown_seconds: int = Field(default=30, ge=1, le=300)
    training_concurrency: int = Field(default=1, ge=1, le=4)
    training_wait_seconds: int = Field(default=30, ge=1, le=300)
    model_max_count: int = Field(default=20, ge=1, le=500)
    model_max_storage_mb: int = Field(default=900, ge=50)
    model_min_free_mb: int = Field(default=100, ge=10)
    model_versions_to_keep: int = Field(default=2, ge=1, le=10)
    artifact_lock_timeout_seconds: int = Field(default=900, ge=30, le=3600)
    trusted_proxy_ips: list[str] = Field(default_factory=list)

    @field_validator("trusted_proxy_ips")
    @classmethod
    def validate_trusted_proxy_ips(cls, values: list[str]) -> list[str]:
        """Accept exact proxy addresses only; broad networks and wildcards are unsafe."""
        normalised: list[str] = []
        for value in values:
            if "/" in value or value.strip() == "*":
                raise ValueError("trusted_proxy_ips entries must be exact IP addresses")
            normalised.append(str(ip_address(value.strip())))
        return list(dict.fromkeys(normalised))

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_nested_delimiter": "__",
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
SCHEMA_VERSION = 3

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
