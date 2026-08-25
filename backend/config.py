"""Application configuration with environment variable support (4.2)."""

import tomllib
from ipaddress import ip_address, ip_network
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
    historical_years: int = 8
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
    preview_cors_origin_regex: str | None = None

    cache_ttl: int = 300
    info_cache_ttl: int = 3600
    cache_max_size: int = 256
    model_type: str = "bilstm_attention_direction"
    validation: ValidationConfig = ValidationConfig()
    prediction_workers: int = Field(default=2, ge=1, le=8)
    prediction_queue_size: int = Field(default=4, ge=0, le=64)
    prediction_timeout_seconds: int = Field(default=600, ge=30, le=3600)
    snapshot_build_wait_seconds: int = Field(default=45, ge=5, le=300)
    upstream_circuit_cooldown_seconds: int = Field(default=30, ge=1, le=300)
    training_concurrency: int = Field(default=1, ge=1, le=4)
    training_wait_seconds: int = Field(default=30, ge=1, le=300)
    model_max_count: int = Field(default=20, ge=1, le=500)
    model_max_storage_mb: int = Field(default=900, ge=50)
    model_min_free_mb: int = Field(default=100, ge=10)
    model_versions_to_keep: int = Field(default=2, ge=1, le=10)
    artifact_lock_timeout_seconds: int = Field(default=900, ge=30, le=3600)
    trusted_proxy_ips: list[str] = Field(default_factory=list)
    deployment_provider: str | None = None
    deployment_environment: str | None = None
    deployment_commit: str | None = None

    # Hybrid server/browser training foundation (defaults keep today's behaviour).
    server_forecast_serving_enabled: bool = False
    server_training_enabled: bool = False
    server_forecast_allowlist: list[str] = Field(default_factory=list)
    server_forecast_max_age_hours: int = Field(default=36, ge=1, le=24 * 30)
    server_forecast_cache_ttl: int = Field(default=900, ge=0, le=86400)
    server_bundle_retention_days: int = Field(default=30, ge=1, le=3650)
    server_forecast_private_key_path: str | None = None
    server_forecast_public_key_path: str | None = None
    registry_database_url: str | None = None

    # Certified global-volatility serving (v2). Absent paths keep the
    # endpoint fail-closed on an explicit no-certified-model state.
    volatility_release_dir: str | None = None
    volatility_public_key_path: str | None = None
    volatility_serving_required: bool = False
    volatility_forecast_cache_ttl: int = Field(default=900, ge=0, le=86400)
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_key_prefix: str = "artifacts"
    training_mode: Literal["browser_only", "hybrid", "server_pretrained"] = "browser_only"

    @field_validator("server_forecast_allowlist", mode="before")
    @classmethod
    def parse_server_forecast_allowlist(cls, value: object) -> object:
        """Accept a comma-separated env string in addition to JSON lists."""
        if isinstance(value, str):
            return [item.strip().upper() for item in value.split(",") if item.strip()]
        return value

    @field_validator("trusted_proxy_ips")
    @classmethod
    def validate_trusted_proxy_ips(cls, values: list[str]) -> list[str]:
        """Accept exact IP addresses or non-wildcard CIDR networks."""
        normalised: list[str] = []
        for value in values:
            val = value.strip()
            if not val or val == "*":
                raise ValueError("Wildcard '*' proxy trust is not permitted.")
            try:
                if "/" in val:
                    net = ip_network(val, strict=False)
                    if net.prefixlen == 0:
                        raise ValueError("Wildcard /0 CIDR is not permitted.")
                    normalised.append(str(net))
                else:
                    addr = ip_address(val)
                    normalised.append(str(addr))
            except ValueError as err:
                raise ValueError(f"Invalid IP address or CIDR: {val}") from err
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
# Legacy offline trainer schema (kept for the server-side research pipelines).
SCHEMA_VERSION = 3

# Browser snapshot schema. Schema v4 replaces absolute price-level features
# with stationary ratios, returns, realized volatility, and market-relative
# measures so models learn movement instead of historical price levels.
SNAPSHOT_SCHEMA_VERSION = 4

# The training target contract for schema-v4 browser snapshots: the model
# predicts cumulative log returns (log close[t+h] / close[t-1]) instead of
# absolute prices, and prices are reconstructed from the latest close.
TARGET_MODE = "cumulative_log_return_v1"

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

# Stationary feature schema for browser-trained models (schema v4). Every
# price-level indicator is replaced by a ratio or return measured against the
# current close or previous close. All rolling transforms are causal.
FEATURE_CONFIG_V4 = {
    "price_returns": [
        "Log_Open_Rel",
        "Log_High_Rel",
        "Log_Low_Rel",
        "Return_1D",
        "Volume_Log1p_Change",
    ],
    "technical_ratio": [
        "Close_SMA_20",
        "Close_EMA_20",
        "RSI_14_Centered",
        "MACD_Close",
        "MACD_Signal_Close",
        "BB_Upper_Rel",
        "BB_Lower_Rel",
        "ATR_14_Rel",
        "OBV_Change_Z",
    ],
    "momentum_vol": [
        "Return_5D",
        "Return_20D",
        "Realized_Vol_5D",
        "Realized_Vol_20D",
    ],
    "market": [
        "SPY_Return_1D",
        "QQQ_Return_1D",
        "VIX_Return_1D",
        "TNX_Return_1D",
        "Return_Rel_SPY_1D",
        "Beta_SPY_20D",
    ],
    "calendar": ["Month_Sin", "Month_Cos", "Day_Sin", "Day_Cos"],
}

FEATURES_V4: list[str] = (
    FEATURE_CONFIG_V4["price_returns"]
    + FEATURE_CONFIG_V4["technical_ratio"]
    + FEATURE_CONFIG_V4["momentum_vol"]
    + FEATURE_CONFIG_V4["market"]
    + FEATURE_CONFIG_V4["calendar"]
)
