"""Frozen contracts for server-pretrained forecast artifacts (schema v4).

Every server artifact is identified by an immutable ``version_id`` of the form
``{ticker}-{forecast_type}-{utc-compact-ts}-{gitsha12}-{snapshot-hash8}`` and
carries reproducibility metadata that mirrors the browser snapshot contract:
schema v4, the 28 ``FEATURES_V4`` columns in exact order, ``TARGET_MODE`` and a
60-step window producing 30 cumulative log-return horizons.
"""

from __future__ import annotations

import math
import re
import subprocess
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config import MAX_FORECAST_DAYS, SNAPSHOT_SCHEMA_VERSION, TARGET_MODE, WINDOW_SIZE

FORECAST_LENGTH = MAX_FORECAST_DAYS  # frozen output length: 30 horizons
HISTORY_DISPLAY_WINDOW = 120  # sessions of close history embedded in each bundle
GIT_SHA_LENGTH = 12
_VERSION_TS_FORMAT = "%Y%m%dT%H%M%SZ"
_VERSION_ID_PATTERN = re.compile(
    r"^[A-Z0-9.-]+-(price|direction)-\d{8}T\d{6}Z-(?:[0-9a-f]{12}|unknown)-(?:[0-9a-f]{8}|unknown)$"
)


def _snapshot_id_short(snapshot_id: str | None) -> str:
    """Collapse a snapshot id (e.g. ``sha256:...``) to a stable 8-char suffix."""
    if not snapshot_id or snapshot_id == "unknown":
        return "unknown"
    hex_chars = re.sub(r"[^0-9a-fA-F]", "", snapshot_id)
    if not hex_chars:
        return "unknown"
    return hex_chars[-8:].lower().zfill(8)


def git_commit_short(length: int = GIT_SHA_LENGTH) -> str:
    """Return the current HEAD commit sha (truncated), or ``"unknown"`` on failure."""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except Exception:
        return "unknown"
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return "unknown"
    return commit[:length]


def make_version_id(
    ticker: str,
    *,
    trained_at: datetime | None = None,
    git_commit: str | None = None,
    forecast_type: Literal["price", "direction"] = "price",
    snapshot_id: str | None = None,
) -> str:
    """Build the immutable artifact identity.

    Format: ``{ticker}-{forecast_type}-{utc-ts}-{gitsha12}-{snapshot_short8}``.
    Including the forecast type and snapshot hash keeps distinct jobs (price vs
    direction, retrains on new data) from ever colliding on the same key.
    """

    moment = trained_at or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    timestamp = moment.astimezone(UTC).strftime(_VERSION_TS_FORMAT)
    commit = (git_commit or git_commit_short())[:GIT_SHA_LENGTH]
    snapshot_short = _snapshot_id_short(snapshot_id)
    return f"{ticker}-{forecast_type}-{timestamp}-{commit}-{snapshot_short}"


class ServerArtifactKey(BaseModel):
    """Immutable identity block for a server-trained artifact."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    forecast_type: Literal["price", "direction"] = "price"
    profile: str = "default"
    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    snapshot_id: str
    trained_at: datetime
    version_id: str

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: int) -> int:
        if value != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be the frozen v{SNAPSHOT_SCHEMA_VERSION}")
        return value

    @field_validator("version_id")
    @classmethod
    def validate_version_id(cls, value: str) -> str:
        if not value or not _VERSION_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "version_id must be '{ticker}-{forecast_type}-{utc-ts}-{gitsha12}-{snapshort8}'"
            )
        return value

    @classmethod
    def create(
        cls,
        *,
        ticker: str,
        snapshot_id: str,
        trained_at: datetime | None = None,
        forecast_type: Literal["price", "direction"] = "price",
        profile: str = "default",
        git_commit: str | None = None,
    ) -> ServerArtifactKey:
        moment = trained_at or datetime.now(UTC)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return cls(
            ticker=ticker,
            forecast_type=forecast_type,
            profile=profile,
            snapshot_id=snapshot_id,
            trained_at=moment,
            version_id=make_version_id(
                ticker,
                trained_at=moment,
                git_commit=git_commit,
                forecast_type=forecast_type,
                snapshot_id=snapshot_id,
            ),
        )


class RobustScalerParams(BaseModel):
    """Robust-scaler parameters (medians/IQRs) fitted on the train slice only."""

    model_config = ConfigDict(frozen=True)

    medians: list[float]
    iqrs: list[float]


class ReproducibilityMetadata(BaseModel):
    """Everything required to reproduce a server forecast from the snapshot."""

    model_config = ConfigDict(frozen=True)

    feature_names: list[str]
    window_size: int = WINDOW_SIZE
    target_mode: str = TARGET_MODE
    scaler: RobustScalerParams
    horizons: list[int] = Field(default_factory=lambda: list(range(1, FORECAST_LENGTH + 1)))
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    python_version: str
    library_versions: dict[str, str] = Field(default_factory=dict)
    git_commit: str

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, value: int) -> int:
        if value != WINDOW_SIZE:
            raise ValueError(f"window_size must be the frozen value {WINDOW_SIZE}")
        return value

    @field_validator("target_mode")
    @classmethod
    def validate_target_mode(cls, value: str) -> str:
        if value != TARGET_MODE:
            raise ValueError(f"target_mode must be the frozen contract '{TARGET_MODE}'")
        return value


class ServerModelRecord(BaseModel):
    """Registry row for one immutable server artifact."""

    key: ServerArtifactKey
    reproducibility: ReproducibilityMetadata
    sha256_digest: str
    signature: str | None = None
    status: Literal["candidate", "promoted", "rejected"] = "candidate"

    @field_validator("sha256_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("sha256_digest must be 64 lowercase hex characters")
        return value


class CompatibilityReport(BaseModel):
    """Outcome of a compatibility check against the frozen browser contract."""

    compatible: bool
    reason: str


class ServerForecastBundle(BaseModel):
    """Signed, precomputed 30-step forecast served read-only by the API.

    The bundle is fully self-contained: it embeds the trailing close-history
    (``historical_dates``/``historical_prices``) so serving never needs to reach
    upstream providers, and every price forecast is derived from the fitted
    complex scaler + model at training time.
    """

    ticker: str
    forecast_type: Literal["price", "direction"] = "price"
    version_id: str
    origin_close: float = Field(gt=0)
    origin_date: date
    future_dates: list[date]
    predicted_log_returns: list[float]
    predicted_prices: list[float]
    historical_dates: list[date]
    historical_prices: list[float]
    evidence: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime

    @field_validator("future_dates", "predicted_log_returns", "predicted_prices")
    @classmethod
    def validate_vector_length(cls, value: list[Any]) -> list[Any]:
        if len(value) != FORECAST_LENGTH:
            raise ValueError(f"forecast vectors must contain exactly {FORECAST_LENGTH} entries")
        return value

    @field_validator("historical_dates", "historical_prices")
    @classmethod
    def validate_history_length(cls, value: list[Any]) -> list[Any]:
        if len(value) != HISTORY_DISPLAY_WINDOW:
            raise ValueError(
                f"historical chart vectors must contain exactly {HISTORY_DISPLAY_WINDOW} entries"
            )
        return value

    @field_validator("predicted_prices", "historical_prices")
    @classmethod
    def validate_prices_finite_and_positive(cls, value: list[float]) -> list[float]:
        for price in value:
            if not math.isfinite(price) or price <= 0:
                raise ValueError("prices must all be finite and strictly positive")
        return value

    @field_validator("predicted_log_returns")
    @classmethod
    def validate_returns_finite(cls, value: list[float]) -> list[float]:
        if any(not math.isfinite(return_) for return_ in value):
            raise ValueError("predicted_log_returns must all be finite")
        return value

    @model_validator(mode="after")
    def validate_temporal_consistency(self) -> ServerForecastBundle:
        future = self.future_dates
        if any(future[i] >= future[i + 1] for i in range(FORECAST_LENGTH - 1)):
            raise ValueError("future_dates must be strictly increasing")
        if future[0] <= self.origin_date:
            raise ValueError("the first future date must be after origin_date")

        history = self.historical_dates
        if any(history[i] >= history[i + 1] for i in range(HISTORY_DISPLAY_WINDOW - 1)):
            raise ValueError("historical_dates must be strictly increasing")
        if history[-1] != self.origin_date:
            raise ValueError("the last historical date must equal origin_date")

        if self.historical_prices[-1] != self.origin_close:
            raise ValueError("the last historical price must equal origin_close")

        price_count = len(self.historical_prices)
        if len(history) != price_count:
            raise ValueError("historical_dates and historical_prices lengths must match")
        return self
