"""Shared forecast response contracts for the public API and the
server-forecast serving layer.

Both ``api.py`` (browser path) and ``server_models/api.py`` (server-pretrained
path) return the same envelope, so a server bundle payload validates against
the same :class:`PriceForecastResponse` used by the public forecast endpoints.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TIMING_FIELDS = (
    "queue_wait",
    "market_data",
    "feature_preparation",
    "artifact_load_validation",
    "training",
    "inference",
    "total",
)
EXECUTION_MODES = (
    "response_cache_hit",
    "artifact_loaded",
    "baseline_fallback",
    "trained",
    "coalesced",
)
ARTIFACT_STATES = ("fresh", "missing", "stale", "incompatible")
ARTIFACT_ACTIONS = ("loaded", "retrained", "not_applicable")
STATUS_STAGES = (
    "queued",
    "downloading_market_data",
    "preparing_features",
    "checking_artifact",
    "training",
    "generating_forecast",
    "completed",
    "failed",
)

NonNegativeSeconds = Annotated[float, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1)]


class PredictionTimings(BaseModel):
    queue_wait: NonNegativeSeconds | None
    market_data: NonNegativeSeconds | None
    feature_preparation: NonNegativeSeconds | None
    artifact_load_validation: NonNegativeSeconds | None
    training: NonNegativeSeconds | None
    inference: NonNegativeSeconds | None
    total: NonNegativeSeconds


class PredictionExecution(BaseModel):
    mode: Literal[
        "response_cache_hit",
        "artifact_loaded",
        "baseline_fallback",
        "trained",
        "coalesced",
    ]
    coalesced: bool


class ForecastMetadata(BaseModel):
    """Stable telemetry contract plus permissive legacy runtime diagnostics."""

    model_config = ConfigDict(extra="allow")

    timings_seconds: PredictionTimings
    execution: PredictionExecution
    artifact_state_before: Literal["fresh", "missing", "stale", "incompatible"] | None
    artifact_action: Literal["loaded", "retrained", "not_applicable"]


class ForecastResponse(BaseModel):
    """Stable forecast envelope; metrics remain model-specific legacy data."""

    model_config = ConfigDict(extra="allow")

    ticker: str
    forecast_days: int
    future_dates: list[str]
    metrics: dict[str, Any]
    metadata: ForecastMetadata


class PriceForecastResponse(ForecastResponse):
    historical_dates: list[str]
    historical_prices: list[float]
    predicted_prices: list[float]


class AttentionWeight(BaseModel):
    index: int
    date: str
    weight: float


class DirectionForecastResponse(ForecastResponse):
    directions: list[Literal["Up", "Down"]]
    probabilities: list[Probability]
    attention_weights: list[AttentionWeight]
    sentiment: dict[str, Any]


class PredictionStatusResponse(BaseModel):
    status: Literal["queued", "running", "completed", "failed"]
    stage: Literal[
        "queued",
        "downloading_market_data",
        "preparing_features",
        "checking_artifact",
        "training",
        "generating_forecast",
        "completed",
        "failed",
    ]
    coalesced: bool
