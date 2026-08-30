"""Machine-readable V11.2 development evidence and reporting helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .v11_2_protocol import canonical_json_digest
from .v11_2_trainer import V112Forecast, V112ResidualTrainingResult


@dataclass(frozen=True)
class V112SeedEvidence:
    seed: int
    horizon: int
    family: str
    best_epoch: int | None
    epoch_zero_crps: float | None
    validation_crps: float
    validation_qlike: float
    validation_coverage_80: float
    stock_origin_observations: int
    unique_sessions: int
    predictions_sha256: str
    state_sha256: str | None
    stop_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predictions_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def seed_evidence_from_forecast(
    *,
    seed: int,
    horizon: int,
    forecast: V112Forecast,
    dates: list[str] | tuple[str, ...],
    training: V112ResidualTrainingResult | None = None,
) -> V112SeedEvidence:
    unique_sessions = len(set(str(value) for value in dates))
    metrics = forecast.metrics()
    return V112SeedEvidence(
        seed=seed,
        horizon=horizon,
        family=forecast.family,
        best_epoch=training.best_epoch if training else None,
        epoch_zero_crps=training.epoch_zero_crps if training else None,
        validation_crps=metrics["crps_mean"],
        validation_qlike=metrics["qlike_mean"],
        validation_coverage_80=metrics["coverage_80"],
        stock_origin_observations=len(forecast.crps),
        unique_sessions=unique_sessions,
        predictions_sha256=predictions_digest(forecast.location, forecast.variance),
        state_sha256=training.best_state_sha256 if training else None,
        stop_reason=training.stop_reason if training else None,
    )


def write_seed_evidence(evidence: V112SeedEvidence, path: Path) -> str:
    payload = evidence.to_dict()
    payload["evidence_sha256"] = canonical_json_digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(payload["evidence_sha256"])


def write_development_report(payload: dict[str, Any], path: Path) -> str:
    """Write a report that cannot accidentally claim the final holdout was used."""
    if payload.get("sealed_test_status") != "LOCKED_UNOPENED":
        raise ValueError("V11.2 development reports must state LOCKED_UNOPENED")
    body = dict(payload)
    body["report_sha256"] = canonical_json_digest(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    return str(body["report_sha256"])
