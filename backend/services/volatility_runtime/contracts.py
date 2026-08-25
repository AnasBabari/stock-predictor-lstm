"""Frozen serving contracts for the certified global-volatility ONNX ensemble.

The serving layer never invents schema details: feature order, horizons, and
window length are bound to the certified development protocol here. Any drift
between a signed release manifest and this module fails closed before a single
tensor reaches the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5
from release.bundle import RUNTIME_SCHEMA_VERSION

VOLATILITY_HORIZONS = (1, 3, 5, 7, 14, 30)
VOLATILITY_WINDOW_SIZE = 60

MODEL_INPUT_FEATURES = "features"
MODEL_INPUT_BASELINE = "baseline_variance"
MODEL_INPUT_NEWS = "news_features"

MODEL_OUTPUT_VARIANCE = "forecast_variance"
MODEL_OUTPUT_LOCATION = "return_location"
MODEL_OUTPUT_PROBABILITIES = "direction_probabilities"
MODEL_OUTPUT_RETURN_VARIANCE = "return_variance"

EXPECTED_OUTPUT_NAMES: tuple[str, ...] = (
    MODEL_OUTPUT_VARIANCE,
    MODEL_OUTPUT_LOCATION,
    MODEL_OUTPUT_PROBABILITIES,
    MODEL_OUTPUT_RETURN_VARIANCE,
)

MAX_ENSEMBLE_MEMBERS = 8
MAX_NEWS_FEATURE_COUNT = 1024
PROBABILITY_SUM_TOLERANCE = 1e-4


def _validate_member_file(name: object) -> str:
    if not isinstance(name, str) or not name or len(name) > 240:
        raise ValueError("ensemble member file name is invalid")
    if "\\" in name or ":" in name or name.endswith("/"):
        raise ValueError("ensemble member file name must be a portable posix path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("ensemble member file path escapes the release directory")
    if not name.endswith(".onnx"):
        raise ValueError("ensemble member files must be ONNX artifacts")
    return name


@dataclass(frozen=True)
class VolatilityRuntimeContract:
    """Immutable I/O contract shared by every ensemble member session."""

    model_id: str
    feature_names: tuple[str, ...]
    member_seeds: tuple[int, ...]
    member_files: tuple[str, ...]
    horizons: tuple[int, ...] = VOLATILITY_HORIZONS
    window_size: int = VOLATILITY_WINDOW_SIZE
    news_feature_count: int = 0
    certified_horizons: tuple[int, ...] | None = None
    certification_metrics: Mapping[int, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id or len(self.model_id) > 128:
            raise ValueError("volatility model id is missing or oversized")
        if self.feature_names != DEPLOYABLE_FEATURE_COLUMNS_V5:
            raise ValueError("serving feature order does not match the certified schema")
        if self.horizons != VOLATILITY_HORIZONS:
            raise ValueError(f"serving horizons must be exactly {VOLATILITY_HORIZONS}")
        if self.window_size != VOLATILITY_WINDOW_SIZE:
            raise ValueError(
                f"serving window must be exactly {VOLATILITY_WINDOW_SIZE} sessions",
            )
        if (
            not isinstance(self.news_feature_count, int)
            or isinstance(self.news_feature_count, bool)
            or not 0 <= self.news_feature_count <= MAX_NEWS_FEATURE_COUNT
        ):
            raise ValueError("news feature count is out of bounds")
        members = (self.member_seeds, self.member_files)
        if any(not isinstance(group, tuple) for group in members):
            raise ValueError("ensemble membership must be declared as tuples")
        if len(self.member_seeds) != len(self.member_files):
            raise ValueError("ensemble seeds and files are misaligned")
        if not 1 <= len(self.member_seeds) <= MAX_ENSEMBLE_MEMBERS:
            raise ValueError(f"ensemble requires one to {MAX_ENSEMBLE_MEMBERS} members")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 1
            for seed in self.member_seeds
        ):
            raise ValueError("ensemble member seeds must be positive integers")
        if tuple(sorted(self.member_seeds)) != self.member_seeds:
            raise ValueError("ensemble member seeds must be unique and ascending")
        for file_name in self.member_files:
            _validate_member_file(file_name)
        if len(set(self.member_files)) != len(self.member_files):
            raise ValueError("ensemble member files must be unique")
        if self.certified_horizons is not None:
            certified = self.certified_horizons
            if (
                not isinstance(certified, tuple)
                or any(isinstance(h, bool) or not isinstance(h, int) for h in certified)
                or tuple(sorted(certified)) != certified
                or any(h not in self.horizons for h in certified)
                or not certified
            ):
                raise ValueError("certified horizons must be a sorted subset of serving horizons")
        if self.certification_metrics is not None:
            if not isinstance(self.certification_metrics, Mapping):
                raise ValueError("certification metrics must be a mapping")
            for horizon, summary in self.certification_metrics.items():
                if horizon not in self.horizons or not isinstance(summary, Mapping):
                    raise ValueError("certification metrics keys must be serving horizons")

    @classmethod
    def from_release_metadata(
        cls, metadata: Any, available_files: set[str]
    ) -> VolatilityRuntimeContract:
        """Parse signed bundle metadata; unknown shapes fail closed."""
        if not isinstance(metadata, dict):
            raise ValueError("release metadata is missing or malformed")
        if metadata.get("runtime_schema") != RUNTIME_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported volatility runtime schema: {metadata.get('runtime_schema')!r}"
            )
        model_id = metadata.get("model_id")
        feature_names = metadata.get("feature_names")
        horizons = metadata.get("horizons")
        window_size = metadata.get("window_size")
        news_feature_count = metadata.get("news_feature_count", 0)
        raw_members = metadata.get("members")
        certified_horizons_raw = metadata.get("certified_horizons")
        if certified_horizons_raw is not None:
            if not isinstance(certified_horizons_raw, list) or any(
                isinstance(h, bool) or not isinstance(h, int) for h in certified_horizons_raw
            ):
                raise ValueError("release certified horizons are malformed")
            certified_horizons: tuple[int, ...] | None = tuple(certified_horizons_raw)
        else:
            certified_horizons = None
        certification_metrics_raw = metadata.get("certification_metrics")
        certification_metrics: dict[int, dict[str, Any]] | None
        if certification_metrics_raw is None:
            certification_metrics = None
        else:
            if not isinstance(certification_metrics_raw, dict):
                raise ValueError("release certification metrics are malformed")
            certification_metrics = {}
            for raw_key, summary in certification_metrics_raw.items():
                if isinstance(raw_key, bool) or not isinstance(summary, dict):
                    raise ValueError("release certification metrics are malformed")
                if isinstance(raw_key, int):
                    key = raw_key
                elif isinstance(raw_key, str) and raw_key.isdecimal():
                    key = int(raw_key)
                else:
                    raise ValueError("release certification metrics are malformed")
                certification_metrics[key] = dict(summary)
        if not isinstance(model_id, str):
            raise ValueError("release model id is missing")
        if not isinstance(feature_names, list) or not all(
            isinstance(name, str) for name in feature_names
        ):
            raise ValueError("release feature names are malformed")
        if not isinstance(horizons, list) or not all(
            isinstance(h, int) and not isinstance(h, bool) for h in horizons
        ):
            raise ValueError("release horizons are malformed")
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise ValueError("release window size is malformed")
        if not isinstance(news_feature_count, int) or isinstance(news_feature_count, bool):
            raise ValueError("release news feature count is malformed")
        if not isinstance(raw_members, list):
            raise ValueError("release ensemble membership is missing")
        seeds: list[int] = []
        files: list[str] = []
        for row in raw_members:
            if not isinstance(row, dict):
                raise ValueError("release ensemble member entry is malformed")
            seed, file_name = row.get("seed"), row.get("file")
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError("release ensemble member seed is malformed")
            _validate_member_file(file_name)
            if file_name not in available_files:
                raise ValueError("release ensemble member is absent from the signed file manifest")
            seeds.append(seed)
            files.append(file_name)
        return cls(
            model_id=model_id,
            feature_names=tuple(feature_names),
            member_seeds=tuple(seeds),
            member_files=tuple(files),
            horizons=tuple(horizons),
            window_size=int(window_size),
            news_feature_count=int(news_feature_count),
            certified_horizons=certified_horizons,
            certification_metrics=certification_metrics,
        )

    @property
    def horizon_count(self) -> int:
        return len(self.horizons)

    def is_certified_horizon(self, horizon: int) -> bool:
        """True when the locked certification cleared this specific horizon."""
        if self.certified_horizons is None:
            return True
        return horizon in self.certified_horizons

    def certification_summary(self, horizon: int) -> Mapping[str, Any] | None:
        if self.certification_metrics is None:
            return None
        summary = self.certification_metrics.get(horizon)
        return dict(summary) if isinstance(summary, Mapping) else None

    def expected_input_names(self) -> tuple[str, ...]:
        inputs = [MODEL_INPUT_FEATURES, MODEL_INPUT_BASELINE]
        if self.news_feature_count:
            inputs.append(MODEL_INPUT_NEWS)
        return tuple(inputs)

    def prepare_feed(
        self,
        features: np.ndarray,
        baseline_variance: np.ndarray,
        news_features: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Validate single-origin arrays and batch them into the model feed."""
        prepared = {
            MODEL_INPUT_FEATURES: self._prepare_matrix(
                features,
                (self.window_size, len(self.feature_names)),
                "feature window",
            ),
            MODEL_INPUT_BASELINE: self._prepare_vector(
                baseline_variance, "causal baseline variance"
            ),
        }
        if self.news_feature_count:
            if news_features is None:
                raise ValueError("this release requires news features at inference")
            prepared[MODEL_INPUT_NEWS] = self._prepare_vector(news_features, "news features")
        elif news_features is not None:
            raise ValueError("market-only release cannot receive news features")
        return prepared

    @staticmethod
    def _prepare_matrix(values: np.ndarray, shape: tuple[int, int], label: str) -> np.ndarray:
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
        if array.shape != shape:
            raise ValueError(f"{label} has shape {array.shape}, expected {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{label} must be finite")
        return array[np.newaxis, ...]

    @classmethod
    def _prepare_vector(cls, values: np.ndarray, label: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{label} must be a non-empty vector")
        if not np.isfinite(array).all() or (array <= 0).any():
            raise ValueError(f"{label} must be finite and strictly positive")
        return np.ascontiguousarray(array[np.newaxis, :])

    def validate_outputs(self, outputs: Mapping[str, np.ndarray]) -> None:
        """Fail closed on any malformed, non-finite, or degenerate member output."""
        if set(outputs) != set(EXPECTED_OUTPUT_NAMES):
            raise ValueError("member outputs do not match the certified graph signature")
        batch = -1
        # Trailing dimensions after the leading batch axis.
        trailing_dimensions = {
            MODEL_OUTPUT_VARIANCE: 1,
            MODEL_OUTPUT_LOCATION: 1,
            MODEL_OUTPUT_PROBABILITIES: 2,
            MODEL_OUTPUT_RETURN_VARIANCE: 1,
        }
        arrays = {name: np.asarray(outputs[name]) for name in EXPECTED_OUTPUT_NAMES}
        for name, expected_trailing in trailing_dimensions.items():
            array = arrays[name]
            if array.ndim != 1 + expected_trailing:
                raise ValueError(f"{name} output rank is incompatible")
            if batch == -1:
                batch = array.shape[0]
            elif array.shape[0] != batch:
                raise ValueError("member outputs disagree on the batch dimension")
        variance = arrays[MODEL_OUTPUT_VARIANCE]
        if variance.shape[-1] != self.horizon_count:
            raise ValueError("forecast variance horizon dimension is incompatible")
        positive_outputs = (variance, arrays[MODEL_OUTPUT_RETURN_VARIANCE])
        for array in positive_outputs:
            if not np.isfinite(array).all() or (array <= 0).any():
                raise ValueError("variance outputs must be finite and strictly positive")
        location = arrays[MODEL_OUTPUT_LOCATION]
        probabilities = arrays[MODEL_OUTPUT_PROBABILITIES]
        if (
            location.shape[-1] != self.horizon_count
            or probabilities.shape[-2] != self.horizon_count
        ):
            raise ValueError("return head outputs have an incompatible horizon dimension")
        if not np.isfinite(location).all():
            raise ValueError("return location must be finite")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("direction probabilities must be finite and non-negative")
        sums = probabilities.sum(axis=-1, dtype=np.float64)
        if float(np.max(np.abs(sums - 1.0))) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError("direction probability rows do not sum to one")


@dataclass(frozen=True)
class VolatilityEnsembleForecast:
    """Validated single-origin forecast distribution returned to the API layer."""

    model_id: str
    forecast_variance: np.ndarray
    return_location: np.ndarray
    direction_probabilities: np.ndarray
    return_variance: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("forecast must carry its model id")
        validate_single_origin_forecast(
            forecast_variance=self.forecast_variance,
            return_location=self.return_location,
            direction_probabilities=self.direction_probabilities,
            return_variance=self.return_variance,
        )


def validate_single_origin_forecast(
    *,
    forecast_variance: np.ndarray,
    return_location: np.ndarray,
    direction_probabilities: np.ndarray,
    return_variance: np.ndarray,
) -> None:
    """Shape/finiteness gate for squeezed (h,), (h,), (h, 3), (h,) forecasts."""
    arrays = {
        "forecast_variance": np.asarray(forecast_variance),
        "return_location": np.asarray(return_location),
        "direction_probabilities": np.asarray(direction_probabilities),
        "return_variance": np.asarray(return_variance),
    }
    if any(array.ndim != 1 for name, array in arrays.items() if name != "direction_probabilities"):
        raise ValueError("single-origin forecast vectors must be one-dimensional")
    if arrays["direction_probabilities"].ndim != 2:
        raise ValueError("direction probabilities must be two-dimensional")
    horizons = {arrays[name].shape[0] for name in ("forecast_variance", "return_location")}
    horizons.add(arrays["direction_probabilities"].shape[0])
    horizons.add(arrays["return_variance"].shape[0])
    if len(horizons) != 1:
        raise ValueError("forecast outputs disagree on the horizon count")
    if next(iter(horizons)) != len(VOLATILITY_HORIZONS):
        raise ValueError("single-origin forecast must cover every certified horizon")
    if (
        not np.isfinite(arrays["forecast_variance"]).all()
        or (arrays["forecast_variance"] <= 0).any()
    ):
        raise ValueError("forecast variance must be finite and strictly positive")
    if not np.isfinite(arrays["return_variance"]).all() or (arrays["return_variance"] <= 0).any():
        raise ValueError("return variance must be finite and strictly positive")
    if not np.isfinite(arrays["return_location"]).all():
        raise ValueError("return location must be finite")
    probabilities = arrays["direction_probabilities"]
    if (
        probabilities.shape[1] != 3
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
    ):
        raise ValueError("direction probabilities must be finite, non-negative, and three-class")
    sums = probabilities.sum(axis=1, dtype=np.float64)
    if float(np.max(np.abs(sums - 1.0))) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError("direction probability rows do not sum to one")
