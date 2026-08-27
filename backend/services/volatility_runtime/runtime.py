"""CPU-only ONNX inference runtime for one certified volatility ensemble.

The runtime loads only signed, checksum-verified release bundles, opens every
member graph on the CPU execution provider, and averages member outputs in
ascending seed order. Any signature failure, schema drift, malformed tensor,
or provider deviation aborts inference before a forecast is returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .contracts import (
    EXPECTED_OUTPUT_NAMES,
    VolatilityEnsembleForecast,
    VolatilityRuntimeContract,
)

if TYPE_CHECKING:  # pragma: no cover - import used only for typing
    from services.volatility_snapshot import VolatilityInferenceSnapshot


class _MemberSession(Protocol):
    """Narrow structural view of an ``onnxruntime.InferenceSession``."""

    def get_inputs(self) -> Sequence[Any]: ...

    def get_outputs(self) -> Sequence[Any]: ...

    def run(self, output_names: object, input_feed: Mapping[str, np.ndarray]) -> list[Any]: ...


def _open_session(path: Path) -> _MemberSession:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required to serve the certified volatility model; "
            "install the CPU-only package",
        ) from error
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])  # type: ignore[no-any-return]


def _verify_bundle(release_dir: Path, *, public_key_path: Path) -> dict:
    from release.bundle import verify_release

    return verify_release(release_dir, public_key_path=public_key_path)


def _node_names(nodes: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(node.name) for node in nodes)


def _member_path(release_dir: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    resolved = release_dir.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(release_dir.resolve()):
        raise ValueError("ensemble member path escapes the release directory")
    return resolved


class VolatilityOnnxRuntime:
    """Deterministic CPU ensemble over one signed volatility release."""

    def __init__(
        self, contract: VolatilityRuntimeContract, sessions: Sequence[_MemberSession]
    ) -> None:
        if not isinstance(contract, VolatilityRuntimeContract):
            raise ValueError("volatility runtime requires its frozen contract")
        if len(sessions) != len(contract.member_files):
            raise ValueError("one loaded session per ensemble member is required")
        expected_inputs = contract.expected_input_names()
        for session in sessions:
            if _node_names(session.get_inputs()) != expected_inputs:
                raise ValueError("member session inputs deviate from the certified contract")
            if _node_names(session.get_outputs()) != EXPECTED_OUTPUT_NAMES:
                raise ValueError("member session outputs deviate from the certified contract")
        self.contract = contract
        self._sessions = tuple(sessions)

    @property
    def model_id(self) -> str:
        return self.contract.model_id

    @property
    def member_seeds(self) -> tuple[int, ...]:
        return self.contract.member_seeds

    @property
    def metric_source(self) -> str:
        return self.contract.metric_source

    @property
    def certification_scope(self) -> str:
        return self.contract.certification_scope

    @property
    def model_version(self) -> str | None:
        return self.contract.model_version

    @property
    def news_status(self) -> str:
        return self.contract.news_status

    def is_certified_horizon(self, horizon: int) -> bool:
        return self.contract.is_certified_horizon(horizon)

    def certified_horizon_list(self) -> tuple[int, ...]:
        return self.contract.certified_horizon_list()

    def certification_summary(self, horizon: int) -> dict | None:
        summary = self.contract.certification_summary(horizon)
        return dict(summary) if summary is not None else None

    @classmethod
    def from_release_bundle(
        cls, release_dir: Path, *, public_key_path: Path
    ) -> VolatilityOnnxRuntime:
        """Verify a signed bundle, rebuild its contract, and load member sessions."""

        directory = Path(release_dir)
        manifest = _verify_bundle(directory, public_key_path=Path(public_key_path))
        metadata = manifest.get("metadata")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("release manifest has no file table")
        contract = VolatilityRuntimeContract.from_release_metadata(metadata, set(files))
        sessions = [_open_session(_member_path(directory, name)) for name in contract.member_files]
        return cls(contract, sessions)

    def forecast(
        self,
        snapshot: VolatilityInferenceSnapshot,
        *,
        news_features: np.ndarray | None = None,
    ) -> VolatilityEnsembleForecast:
        """Run every member on one causal snapshot and average their outputs."""
        if tuple(snapshot.feature_names) != self.contract.feature_names:
            raise ValueError("snapshot feature order does not match the signed release")
        feed = self.contract.prepare_feed(
            snapshot.features,
            snapshot.causal_har_variance,
            news_features=news_features,
        )
        collected: dict[str, list[np.ndarray]] = {name: [] for name in EXPECTED_OUTPUT_NAMES}
        for index, session in enumerate(self._sessions):
            seed = self.contract.member_seeds[index]
            try:
                raw_outputs = session.run(None, feed)
            except Exception as error:
                raise RuntimeError(f"volatility member (seed {seed}) inference failed") from error
            if len(raw_outputs) != len(EXPECTED_OUTPUT_NAMES):
                raise RuntimeError(f"volatility member (seed {seed}) returned an unexpected arity")
            outputs = dict(zip(EXPECTED_OUTPUT_NAMES, raw_outputs, strict=True))
            self.contract.validate_outputs(outputs)
            for name in EXPECTED_OUTPUT_NAMES:
                collected[name].append(np.asarray(outputs[name]))
        averaged = {
            name: np.mean(np.stack(values, axis=0), axis=0, dtype=np.float64)
            for name, values in collected.items()
        }
        return VolatilityEnsembleForecast(
            model_id=self.model_id,
            forecast_variance=averaged[EXPECTED_OUTPUT_NAMES[0]].astype(np.float32)[0],
            return_location=averaged[EXPECTED_OUTPUT_NAMES[1]].astype(np.float32)[0],
            direction_probabilities=averaged[EXPECTED_OUTPUT_NAMES[2]].astype(np.float32)[0],
            return_variance=averaged[EXPECTED_OUTPUT_NAMES[3]].astype(np.float32)[0],
        )
