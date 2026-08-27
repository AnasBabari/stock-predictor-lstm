from __future__ import annotations

import sys

import numpy as np
import pytest

from panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5
from services.volatility_runtime import (
    VolatilityEnsembleForecast,
    VolatilityOnnxRuntime,
    VolatilityRuntimeContract,
)
from services.volatility_runtime import runtime as runtime_module


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def __init__(
        self,
        outputs: dict[str, np.ndarray],
        *,
        inputs: tuple[str, ...] = ("features", "baseline_variance"),
        output_names: tuple[str, ...] = (
            "forecast_variance",
            "return_location",
            "direction_probabilities",
            "return_variance",
        ),
    ) -> None:
        self.outputs = outputs
        self.input_names = inputs
        self.output_names = output_names

    def get_inputs(self) -> list[_Node]:
        return [_Node(name) for name in self.input_names]

    def get_outputs(self) -> list[_Node]:
        return [_Node(name) for name in self.output_names]

    def run(self, _output_names: object, feed: dict) -> list[np.ndarray]:
        assert set(feed) == set(self.input_names)
        return [self.outputs[name] for name in self.output_names]


def _valid_outputs(variance: float = 2e-4) -> dict[str, np.ndarray]:
    ones = np.ones((1, 6), dtype=np.float32)
    probabilities = np.tile(np.array([0.5, 0.3, 0.2], dtype=np.float32), (1, 6, 1))
    return {
        "forecast_variance": variance * ones,
        "return_location": 0.01 * ones,
        "direction_probabilities": probabilities,
        "return_variance": (variance * 1.5) * ones,
    }


def _contract(**overrides) -> VolatilityRuntimeContract:
    payload: dict = {
        "model_id": "global-volatility-tcn-v1",
        "feature_names": DEPLOYABLE_FEATURE_COLUMNS_V5,
        "member_seeds": (41, 42),
        "member_files": ("members/seed-41.onnx", "members/seed-42.onnx"),
    }
    payload.update(overrides)
    return VolatilityRuntimeContract(**payload)


def _single_contract() -> VolatilityRuntimeContract:
    return _contract(member_seeds=(41,), member_files=("members/seed-41.onnx",))


def test_contract_binds_certified_schema_and_rejects_drift() -> None:
    contract = _contract()
    assert contract.horizons == (1, 3, 5, 7, 14, 30)
    assert contract.window_size == 60
    assert contract.expected_input_names() == ("features", "baseline_variance")
    with pytest.raises(ValueError, match="feature order"):
        _contract(feature_names=tuple(reversed(DEPLOYABLE_FEATURE_COLUMNS_V5)))
    with pytest.raises(ValueError, match="horizons"):
        _contract(horizons=(1, 3, 5, 7, 14))
    with pytest.raises(ValueError, match="window"):
        _contract(window_size=30)


def test_release_metadata_round_trips_json_certification_keys() -> None:
    metadata = {
        "runtime_schema": "volatility-runtime-v1",
        "model_id": "global-volatility-tcn-v1",
        "horizons": [1, 3, 5, 7, 14, 30],
        "window_size": 60,
        "news_feature_count": 0,
        "feature_names": list(DEPLOYABLE_FEATURE_COLUMNS_V5),
        "members": [{"seed": 41, "file": "members/seed-41.onnx"}],
        "certified_horizons": [1, 3, 5, 7],
        "certification_metrics": {
            "7": {"decision": "pass", "relative_qlike": 0.91},
        },
    }
    contract = VolatilityRuntimeContract.from_release_metadata(metadata, {"members/seed-41.onnx"})
    assert contract.is_certified_horizon(7)
    assert not contract.is_certified_horizon(14)
    assert contract.certification_summary(7)["decision"] == "pass"


def test_release_metadata_preserves_signed_v8_evidence_identity() -> None:
    metadata = {
        "runtime_schema": "volatility-runtime-v1",
        "model_id": "global-volatility-v8-numeric:fixture",
        "model_version": "global-volatility-v8-numeric",
        "protocol_version": "global-volatility-distribution-v8-numeric",
        "metric_source": "locked_historical_temporal_test_plus_asset_transfer",
        "certification_scope": "historical_temporal_test_plus_asset_transfer",
        "news_status": "not_certified",
        "horizons": [1, 3, 5, 7, 14, 30],
        "window_size": 60,
        "news_feature_count": 0,
        "feature_names": list(DEPLOYABLE_FEATURE_COLUMNS_V5),
        "members": [{"seed": 41, "file": "members/seed-41.onnx"}],
        "certified_horizons": [1, 3, 5, 7],
    }
    contract = VolatilityRuntimeContract.from_release_metadata(metadata, {"members/seed-41.onnx"})
    assert contract.model_version == "global-volatility-v8-numeric"
    assert contract.metric_source == "locked_historical_temporal_test_plus_asset_transfer"
    assert contract.certification_scope == "historical_temporal_test_plus_asset_transfer"
    assert contract.news_status == "not_certified"

    with pytest.raises(ValueError, match="metric source"):
        VolatilityRuntimeContract.from_release_metadata(
            {**metadata, "metric_source": "made_up_evidence"},
            {"members/seed-41.onnx"},
        )


def test_contract_rejects_invalid_membership() -> None:
    with pytest.raises(ValueError, match="unique and ascending"):
        _contract(member_seeds=(42, 41))
    with pytest.raises(ValueError, match="one to"):
        _contract(member_seeds=(), member_files=())
    with pytest.raises(ValueError, match="ONNX"):
        _contract(member_files=("members/seed-41.pt", "members/seed-42.onnx"))
    with pytest.raises(ValueError, match="escapes"):
        _contract(member_files=("../seed-41.onnx", "members/seed-42.onnx"))
    with pytest.raises(ValueError, match="misaligned"):
        _contract(member_seeds=(41,))


def test_prepare_feed_batches_and_validates_inputs() -> None:
    contract = _contract()
    features = np.random.default_rng(3).normal(size=(60, len(DEPLOYABLE_FEATURE_COLUMNS_V5)))
    baseline = np.full(6, 2e-4)
    feed = contract.prepare_feed(features, baseline)
    assert feed["features"].shape == (1, 60, len(DEPLOYABLE_FEATURE_COLUMNS_V5))
    assert feed["features"].dtype == np.float32
    assert feed["baseline_variance"].shape == (1, 6)
    with pytest.raises(ValueError, match="finite"):
        contract.prepare_feed(np.zeros((60, 26)) + np.nan, baseline)
    with pytest.raises(ValueError, match="strictly positive"):
        contract.prepare_feed(features, baseline * 0)
    news_contract = _contract(news_feature_count=4)
    with pytest.raises(ValueError, match="requires news"):
        news_contract.prepare_feed(features, baseline)
    with pytest.raises(ValueError, match="cannot receive news"):
        contract.prepare_feed(features, baseline, news_features=np.ones(4))


def test_runtime_ensemble_mean_is_deterministic_in_seed_order() -> None:
    contract = _contract()  # two members, seeds (41, 42)
    first = _FakeSession(_valid_outputs(variance=1e-4))
    second = _FakeSession(_valid_outputs(variance=3e-4))
    snapshot_features = np.zeros((60, len(DEPLOYABLE_FEATURE_COLUMNS_V5)), dtype=np.float32)

    class _Snapshot:
        feature_names = DEPLOYABLE_FEATURE_COLUMNS_V5
        features = snapshot_features
        causal_har_variance = np.full(6, 2e-4, dtype=np.float32)

    forecast = VolatilityOnnxRuntime(contract, [first, second]).forecast(_Snapshot())
    assert isinstance(forecast, VolatilityEnsembleForecast)
    assert forecast.model_id == "global-volatility-tcn-v1"
    assert forecast.forecast_variance.shape == (6,)
    np.testing.assert_allclose(forecast.forecast_variance, 2e-4, rtol=1e-6)
    np.testing.assert_allclose(forecast.return_variance, 3e-4, rtol=1e-6)
    np.testing.assert_allclose(
        forecast.direction_probabilities.sum(axis=1),
        np.ones(6),
        atol=1e-6,
    )
    again = VolatilityOnnxRuntime(contract, [first, second]).forecast(_Snapshot())
    np.testing.assert_array_equal(forecast.direction_probabilities, again.direction_probabilities)


def test_runtime_rejects_sessions_deviating_from_contract() -> None:
    with pytest.raises(ValueError, match="inputs deviate"):
        VolatilityOnnxRuntime(
            _single_contract(), [_FakeSession(_valid_outputs(), inputs=("x", "y"))]
        )
    renamed = _FakeSession(
        _valid_outputs(),
        output_names=(
            "variance",
            "return_location",
            "direction_probabilities",
            "return_variance",
        ),
    )
    with pytest.raises(ValueError, match="outputs deviate"):
        VolatilityOnnxRuntime(_single_contract(), [renamed])
    with pytest.raises(ValueError, match="per ensemble member"):
        VolatilityOnnxRuntime(_single_contract(), [])


def test_runtime_fails_closed_on_invalid_member_output() -> None:
    broken = _valid_outputs()
    broken["forecast_variance"] = -np.abs(broken["forecast_variance"])
    runtime = VolatilityOnnxRuntime(_single_contract(), [_FakeSession(broken)])

    class _Snapshot:
        feature_names = DEPLOYABLE_FEATURE_COLUMNS_V5
        features = np.zeros((60, len(DEPLOYABLE_FEATURE_COLUMNS_V5)), dtype=np.float32)
        causal_har_variance = np.full(6, 2e-4, dtype=np.float32)

    with pytest.raises(ValueError, match="strictly positive"):
        runtime.forecast(_Snapshot())
    off_simplex = _valid_outputs()
    off_simplex["direction_probabilities"][..., 0] += 0.05
    bad = VolatilityOnnxRuntime(_single_contract(), [_FakeSession(off_simplex)])
    with pytest.raises(ValueError, match="sum to one"):
        bad.forecast(_Snapshot())


def test_from_release_bundle_builds_verified_contract(monkeypatch, tmp_path) -> None:
    opened: list[str] = []

    def fake_open(path):
        opened.append(path.name)
        return _FakeSession(_valid_outputs())

    metadata = {
        "runtime_schema": "volatility-runtime-v1",
        "model_id": "global-volatility-tcn-v1",
        "horizons": [1, 3, 5, 7, 14, 30],
        "window_size": 60,
        "news_feature_count": 0,
        "feature_names": list(DEPLOYABLE_FEATURE_COLUMNS_V5),
        "members": [
            {"seed": 41, "file": "members/seed-41.onnx"},
            {"seed": 42, "file": "members/seed-42.onnx"},
        ],
    }
    manifest = {
        "metadata": metadata,
        "files": {"members/seed-41.onnx": "0" * 64, "members/seed-42.onnx": "0" * 64},
    }
    monkeypatch.setattr(runtime_module, "_verify_bundle", lambda *_args, **_kw: manifest)
    monkeypatch.setattr(runtime_module, "_open_session", fake_open)
    runtime = VolatilityOnnxRuntime.from_release_bundle(
        tmp_path, public_key_path=tmp_path / "public.pem"
    )
    assert opened == ["seed-41.onnx", "seed-42.onnx"]
    assert runtime.member_seeds == (41, 42)
    with pytest.raises(ValueError, match="absent from the signed file manifest"):
        orphaned = dict(metadata)
        orphaned["members"] = [{"seed": 41, "file": "members/ghost.onnx"}]
        monkeypatch.setattr(
            runtime_module, "_verify_bundle", lambda *_a, **_kw: {"metadata": orphaned, "files": {}}
        )
        VolatilityOnnxRuntime.from_release_bundle(tmp_path, public_key_path=tmp_path / "public.pem")


def test_missing_onnxruntime_yields_actionable_error(monkeypatch, tmp_path) -> None:
    metadata = {
        "runtime_schema": "volatility-runtime-v1",
        "model_id": "global-volatility-tcn-v1",
        "horizons": [1, 3, 5, 7, 14, 30],
        "window_size": 60,
        "news_feature_count": 0,
        "feature_names": list(DEPLOYABLE_FEATURE_COLUMNS_V5),
        "members": [{"seed": 41, "file": "members/seed-41.onnx"}],
    }
    manifest = {"metadata": metadata, "files": {"members/seed-41.onnx": "0" * 64}}
    monkeypatch.setattr(runtime_module, "_verify_bundle", lambda *_args, **_kw: manifest)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(RuntimeError, match="onnxruntime"):
        VolatilityOnnxRuntime.from_release_bundle(tmp_path, public_key_path=tmp_path / "public.pem")
