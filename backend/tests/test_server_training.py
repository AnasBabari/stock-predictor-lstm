import numpy as np
import pandas as pd
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sklearn.preprocessing import RobustScaler

from artifacts.signing import Ed25519ManifestSigner, Ed25519ManifestVerifier
from config import FEATURES_V4
from server_models.contracts import (
    FORECAST_LENGTH,
    HISTORY_DISPLAY_WINDOW,
    WINDOW_SIZE,
    ServerForecastBundle,
)
from server_models.db import InMemoryRegistry
from server_models.storage import InMemoryObjectStore, ObjectStoreError
from server_models.training import train_server_forecast


class RecordingElasticNet:
    """Strict stand-in for ElasticNetForecaster: records every input it sees."""

    instances: list["RecordingElasticNet"] = []

    def __init__(self) -> None:
        self.predict_calls: list[np.ndarray] = []
        RecordingElasticNet.instances.append(self)

    def fit(self, features, targets):
        self.fit_input = (np.array(features), np.array(targets))
        return self

    def predict(self, features) -> np.ndarray:
        self.predict_calls.append(np.array(features))
        return np.full((features.shape[0], FORECAST_LENGTH), 0.001)


class RecordingRobustScaler:
    """Records every scaler instance while delegating to the real RobustScaler,
    so tests can recompute the exact expected inference input."""

    instances: list["RecordingRobustScaler"] = []

    def __init__(self) -> None:
        self._inner = RobustScaler()
        RecordingRobustScaler.instances.append(self)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def fit(self, features):
        self._inner.fit(features)
        return self

    def transform(self, features):
        return self._inner.transform(features)


def _stub_fetch(monkeypatch) -> pd.DataFrame:
    rows = WINDOW_SIZE + HISTORY_DISPLAY_WINDOW + 40
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.standard_normal((rows, len(FEATURES_V4))), columns=FEATURES_V4)
    close_prices = np.full(rows, 100.0) + rng.uniform(-2, 2, rows)
    dates = pd.date_range("2020-01-01", periods=rows)

    monkeypatch.setattr(
        "server_models.training.fetch_browser_data", lambda t: (df, close_prices, dates, {})
    )
    return df


def _stub_scaler(monkeypatch) -> None:
    monkeypatch.setattr("server_models.training.RobustScaler", RecordingRobustScaler)


def _mock_run(monkeypatch, elastic_net_rmse: float, elastic_net_mae: float, calls=None) -> None:
    def fake_run(*args, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        return {
            "models": {
                "elastic_net": {
                    "promotion": {"promoted": elastic_net_rmse < 1.0 and elastic_net_mae < 1.0},
                    "aggregate": {
                        "pooled": {
                            "relative_rmse": elastic_net_rmse,
                            "relative_mae": elastic_net_mae,
                        },
                        "per_horizon": {},
                    },
                },
                "ridge": {
                    "promotion": {"promoted": False},
                    "aggregate": {
                        "pooled": {"relative_rmse": 1.2, "relative_mae": 1.2},
                        "per_horizon": {},
                    },
                },
            }
        }

    monkeypatch.setattr("server_models.training.run_baseline_experiment", fake_run)


def _registry_and_storage():
    registry = InMemoryRegistry()
    storage = InMemoryObjectStore()
    signer = Ed25519ManifestSigner(Ed25519PrivateKey.generate())
    return registry, storage, signer


def _stub_model(monkeypatch) -> None:
    monkeypatch.setattr("server_models.training.ElasticNetForecaster", RecordingElasticNet)


def test_train_server_forecast_promotes_when_both_metrics_pass(monkeypatch):
    registry, storage, signer = _registry_and_storage()
    _stub_fetch(monkeypatch)
    _stub_model(monkeypatch)
    _mock_run(monkeypatch, elastic_net_rmse=0.8, elastic_net_mae=0.8)

    record = train_server_forecast("AAPL", registry, storage, signer)

    assert record is not None
    assert record.status == "promoted"
    promoted = registry.get_promoted("AAPL")
    assert promoted is not None and promoted.key.version_id == record.key.version_id
    assert storage.bundle_exists(record.key.version_id)


def test_train_server_forecast_skips_promotion_when_both_metrics_fail(monkeypatch):
    registry, storage, signer = _registry_and_storage()
    _stub_fetch(monkeypatch)
    _mock_run(monkeypatch, elastic_net_rmse=1.2, elastic_net_mae=1.2)

    record = train_server_forecast("AAPL", registry, storage, signer)

    assert record is None
    assert registry.get_promoted("AAPL") is None
    assert len(storage._objects) == 0


@pytest.mark.parametrize(
    ("rmse", "mae"),
    [
        (0.97, 1.0),  # mae exactly at threshold
        (1.0, 0.97),  # rmse exactly at threshold
        (0.98, 0.98),  # both exactly at threshold
        (0.8, 1.2),  # mae over
        (1.2, 0.8),  # rmse over
    ],
)
def test_train_server_forecast_requires_both_metrics_strictly_below_threshold(
    monkeypatch, rmse: float, mae: float
):
    registry, storage, signer = _registry_and_storage()
    _stub_fetch(monkeypatch)
    _mock_run(monkeypatch, elastic_net_rmse=rmse, elastic_net_mae=mae)

    record = train_server_forecast("AAPL", registry, storage, signer)

    assert record is None
    assert registry.get_promoted("AAPL") is None
    assert len(storage._objects) == 0


def test_train_server_forecast_runs_baseline_without_hgb(monkeypatch):
    """Gate parity: the server training path must never evaluate HGB and must
    use the full forecast horizon range, mirroring the research certification."""
    from config import MAX_FORECAST_DAYS

    registry, storage, signer = _registry_and_storage()
    _stub_fetch(monkeypatch)
    _stub_model(monkeypatch)
    calls: list[dict] = []
    _mock_run(monkeypatch, elastic_net_rmse=0.8, elastic_net_mae=0.8, calls=calls)

    train_server_forecast("AAPL", registry, storage, signer)

    config = calls[0]["config"]
    assert config.include_hgb is False
    assert config.horizons == tuple(range(1, MAX_FORECAST_DAYS + 1))


def test_train_server_forecast_infers_from_the_latest_window(monkeypatch):
    """The inference slice must be the raw latest WINDOW_SIZE rows transformed
    with the fitted scaler — never a stale row of the windowed dataset."""
    registry, storage, signer = _registry_and_storage()
    df = _stub_fetch(monkeypatch)
    _stub_scaler(monkeypatch)
    _stub_model(monkeypatch)
    _mock_run(monkeypatch, elastic_net_rmse=0.8, elastic_net_mae=0.8)

    record = train_server_forecast("AAPL", registry, storage, signer)

    model = RecordingElasticNet.instances[-1]
    assert model.predict_calls, "model.predict must have been called"
    scaler = RecordingRobustScaler.instances[-1]
    feature_values = df[FEATURES_V4].to_numpy(dtype=np.float64)
    feature_count = feature_values.shape[1]
    expected = scaler.transform(feature_values[-WINDOW_SIZE:].reshape(-1, feature_count)).reshape(
        1, WINDOW_SIZE, feature_count
    )
    np.testing.assert_allclose(model.predict_calls[0], expected, rtol=1e-9, atol=1e-12)
    bundle = ServerForecastBundle.model_validate_json(storage.get_bundle(record.key.version_id))
    assert bundle.origin_close == pytest.approx(float(bundle.historical_prices[-1]))
    assert bundle.origin_date == bundle.historical_dates[-1]
    # The future horizon must strictly follow the origin session.
    assert bundle.future_dates[0] > bundle.origin_date
    assert all(
        bundle.future_dates[i] < bundle.future_dates[i + 1]
        for i in range(len(bundle.future_dates) - 1)
    )


def test_train_server_forecast_bundle_is_signed_and_self_contained(monkeypatch):
    private_key = Ed25519PrivateKey.generate()
    registry = InMemoryRegistry()
    storage = InMemoryObjectStore()
    signer = Ed25519ManifestSigner(private_key)
    _stub_fetch(monkeypatch)
    _stub_model(monkeypatch)
    _mock_run(monkeypatch, elastic_net_rmse=0.8, elastic_net_mae=0.8)

    record = train_server_forecast("AAPL", registry, storage, signer)

    bundle_bytes = storage.get_bundle(record.key.version_id)
    assert record.signature
    verifier = Ed25519ManifestVerifier(private_key.public_key())
    assert verifier(bundle_bytes, record.signature) is True

    bundle = ServerForecastBundle.model_validate_json(bundle_bytes)
    assert len(bundle.historical_prices) == HISTORY_DISPLAY_WINDOW
    assert len(bundle.historical_dates) == HISTORY_DISPLAY_WINDOW
    assert bundle.evidence["metric_source"] == "server_purged_walk_forward"
    assert bundle.evidence["family"] == "elastic_net"


def test_train_server_forecast_bundle_is_immutable(monkeypatch):
    registry, storage, signer = _registry_and_storage()
    _stub_fetch(monkeypatch)
    _stub_model(monkeypatch)
    _mock_run(monkeypatch, elastic_net_rmse=0.8, elastic_net_mae=0.8)

    record = train_server_forecast("AAPL", registry, storage, signer)
    stored = storage.get_bundle(record.key.version_id)

    with pytest.raises(ObjectStoreError, match="immutable"):
        storage.put_bundle(record.key.version_id, b"tampered")
    assert storage.get_bundle(record.key.version_id) == stored
