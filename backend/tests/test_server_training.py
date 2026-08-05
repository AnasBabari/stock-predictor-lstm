from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from server_models.training import train_server_forecast


def _stub_fetch(monkeypatch) -> None:
    from config import FEATURES_V4, MAX_FORECAST_DAYS, WINDOW_SIZE

    rows = WINDOW_SIZE + MAX_FORECAST_DAYS + 10
    df = pd.DataFrame(np.random.randn(rows, len(FEATURES_V4)), columns=FEATURES_V4)
    close_prices = np.random.uniform(10, 100, rows)
    dates = pd.date_range("2020-01-01", periods=rows)

    monkeypatch.setattr(
        "server_models.training.fetch_browser_data", lambda t: (df, close_prices, dates, {})
    )


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
    registry, storage, signer = MagicMock(), MagicMock(), MagicMock()
    signer.sign.return_value = "fake_signature"
    return registry, storage, signer


def test_train_server_forecast_promotes_when_both_metrics_pass(monkeypatch):
    registry, storage, signer = _registry_and_storage()
    _stub_fetch(monkeypatch)
    _mock_run(monkeypatch, elastic_net_rmse=0.8, elastic_net_mae=0.8)

    record = train_server_forecast("AAPL", registry, storage, signer)

    assert record is not None
    assert record.status == "promoted"
    registry.promote_model.assert_called_once_with(record)
    storage.put_bundle.assert_called_once()


def test_train_server_forecast_skips_promotion_when_both_metrics_fail(monkeypatch):
    registry, storage, signer = _registry_and_storage()
    _stub_fetch(monkeypatch)
    _mock_run(monkeypatch, elastic_net_rmse=1.2, elastic_net_mae=1.2)

    record = train_server_forecast("AAPL", registry, storage, signer)

    assert record is None
    registry.promote_model.assert_not_called()
    storage.put_bundle.assert_not_called()


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
    registry.promote_model.assert_not_called()
    storage.put_bundle.assert_not_called()


def test_train_server_forecast_runs_baseline_without_hgb(monkeypatch):
    """Gate parity: the server training path must never evaluate HGB and must
    use the full forecast horizon range, mirroring the research certification."""
    from config import MAX_FORECAST_DAYS

    registry, storage, signer = _registry_and_storage()
    _stub_fetch(monkeypatch)
    calls: list[dict] = []
    _mock_run(monkeypatch, elastic_net_rmse=0.8, elastic_net_mae=0.8, calls=calls)

    train_server_forecast("AAPL", registry, storage, signer)

    config = calls[0]["config"]
    assert config.include_hgb is False
    assert config.horizons == tuple(range(1, MAX_FORECAST_DAYS + 1))
