from unittest.mock import MagicMock

from server_models.training import train_server_forecast


def test_train_server_forecast_promotes_on_success(monkeypatch):
    mock_registry = MagicMock()
    mock_storage = MagicMock()
    mock_signer = MagicMock()
    mock_signer.sign.return_value = "fake_signature"

    # Mock fetch_browser_data to return a tiny valid dataframe
    import numpy as np
    import pandas as pd

    from config import FEATURES_V4, MAX_FORECAST_DAYS, WINDOW_SIZE

    # We need at least WINDOW_SIZE + MAX_FORECAST_DAYS + 1 rows
    rows = WINDOW_SIZE + MAX_FORECAST_DAYS + 10
    df = pd.DataFrame(np.random.randn(rows, len(FEATURES_V4)), columns=FEATURES_V4)
    close_prices = np.random.uniform(10, 100, rows)
    dates = pd.date_range("2020-01-01", periods=rows)

    monkeypatch.setattr(
        "server_models.training.fetch_browser_data", lambda t: (df, close_prices, dates, {})
    )

    # Mock run_baseline_experiment to return a passing result
    mock_result = {
        "models": {
            "elastic_net": {
                "promotion": {"promoted": True},
                "aggregate": {"pooled": {"relative_rmse": 0.8}, "per_horizon": {}},
            },
            "ridge": {
                "promotion": {"promoted": False},
                "aggregate": {"pooled": {"relative_rmse": 1.2}, "per_horizon": {}},
            },
        }
    }
    monkeypatch.setattr(
        "server_models.training.run_baseline_experiment", lambda *args, **kwargs: mock_result
    )

    record = train_server_forecast("AAPL", mock_registry, mock_storage, mock_signer)

    assert record is not None
    assert record.status == "promoted"
    mock_registry.promote_model.assert_called_once_with(record)
    mock_storage.put_bundle.assert_called_once()


def test_train_server_forecast_skips_promotion_on_fail(monkeypatch):
    mock_registry = MagicMock()
    mock_storage = MagicMock()
    mock_signer = MagicMock()

    import numpy as np
    import pandas as pd

    from config import FEATURES_V4, MAX_FORECAST_DAYS, WINDOW_SIZE

    rows = WINDOW_SIZE + MAX_FORECAST_DAYS + 10
    df = pd.DataFrame(np.random.randn(rows, len(FEATURES_V4)), columns=FEATURES_V4)
    close_prices = np.random.uniform(10, 100, rows)
    dates = pd.date_range("2020-01-01", periods=rows)

    monkeypatch.setattr(
        "server_models.training.fetch_browser_data", lambda t: (df, close_prices, dates, {})
    )

    # Mock run_baseline_experiment to return a failing result
    mock_result = {
        "models": {
            "elastic_net": {
                "promotion": {"promoted": False},
                "aggregate": {"pooled": {"relative_rmse": 1.2}, "per_horizon": {}},
            },
            "ridge": {
                "promotion": {"promoted": False},
                "aggregate": {"pooled": {"relative_rmse": 1.2}, "per_horizon": {}},
            },
        }
    }
    monkeypatch.setattr(
        "server_models.training.run_baseline_experiment", lambda *args, **kwargs: mock_result
    )

    record = train_server_forecast("AAPL", mock_registry, mock_storage, mock_signer)

    assert record is None
    mock_registry.promote_model.assert_not_called()
    mock_storage.put_bundle.assert_not_called()
