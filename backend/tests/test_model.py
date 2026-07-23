# backend/tests/test_model.py
from config import MAX_FORECAST_DAYS


def test_build_model_output_shape():
    from model import build_model

    m = build_model(forecast_days=7)
    assert m.output_shape == (None, 7)


def test_build_model_default_output():
    from model import build_model

    m = build_model()
    assert m.output_shape == (None, MAX_FORECAST_DAYS)


def test_evaluate_model_returns_all_keys(trained_model):
    from model import evaluate_model

    model, (_, X_test, _, y_test, scaler) = trained_model
    metrics = evaluate_model(model, X_test, y_test, scaler)
    assert set(metrics) == {"rmse", "mae", "mape", "r2", "directional_accuracy"}


def test_evaluate_model_values_are_finite(trained_model):
    import math

    from model import evaluate_model

    model, (_, X_test, _, y_test, scaler) = trained_model
    metrics = evaluate_model(model, X_test, y_test, scaler)
    for k, v in metrics.items():
        if v is not None:
            assert math.isfinite(v), f"{k} is not finite: {v}"


def test_evaluate_model_empty_returns_nones(preprocessed):
    import numpy as np

    from model import build_model, evaluate_model

    X_train, _, y_train, _, scaler = preprocessed
    model = build_model()
    metrics = evaluate_model(model, np.array([]), np.array([]), scaler)
    assert all(v is None for v in metrics.values())


def test_predict_future_length(trained_model, synthetic_prices):
    from model import predict_future

    model, (*_, scaler) = trained_model
    for days in [1, 7, 14, 30]:
        preds = predict_future(model, synthetic_prices, scaler, days=days)
        assert len(preds) == days, f"Expected {days} predictions, got {len(preds)}"


def test_predict_future_prices_positive(trained_model, synthetic_prices):
    from model import predict_future

    model, (*_, scaler) = trained_model
    preds = predict_future(model, synthetic_prices, scaler, days=7)
    assert all(p > 0 for p in preds)


def test_directional_accuracy_in_range(trained_model):
    from model import evaluate_model

    model, (_, X_test, _, y_test, scaler) = trained_model
    metrics = evaluate_model(model, X_test, y_test, scaler)
    da = metrics["directional_accuracy"]
    if da is not None:
        assert 0.0 <= da <= 1.0


def test_mape_is_percentage(trained_model):
    from model import evaluate_model

    model, (_, X_test, _, y_test, scaler) = trained_model
    metrics = evaluate_model(model, X_test, y_test, scaler)
    if metrics["mape"] is not None:
        assert metrics["mape"] >= 0


def test_scaler_persistence_and_loading(preprocessed, tmp_path, monkeypatch):
    import model as model_module
    from model import load_or_train, train_model

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    X_train, X_test, y_train, y_test, scaler = preprocessed

    # Train model & scaler
    trained_m, trained_s = train_model(
        X_train, y_train, X_test, y_test, ticker="PERSIST_TEST", scaler=scaler
    )
    assert (tmp_path / "PERSIST_TEST_model.keras").exists()
    assert (tmp_path / "PERSIST_TEST_scaler.joblib").exists()

    # Load from cache
    loaded_m, loaded_s = load_or_train(
        "PERSIST_TEST", X_train, y_train, X_test, y_test, scaler=scaler
    )
    assert loaded_s is not None
    assert getattr(loaded_s, "data_min_", None) is not None


def test_build_attention_lstm_model_outputs():
    from model import build_attention_lstm_model
    from config import WINDOW_SIZE

    m = build_attention_lstm_model(forecast_days=7)
    
    # Ensure it's a multi-output functional Model, not a Sequential
    assert len(m.outputs) == 2, "Model should return exactly 2 outputs"
    
    predictions_shape = m.outputs[0].shape
    attention_weights_shape = m.outputs[1].shape
    
    # Check shape: (batch_size, forecast_days)
    assert predictions_shape == (None, 7), f"Expected predictions shape (None, 7), got {predictions_shape}"
    
    # Check shape: (batch_size, sequence_length, sequence_length)
    assert attention_weights_shape == (None, WINDOW_SIZE, WINDOW_SIZE), f"Expected attention weights shape (None, {WINDOW_SIZE}, {WINDOW_SIZE}), got {attention_weights_shape}"
