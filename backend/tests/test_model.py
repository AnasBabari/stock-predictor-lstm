# backend/tests/test_model.py
import json

import numpy as np
import pytest

from config import FEATURES, MAX_FORECAST_DAYS, WINDOW_SIZE


def test_build_lstm_model_output_shape():
    from model import build_lstm_model

    m = build_lstm_model(forecast_days=7, num_features=len(FEATURES))
    assert m.output_shape == (None, 7)


def test_build_lstm_model_default_output():
    from model import build_lstm_model

    m = build_lstm_model()
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
    from model import build_lstm_model, evaluate_model

    X_train, _, y_train, _, scaler = preprocessed
    model = build_lstm_model()
    metrics = evaluate_model(model, np.array([]), np.array([]), scaler)
    assert all(v is None for v in metrics.values())


def test_predict_future_length(trained_model, synthetic_feature_df):
    from model import predict_future

    model, (*_, scaler) = trained_model
    for days in [1, 7, 14, 30]:
        preds = predict_future(model, synthetic_feature_df, scaler, days=days)
        assert len(preds) == days, f"Expected {days} predictions, got {len(preds)}"


def test_predict_future_prices_positive(trained_model, synthetic_feature_df):
    from model import predict_future

    model, (*_, scaler) = trained_model
    preds = predict_future(model, synthetic_feature_df, scaler, days=7)
    assert all(p > 0 for p in preds)


def test_directional_accuracy_in_range(trained_model):
    from model import evaluate_model

    model, (_, X_test, _, y_test, scaler) = trained_model
    metrics = evaluate_model(model, X_test, y_test, scaler)
    da = metrics["directional_accuracy"]
    if da is not None:
        assert 0.0 <= da <= 1.0


def test_scaler_persistence_and_metadata_loading(preprocessed, tmp_path, monkeypatch):
    import model as model_module
    from model import is_schema_valid, load_metadata, load_or_train, train_model

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    X_train, X_test, y_train, y_test, scaler = preprocessed

    # Train model & scaler (no feature_df, so WFV is skipped, but file structure still tested)
    trained_m, trained_s = train_model(
        X_train, y_train, X_test, y_test, ticker="PERSIST_TEST", scaler=scaler
    )

    base_dir = tmp_path / "PERSIST_TEST" / "lstm"
    assert (base_dir / "model.keras").exists()
    assert (base_dir / "scaler.joblib").exists()
    assert (base_dir / "metadata.json").exists()
    # Phase 3: walk-forward outputs always written (empty when no feature_df)
    assert (base_dir / "cross_validation.json").exists()
    assert (base_dir / "validation_results.json").exists()

    meta = load_metadata("PERSIST_TEST", "lstm")
    assert meta["schema_version"] == 2
    assert meta["feature_count"] == len(FEATURES)
    assert is_schema_valid("PERSIST_TEST", "lstm")

    # Load from cache
    loaded_m, loaded_s = load_or_train(
        "PERSIST_TEST", X_train, y_train, X_test, y_test, scaler=scaler
    )
    assert loaded_s is not None
    assert getattr(loaded_s, "data_min_", None) is not None


def test_build_bilstm_attention_model_outputs():
    from model import build_bilstm_attention_direction, build_bilstm_attention_regression

    m_dir = build_bilstm_attention_direction(forecast_days=7, num_features=len(FEATURES))
    assert len(m_dir.outputs) == 2, "Model should return exactly 2 outputs"
    assert m_dir.outputs[0].shape == (None, 7)
    assert m_dir.outputs[1].shape == (None, WINDOW_SIZE, 1)

    m_reg = build_bilstm_attention_regression(forecast_days=7, num_features=len(FEATURES))
    assert len(m_reg.outputs) == 2
    assert m_reg.outputs[0].shape == (None, 7)
    assert m_reg.outputs[1].shape == (None, WINDOW_SIZE, 1)


def test_train_model_writes_walk_forward_outputs(preprocessed, tmp_path, monkeypatch):
    """Phase 3: train_model must always write cross_validation.json and validation_results.json."""
    import model as model_module
    from model import train_model

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    X_train, X_test, y_train, y_test, scaler = preprocessed

    train_model(X_train, y_train, X_test, y_test, ticker="WFV_TEST", scaler=scaler)

    base_dir = tmp_path / "WFV_TEST" / "lstm"
    assert (base_dir / "cross_validation.json").exists()
    assert (base_dir / "validation_results.json").exists()

    with open(base_dir / "cross_validation.json") as f:
        cv = json.load(f)
    # Without feature_df, WFV is skipped but file is still written
    assert isinstance(cv, dict)

    with open(base_dir / "validation_results.json") as f:
        vr = json.load(f)
    assert isinstance(vr, list)


def test_train_model_walk_forward_with_feature_df(
    preprocessed, large_synthetic_feature_df, tmp_path, monkeypatch
):
    """Phase 3: when feature_df is provided, WFV runs and produces fold-level diagnostics."""
    import model as model_module
    from data_pipeline import preprocess
    from model import load_cross_validation, load_validation_results, train_model

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))

    # Use large dataset so each fold has enough rows for WINDOW_SIZE + forecast_days sequences
    X_train, X_test, y_train, y_test, scaler, _, _ = preprocess(large_synthetic_feature_df)

    train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        ticker="WFV_DF_TEST",
        scaler=scaler,
        model_type="lstm",
        feature_df=large_synthetic_feature_df,
    )

    cv = load_cross_validation("WFV_DF_TEST", "lstm")
    vr = load_validation_results("WFV_DF_TEST", "lstm")

    assert "folds_completed" in cv
    # At least some folds should complete on 2000 synthetic rows
    assert cv["folds_completed"] >= 1

    # Each fold entry must contain required keys
    for fold in vr:
        assert "fold" in fold
        assert "train_start" in fold
        assert "validation_start" in fold
        assert "residuals" in fold
        assert "actuals" in fold
        assert "predictions" in fold


def test_fold_residuals_structure(preprocessed, large_synthetic_feature_df, tmp_path, monkeypatch):
    """Phase 3: each residual row must contain date, actual, predicted, absolute_error."""
    import model as model_module
    from data_pipeline import preprocess
    from model import load_validation_results, train_model

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))

    # Use large dataset so folds produce residuals
    X_train, X_test, y_train, y_test, scaler, _, _ = preprocess(large_synthetic_feature_df)

    train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        ticker="RESID_TEST",
        scaler=scaler,
        model_type="lstm",
        feature_df=large_synthetic_feature_df,
    )

    vr = load_validation_results("RESID_TEST", "lstm")
    if not vr:
        pytest.skip("No folds completed on this dataset — skip residual structure test")

    first_fold = vr[0]
    assert len(first_fold["residuals"]) > 0
    first_row = first_fold["residuals"][0]
    assert "date" in first_row
    assert "actual" in first_row
    assert "residual" in first_row
    assert "absolute_error" in first_row


def test_metadata_includes_validation_config(preprocessed, tmp_path, monkeypatch):
    """Phase 3: metadata.json must record validation_method and validation_folds."""
    import model as model_module
    from model import load_metadata, train_model

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    X_train, X_test, y_train, y_test, scaler = preprocessed

    train_model(X_train, y_train, X_test, y_test, ticker="META_TEST", scaler=scaler)

    meta = load_metadata("META_TEST", "lstm")
    assert "validation_method" in meta
    assert "validation_folds" in meta
    assert meta["validation_folds"] == 5


def test_train_model_attention_caching_and_metrics(preprocessed, tmp_path, monkeypatch):
    import model as model_module
    from model import load_metrics, train_model

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    X_train, X_test, y_train, y_test, scaler = preprocessed

    y_train_bin = (y_train > np.median(y_train)).astype(int)
    y_test_bin = (y_test > np.median(y_test)).astype(int)

    trained_m, trained_s = train_model(
        X_train,
        y_train_bin,
        X_test,
        y_test_bin,
        ticker="ATTN_TEST",
        scaler=scaler,
        model_type="bilstm_attention_direction",
    )

    base_dir = tmp_path / "ATTN_TEST" / "bilstm_attention_direction"
    assert (base_dir / "model.keras").exists()
    assert (base_dir / "scaler.joblib").exists()
    assert (base_dir / "metrics.json").exists()

    with open(base_dir / "metrics.json") as f:
        metrics = json.load(f)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "naive_baseline" in metrics

    loaded_metrics = load_metrics("ATTN_TEST", model_type="bilstm_attention_direction")
    assert loaded_metrics == metrics


def test_load_or_train_graceful_overwrite(preprocessed, tmp_path, monkeypatch):
    import joblib

    import model as model_module
    from model import load_or_train

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    X_train, X_test, y_train, y_test, scaler = preprocessed

    base_dir = tmp_path / "CORRUPT" / "lstm"
    base_dir.mkdir(parents=True, exist_ok=True)

    model_path = base_dir / "model.keras"
    scaler_path = base_dir / "scaler.joblib"
    meta_path = base_dir / "metadata.json"

    model_path.write_text("this is not a valid keras model")
    meta_path.write_text('{"schema_version": 2, "features": ["Open"]}')
    joblib.dump(scaler, str(scaler_path))

    # load_or_train should catch exception and retrain safely
    loaded_m, loaded_s = load_or_train(
        "CORRUPT", X_train, y_train, X_test, y_test, scaler=scaler, model_type="lstm"
    )

    assert loaded_m is not None
    assert model_path.exists()
