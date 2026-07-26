# backend/tests/test_model.py
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pytest

from config import FEATURES, MAX_FORECAST_DAYS, SCHEMA_VERSION, WINDOW_SIZE


def _active_dir(base_dir):
    pointer = json.loads((base_dir / "current.json").read_text())
    return base_dir / "versions" / pointer["version"]


def _locked_append(lock_path: str, output_path: str, value: str) -> None:
    from model import _process_file_lock

    with _process_file_lock(Path(lock_path), timeout=5):
        output = Path(output_path)
        existing = output.read_text() if output.exists() else ""
        time.sleep(0.1)
        output.write_text(existing + value)


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


def test_direction_metrics_use_all_horizons_and_training_majority_baseline():
    from model import _compute_fold_metrics_direction

    actual = np.array([[1, 0], [1, 0]])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2]])
    training = np.ones((3, 2), dtype=int)
    metrics = _compute_fold_metrics_direction(actual, probabilities, training)
    assert metrics["direction_accuracy"] == 1.0
    assert metrics["naive_baseline"] == 0.5


def test_scaler_persistence_and_metadata_loading(preprocessed, tmp_path, monkeypatch):
    import model as model_module
    from model import is_schema_valid, load_metadata, load_or_train, train_model

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    X_train, X_test, y_train, y_test, scaler = preprocessed

    # Train model & scaler (no feature_df, so WFV is skipped, but file structure still tested)
    trained_m, trained_s = train_model(
        X_train, y_train, X_test, y_test, ticker="PERSISTTEST", scaler=scaler
    )

    base_dir = tmp_path / "PERSISTTEST" / "lstm"
    active_dir = _active_dir(base_dir)
    assert (active_dir / "model.keras").exists()
    assert (active_dir / "scaler.json").exists()
    assert (active_dir / "metadata.json").exists()
    assert (active_dir / "integrity.json").exists()
    # Phase 3: walk-forward outputs always written (empty when no feature_df)
    assert (active_dir / "cross_validation.json").exists()
    assert (active_dir / "validation_results.json").exists()

    meta = load_metadata("PERSISTTEST", "lstm")
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["feature_count"] == len(FEATURES)
    assert is_schema_valid("PERSISTTEST", "lstm")

    # Load from cache
    loaded_m, loaded_s = load_or_train(
        "PERSISTTEST", X_train, y_train, X_test, y_test, scaler=scaler
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

    train_model(X_train, y_train, X_test, y_test, ticker="WFVTEST", scaler=scaler)

    base_dir = tmp_path / "WFVTEST" / "lstm"
    active_dir = _active_dir(base_dir)
    assert (active_dir / "cross_validation.json").exists()
    assert (active_dir / "validation_results.json").exists()

    with open(active_dir / "cross_validation.json") as f:
        cv = json.load(f)
    # Without feature_df, WFV is skipped but file is still written
    assert isinstance(cv, dict)

    with open(active_dir / "validation_results.json") as f:
        vr = json.load(f)
    assert isinstance(vr, list)


def test_train_model_walk_forward_with_feature_df(
    preprocessed, large_synthetic_feature_df, tmp_path, monkeypatch
):
    """Phase 3: when feature_df is provided, WFV runs and produces fold-level diagnostics."""
    import model as model_module
    from config import ValidationConfig
    from data_pipeline import preprocess
    from model import load_cross_validation, load_validation_results, train_model

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(
        model_module,
        "VALIDATION_CONFIG",
        ValidationConfig(folds=1, min_train_size=300, horizon=30),
    )

    # Use large dataset so each fold has enough rows for WINDOW_SIZE + forecast_days sequences
    X_train, X_test, y_train, y_test, scaler, _, _ = preprocess(large_synthetic_feature_df)

    train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        ticker="WFVDFTEST",
        scaler=scaler,
        model_type="lstm",
        feature_df=large_synthetic_feature_df,
    )

    cv = load_cross_validation("WFVDFTEST", "lstm")
    vr = load_validation_results("WFVDFTEST", "lstm")

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
    from config import ValidationConfig
    from data_pipeline import preprocess
    from model import load_validation_results, train_model

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(
        model_module,
        "VALIDATION_CONFIG",
        ValidationConfig(folds=1, min_train_size=300, horizon=30),
    )

    # Use large dataset so folds produce residuals
    X_train, X_test, y_train, y_test, scaler, _, _ = preprocess(large_synthetic_feature_df)

    train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        ticker="RESIDTEST",
        scaler=scaler,
        model_type="lstm",
        feature_df=large_synthetic_feature_df,
    )

    vr = load_validation_results("RESIDTEST", "lstm")
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

    train_model(X_train, y_train, X_test, y_test, ticker="METATEST", scaler=scaler)

    meta = load_metadata("METATEST", "lstm")
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
        ticker="ATTNTEST",
        scaler=scaler,
        model_type="bilstm_attention_direction",
    )

    base_dir = tmp_path / "ATTNTEST" / "bilstm_attention_direction"
    active_dir = _active_dir(base_dir)
    assert (active_dir / "model.keras").exists()
    assert (active_dir / "scaler.json").exists()
    assert (active_dir / "metrics.json").exists()

    with open(active_dir / "metrics.json") as f:
        metrics = json.load(f)

    assert metrics["metric_source"] == "unavailable"

    loaded_metrics = load_metrics("ATTNTEST", model_type="bilstm_attention_direction")
    assert loaded_metrics == metrics


def test_load_or_train_rejects_legacy_corruption(preprocessed, tmp_path, monkeypatch):
    import model as model_module
    from model import load_or_train

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    X_train, X_test, y_train, y_test, scaler = preprocessed

    base_dir = tmp_path / "CORRUPT" / "lstm"
    base_dir.mkdir(parents=True, exist_ok=True)

    model_path = base_dir / "model.keras"
    meta_path = base_dir / "metadata.json"

    model_path.write_text("this is not a valid keras model")
    meta_path.write_text('{"schema_version": 2, "features": ["Open"]}')

    # load_or_train should catch exception and retrain safely
    loaded_m, loaded_s = load_or_train(
        "CORRUPT", X_train, y_train, X_test, y_test, scaler=scaler, model_type="lstm"
    )

    assert loaded_m is not None
    assert (_active_dir(base_dir) / "integrity.json").exists()


def test_validation_split_boundaries_and_no_evaluation_overlap():
    from config import ValidationConfig
    from model import generate_validation_splits

    for method in ("expanding", "rolling"):
        config = ValidationConfig(method=method, folds=3, min_train_size=100, horizon=30, gap=5)
        splits = generate_validation_splits(240, config)
        assert [
            (int(train[0]), int(train[-1]), int(val[0]), int(val[-1])) for train, val in splits
        ] == (
            [(0, 144, 150, 179), (0, 174, 180, 209), (0, 204, 210, 239)]
            if method == "expanding"
            else [(45, 144, 150, 179), (75, 174, 180, 209), (105, 204, 210, 239)]
        )
        for train, evaluation in splits:
            assert set(train).isdisjoint(evaluation)
            assert int(train[-1]) + config.gap < int(evaluation[0])


def test_evaluation_fold_is_not_used_for_fit_or_early_stopping(monkeypatch):
    import model as model_module

    captured = {}

    class FakeModel:
        def fit(self, X, y, **kwargs):
            captured["fit"] = X.copy()
            captured["fit_targets"] = y.copy()
            captured["early_stopping"] = kwargs["validation_data"][0].copy()
            return MagicMock(history={"loss": [1.0], "val_loss": [1.0]})

    from unittest.mock import MagicMock

    monkeypatch.setattr(model_module, "_build_model_for_type", lambda *_args: FakeModel())
    train = np.arange(20 * 2, dtype=float).reshape(20, 1, 2)
    targets = np.arange(20, dtype=float).reshape(20, 1)
    evaluation = np.arange(100, 110, dtype=float)
    model_module._train_single_fold(train, targets, "lstm", 1, 2)

    assert set(captured["fit"].ravel()).isdisjoint(evaluation)
    assert set(captured["early_stopping"].ravel()).isdisjoint(evaluation)
    assert set(captured["early_stopping"].ravel()).issubset(set(train.ravel()))


def test_deterministic_initialisation_is_repeatable():
    from model import build_lstm_model, set_reproducibility

    sample = np.ones((1, WINDOW_SIZE, len(FEATURES)), dtype=np.float32)
    set_reproducibility(123)
    first = build_lstm_model(3, len(FEATURES)).predict(sample, verbose=0)
    set_reproducibility(123)
    second = build_lstm_model(3, len(FEATURES)).predict(sample, verbose=0)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_tampered_artifact_is_rejected(preprocessed, tmp_path, monkeypatch):
    import model as model_module
    from model import ArtifactValidationError, _load_valid_artifact, train_model

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    X_train, X_test, y_train, y_test, scaler = preprocessed
    train_model(X_train, y_train, X_test, y_test, ticker="TAMPER", scaler=scaler)
    active_dir = _active_dir(tmp_path / "TAMPER" / "lstm")
    (active_dir / "scaler.json").write_text("{}")
    with pytest.raises(ArtifactValidationError, match="integrity"):
        _load_valid_artifact("TAMPER", "lstm", MAX_FORECAST_DAYS)


def test_horizon_mismatch_artifact_is_rejected(preprocessed, tmp_path, monkeypatch):
    import model as model_module
    from model import ArtifactValidationError, _load_valid_artifact, train_model

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    X_train, X_test, y_train, y_test, scaler = preprocessed
    train_model(X_train, y_train, X_test, y_test, ticker="HORIZON", scaler=scaler)
    with pytest.raises(ArtifactValidationError, match="incompatible"):
        _load_valid_artifact("HORIZON", "lstm", 3)


def test_process_artifact_lock_serialises_writers(tmp_path):
    context = multiprocessing.get_context("spawn")
    lock_path = str(tmp_path / "artifact.lock")
    output_path = str(tmp_path / "writes.txt")
    processes = [
        context.Process(target=_locked_append, args=(lock_path, output_path, value))
        for value in ("A", "B")
    ]
    for process in processes:
        process.start()
    for process in processes:
        # Spawned Windows workers import TensorFlow before acquiring the lock;
        # allow that one-time import without weakening the serialization assertion.
        process.join(60)
        assert process.exitcode == 0
    assert sorted(Path(output_path).read_text()) == ["A", "B"]


def test_storage_quota_evicts_oldest_unprotected_artifact(tmp_path, monkeypatch):
    import model as model_module

    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_module.settings, "model_max_count", 1)
    monkeypatch.setattr(model_module.settings, "model_max_storage_mb", 900)
    monkeypatch.setattr(model_module.settings, "model_min_free_mb", 10)
    for ticker in ("OLD", "KEEP"):
        root = tmp_path / ticker / "lstm"
        root.mkdir(parents=True)
        (root / "current.json").write_text('{"version":"v1"}')
    model_module.enforce_storage_quota(exclude=("KEEP", "lstm"))
    assert not (tmp_path / "OLD" / "lstm").exists()
    assert (tmp_path / "KEEP" / "lstm" / "current.json").exists()
