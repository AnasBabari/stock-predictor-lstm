import numpy as np
import pytest

from experiments.baselines import SmallTCNForecaster

try:
    import torch
except ImportError:
    torch = None

TORCH_MISSING = torch is None
requires_torch = pytest.mark.skipif(TORCH_MISSING, reason="requires the opt-in torch dependency")


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    """Tiny trending series in the existing runner-test style."""

    close = np.arange(1.0, 181.0)
    lookback = 5
    horizons = (1, 3)
    samples = len(close) - lookback - max(horizons) + 1
    windows = np.lib.stride_tricks.sliding_window_view(close, lookback)[:samples]
    features = np.stack([windows, np.ones_like(windows)], axis=-1)
    origins = close[lookback - 1 : lookback - 1 + samples]
    targets = np.column_stack(
        [np.log(close[lookback - 1 + h : lookback - 1 + h + samples] / origins) for h in horizons]
    )
    return features, targets


def test_construction_without_torch_raises_clear_error():
    if not TORCH_MISSING:
        pytest.skip("torch is installed; the guarded-import path is unreachable.")
    with pytest.raises(RuntimeError, match="PyTorch"):
        SmallTCNForecaster()


def test_invalid_settings_rejected():
    if TORCH_MISSING:
        pytest.skip("construction guards run only where torch is importable.")
    with pytest.raises(ValueError):
        SmallTCNForecaster(channels=0)
    with pytest.raises(ValueError):
        SmallTCNForecaster(learning_rate=0.0)


@requires_torch
def test_fit_predict_shapes():
    features, targets = _fixture()
    forecaster = SmallTCNForecaster(channels=8, blocks=2, epochs=4).fit(features, targets)
    predicted = forecaster.predict(features)
    assert predicted.shape == targets.shape
    assert np.isfinite(predicted).all()


@requires_torch
def test_predictions_are_deterministic_under_fixed_seed():
    features, targets = _fixture()
    first = SmallTCNForecaster(channels=8, blocks=2, epochs=4, seed=7).fit(features, targets)
    second = SmallTCNForecaster(channels=8, blocks=2, epochs=4, seed=7).fit(features, targets)
    assert np.array_equal(first.predict(features), second.predict(features))


@requires_torch
def test_refit_rebuilds_and_predicts():
    features, targets = _fixture()
    split = len(features) - 20
    forecaster = SmallTCNForecaster(channels=8, blocks=2, epochs=4, seed=11).fit(
        features[:split],
        targets[:split],
        validation_data=(features[split:], targets[split:]),
    )
    assert forecaster.selected_epoch in range(1, 5)
    forecaster.refit(features, targets)
    predicted = forecaster.predict(features)
    assert predicted.shape == targets.shape
    assert np.isfinite(predicted).all()


@requires_torch
def test_refit_requires_epoch_selection():
    forecaster = SmallTCNForecaster()
    features, targets = _fixture()
    with pytest.raises(ValueError):
        forecaster.refit(features, targets)


@requires_torch
def test_parameter_count_is_bounded():
    features, targets = _fixture()
    forecaster = SmallTCNForecaster(channels=8, blocks=2, epochs=1).fit(features, targets)
    count = forecaster.parameter_count()
    # Small research-grade footprint: well above zero, far below the LSTM.
    assert 0 < count < 50_000


@requires_torch
def test_metadata_reports_configuration():
    features, targets = _fixture()
    forecaster = SmallTCNForecaster(channels=8, blocks=2, epochs=2, seed=3).fit(features, targets)
    metadata = forecaster.metadata()
    assert metadata["architecture"] == "small_tcn"
    assert metadata["target_type"] == "regression"
    assert metadata["channels"] == 8
    assert metadata["blocks"] == 2
    assert metadata["seed"] == 3
    assert metadata["selected_epoch"] in range(1, 3)
    assert metadata["parameter_count"] == forecaster.parameter_count()


@requires_torch
def test_predict_before_fit_raises():
    features, _ = _fixture()
    with pytest.raises(ValueError):
        SmallTCNForecaster().predict(features)
