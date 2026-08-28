from __future__ import annotations

import numpy as np
import pandas as pd

from research.ndx100.retrospective_benchmark import (
    FEATURE_COLUMNS,
    compute_causal_features,
    fit_train_data,
    forecast_ridge,
)


def _history(rows: int = 160) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-02", periods=rows)
    returns = 0.0004 + 0.01 * np.sin(np.arange(rows) / 7.0)
    closes = 100.0 * np.exp(np.cumsum(returns))
    volumes = 1_000_000.0 + 50_000.0 * np.cos(np.arange(rows) / 5.0)
    return pd.DataFrame({"close": closes, "volume": volumes}, index=index)


def test_causal_features_are_invariant_to_appended_future_rows() -> None:
    history = _history()
    prefix = history.iloc[:120]
    before = compute_causal_features(prefix.close.to_numpy(), prefix.volume.to_numpy())
    after = compute_causal_features(history.close.to_numpy(), history.volume.to_numpy()).iloc[:120]
    np.testing.assert_allclose(before[FEATURE_COLUMNS], after[FEATURE_COLUMNS], equal_nan=True)


def test_training_target_is_next_return_and_excludes_terminal_unknown() -> None:
    history = _history(100)
    x, y, features = fit_train_data(history)
    expected = np.log(history.close).diff().shift(-1)
    valid = features[FEATURE_COLUMNS].notna().all(axis=1) & expected.notna()
    np.testing.assert_allclose(y, expected.loc[valid])
    assert len(x) == int(valid.sum())
    assert history.index[-1] not in features.index[valid]


def test_ridge_forecast_is_finite_and_does_not_read_appended_future() -> None:
    history = _history()
    origin_history = history.iloc[:130]
    expected = forecast_ridge(origin_history, 5)
    mutated_future = history.copy()
    mutated_future.iloc[130:, mutated_future.columns.get_loc("close")] *= 10.0
    actual = forecast_ridge(mutated_future.iloc[:130], 5)
    np.testing.assert_allclose(actual, expected)
    assert expected.shape == (5,)
    assert np.isfinite(expected).all()
