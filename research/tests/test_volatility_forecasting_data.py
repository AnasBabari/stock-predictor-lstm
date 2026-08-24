from __future__ import annotations

import numpy as np
import pandas as pd
from volatility_forecasting.contracts import VolatilityForecastProtocol
from volatility_forecasting.data import (
    build_volatility_panel_examples,
    causal_log_har_forecasts,
)


def _market_frame(rows: int = 420, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=rows)
    returns = rng.normal(0.0002, 0.012, rows)
    close = 80.0 * np.exp(np.cumsum(returns))
    overnight = rng.normal(0.0, 0.003, rows)
    open_price = close * np.exp(overnight)
    range_scale = np.abs(rng.normal(0.006, 0.002, rows))
    high = np.maximum(open_price, close) * np.exp(range_scale)
    low = np.minimum(open_price, close) * np.exp(-range_scale)
    volume = rng.integers(500_000, 3_000_000, rows).astype(float)
    return pd.DataFrame(
        {"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_causal_log_har_does_not_read_future_realizations() -> None:
    rng = np.random.default_rng(3)
    rv = pd.Series(np.exp(rng.normal(-9.0, 0.7, 180)))
    original = causal_log_har_forecasts(rv, (1, 7), minimum_history=60, refit_every=1)

    corrupted = rv.copy()
    corrupted.iloc[121:] *= 1000.0
    changed = causal_log_har_forecasts(corrupted, (1, 7), minimum_history=60, refit_every=1)

    np.testing.assert_allclose(original[120], changed[120], rtol=0, atol=0)
    assert not np.allclose(original[140], changed[140])


def test_causal_log_har_returns_positive_ordered_horizon_sums() -> None:
    rng = np.random.default_rng(4)
    rv = pd.Series(np.exp(rng.normal(-9.0, 0.4, 150)))
    forecasts = causal_log_har_forecasts(rv, (1, 3, 7), minimum_history=60)
    finite = forecasts[np.isfinite(forecasts).all(axis=1)]
    assert len(finite) > 0
    assert (finite > 0).all()
    assert (finite[:, 0] < finite[:, 1]).all()
    assert (finite[:, 1] < finite[:, 2]).all()


def test_incremental_log_har_matches_expanding_normal_equations() -> None:
    rng = np.random.default_rng(44)
    values = np.exp(rng.normal(-8.5, 0.5, 140))
    horizons = (1, 3, 7)
    actual = causal_log_har_forecasts(
        pd.Series(values),
        horizons,
        minimum_history=60,
        refit_every=5,
    )

    expected = np.full_like(actual, np.nan)
    horizon_columns = {horizon: column for column, horizon in enumerate(horizons)}
    coefficients = None
    last_refit = -5
    penalty = np.eye(4) * 1e-4
    penalty[0, 0] = 0.0
    for origin in range(22, len(values)):
        training_origins = range(21, origin)
        design = []
        response = []
        for training_origin in training_origins:
            history = values[: training_origin + 1]
            design.append(
                [
                    1.0,
                    np.log(history[-1]),
                    np.log(np.mean(history[-5:])),
                    np.log(np.mean(history[-22:])),
                ]
            )
            response.append(np.log(values[training_origin + 1]))
        if origin < 60 or len(design) < 20:
            continue
        if coefficients is None or origin - last_refit >= 5:
            matrix = np.asarray(design)
            coefficients = np.linalg.solve(
                matrix.T @ matrix + penalty,
                matrix.T @ np.asarray(response),
            )
            last_refit = origin
        history = list(values[: origin + 1])
        cumulative = 0.0
        for step in range(1, max(horizons) + 1):
            row = np.asarray(
                [
                    1.0,
                    np.log(history[-1]),
                    np.log(np.mean(history[-5:])),
                    np.log(np.mean(history[-22:])),
                ]
            )
            next_variance = max(float(np.exp(np.clip(row @ coefficients, -30.0, 5.0))), 1e-12)
            history.append(next_variance)
            cumulative += next_variance
            if step in horizon_columns:
                expected[origin, horizon_columns[step]] = cumulative

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_panel_examples_align_features_baseline_and_future_targets() -> None:
    protocol = VolatilityForecastProtocol(
        horizons=(1, 3, 7),
        embargo_sessions=7,
        minimum_train_sessions=100,
        validation_sessions=30,
    )
    frame = _market_frame()
    examples = build_volatility_panel_examples(
        {"AAA": frame},
        protocol,
        minimum_har_history=60,
    )

    assert examples.features.shape[1:] == (60, protocol.feature_count)
    assert examples.baseline_variance.shape[1] == 3
    assert examples.realized_variance.shape == examples.baseline_variance.shape
    assert examples.cumulative_returns.shape == examples.baseline_variance.shape
    assert examples.direction_classes.shape == examples.baseline_variance.shape
    assert set(examples.tickers) == {"AAA"}

    first_origin = pd.Timestamp(examples.origin_dates[0])
    origin_position = frame.index.get_loc(first_origin)
    expected_return = np.log(
        frame["Close"].iloc[origin_position + 7] / frame["Close"].iloc[origin_position]
    )
    assert examples.cumulative_returns[0, 2] == np.float32(expected_return)


def test_panel_examples_are_invariant_to_rows_after_an_origin() -> None:
    protocol = VolatilityForecastProtocol(
        horizons=(1, 3, 7),
        embargo_sessions=7,
        minimum_train_sessions=100,
        validation_sessions=30,
    )
    frame = _market_frame()
    baseline = build_volatility_panel_examples({"AAA": frame}, protocol, minimum_har_history=60)
    chosen = 20
    origin_date = pd.Timestamp(baseline.origin_dates[chosen])

    corrupted = frame.copy()
    future_mask = corrupted.index > origin_date
    corrupted.loc[future_mask, ["Open", "High", "Low", "Close"]] *= 1.7
    changed = build_volatility_panel_examples({"AAA": corrupted}, protocol, minimum_har_history=60)
    changed_row = int(np.flatnonzero(changed.origin_dates == baseline.origin_dates[chosen])[0])

    np.testing.assert_allclose(baseline.features[chosen], changed.features[changed_row])
    np.testing.assert_allclose(
        baseline.baseline_variance[chosen], changed.baseline_variance[changed_row]
    )
    # Labels deliberately describe the future and therefore should change.
    assert not np.allclose(
        baseline.realized_variance[chosen], changed.realized_variance[changed_row]
    )
