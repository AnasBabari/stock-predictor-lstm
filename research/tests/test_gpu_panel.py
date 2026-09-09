import numpy as np
import pandas as pd

from research.volatility_structure.gpu_panel import (
    FEATURE_COLUMNS,
    build_ticker_features,
    qlike_objective,
)


def _ohlc_frame(n=300, seed=31):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    index = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1_000_000},
        index=index,
    )


def _spy_frame_like(frame):
    rng = np.random.default_rng(77)
    spy = frame["Close"] * np.exp(np.cumsum(rng.normal(0, 0.0002, len(frame))))
    return pd.DataFrame({"SPY": spy}, index=frame.index)


def test_feature_schema_contract_and_warmup():
    frame = _ohlc_frame()
    out = build_ticker_features(frame, "US", spy=_spy_frame_like(frame))
    assert list(out.columns) == list(FEATURE_COLUMNS)
    assert "sector" not in " ".join(out.columns).lower()
    # Warm-up is dominated by the 60-session rolling variance.
    assert out.iloc[:59].isna().any(axis=None)
    assert np.isfinite(out.iloc[200:].to_numpy()).all()


def test_panel_features_causal_under_post_origin_mutation():
    frame = _ohlc_frame()
    t0 = frame.index[200]
    base = build_ticker_features(frame, "US")
    mutated = frame.copy()
    for column in ("Open", "High", "Low", "Close"):
        mutated.loc[mutated.index > t0, column] *= 10.0
    new = build_ticker_features(mutated, "US")
    before = frame.index <= t0
    pd.testing.assert_frame_equal(base.loc[before], new.loc[before])
    assert (base["rv_20"].dropna() != new["rv_20"].dropna()).any()


def test_qlike_objective_matches_finite_differences_and_optimum():
    objective = qlike_objective()

    class FakeDMatrix:
        def __init__(self, labels):
            self._labels = np.asarray(labels, dtype=float)

        def get_label(self):
            return self._labels

    rng = np.random.default_rng(9)
    labels = np.abs(rng.normal(0.001, 0.0005, 50)) + 1e-6
    preds = rng.normal(-7.0, 1.0, 50)
    grad, hess = objective(preds, FakeDMatrix(labels))
    assert np.isfinite(grad).all() and (hess > 0).all()

    def loss(z):
        return float(np.sum(labels * np.exp(-z) + z))

    eps = 1e-6
    num_grad = np.array(
        [(loss(preds + eps * e) - loss(preds - eps * e)) / (2 * eps) for e in np.eye(len(preds))]
    )
    np.testing.assert_allclose(grad, num_grad, rtol=1e-4, atol=1e-6)
    # At z = log y the gradient vanishes: the objective targets log-mean.
    zero = objective(np.log(labels), FakeDMatrix(labels))[0]
    np.testing.assert_allclose(zero, np.zeros_like(zero), atol=1e-9)
