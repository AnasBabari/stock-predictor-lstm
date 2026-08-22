"""GARCH-LSTM hybrid volatility candidate (slice 9).

A two-branch model per spec §4.3.5: an econometric branch consumes causal
GARCH/EWMA/HAR forecasts (encoding clustering, leverage, and long memory),
while an LSTM branch learns nonlinear departures from those econometric
forecasts over a trailing window of returns and filtered variances.

Objective: QLIKE-compatible loss on the cumulative variance path —
mean(a/p − ln(a/p) − 1) with p = softplus output (+ε) and a = the realized
cumulative variance target from slice 6. Lower is better; the loss is
non-negative and zero iff the forecast matches every origin exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from panel.volatility import (
    _garch_filter,
    ewma_variance,
    fit_garch,
    fit_har,
    har_forecast_path,
)

DEFAULT_LOOKBACK = 20
EPSILON = 1e-10


def fit_econometric(returns_train: np.ndarray, *, gjr: bool = True) -> dict:
    """Fit GARCH/HAR parameters on the training slice (frozen at train time)."""
    return {"params": fit_garch(returns_train, gjr=gjr), "coef": fit_har(returns_train)}


def transform_econometric(
    econometric: dict,
    returns_full: np.ndarray,
    *,
    lookback: int = DEFAULT_LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal econometric features per origin using FROZEN parameters.

    Refitting on the evaluation series would shift the feature distribution
    between train and predict — the parameters are deliberately frozen.
    """
    params = econometric["params"]
    coef = econometric["coef"]
    garch_path = _garch_filter(np.asarray(returns_full, dtype=float), params)
    ewma_path = ewma_variance(np.asarray(returns_full, dtype=float))
    har_next = har_forecast_path(np.asarray(returns_full, dtype=float) ** 2, coef, horizon=1)
    ret = np.asarray(returns_full, dtype=float)

    eps = EPSILON
    features = np.column_stack(
        [
            np.log(garch_path + eps),
            np.log(ewma_path + eps),
            np.nan_to_num(har_next, nan=0.0) + eps,
            np.abs(ret),
            (ret < 0).astype(float),
        ]
    )
    window_ret = np.lib.stride_tricks.sliding_window_view(ret, lookback)
    window_vol = np.lib.stride_tricks.sliding_window_view(np.sqrt(ewma_path), lookback)
    windows = np.stack([window_ret, window_vol], axis=-1)
    return features, windows


def econometric_features(
    returns_train: np.ndarray,
    returns_full: np.ndarray,
    *,
    gjr: bool = True,
    lookback: int = DEFAULT_LOOKBACK,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit-on-train then transform the full series; returns diagnostics."""
    ecom = fit_econometric(returns_train, gjr=gjr)
    features, windows = transform_econometric(ecom, returns_full, lookback=lookback)
    return features, windows, {"econometric": ecom}


def build_dataset(
    returns: np.ndarray,
    *,
    horizon: int,
    econometric: dict,
    lookback: int = DEFAULT_LOOKBACK,
) -> dict[str, np.ndarray | list[int]]:
    """Assemble (windows, eco_features) for valid origins.

    `econometric` is the frozen fit_econometric() output — parameters never
    refit on evaluation data. Origins are determined purely by window/target
    availability; target finiteness is the CALLER's responsibility.
    """
    r = np.asarray(returns, dtype=float)
    features, windows = transform_econometric(econometric, r, lookback=lookback)
    n = len(r)
    valid = [t for t in range(lookback - 1, n - horizon) if not np.isnan(features[t]).any()]
    idx = np.asarray(valid, dtype=int)
    return {
        "origins": idx,
        "windows": windows[idx - (lookback - 1)],
        "features": features[idx],
    }


def qlike_loss_logparam(y_true, log_var_pred):
    """QLIKE with the head parameterized in LOG-variance space.

    y_true: realized variance (raw scale); log_var_pred: log(p).
    Loss = mean( a·e^{−u} + u − ln a − 1 ), whose gradient wrt u is
    1 − a/p — bounded even when p and a live at wildly different scales,
    unlike the raw-variance parameterization.
    """
    import tensorflow as tf  # type: ignore[import-untyped]

    p = tf.exp(tf.clip_by_value(log_var_pred, -30.0, 30.0))
    ratio = tf.maximum(y_true, EPSILON) / p
    return tf.reduce_mean(ratio - tf.math.log(ratio) - 1.0)


def build_garch_lstm(
    lookback: int,
    n_return_channels: int,
    n_econometric_features: int,
    horizon: int,
    *,
    lstm_units: tuple[int, int] = (16, 8),
    dropout: float = 0.1,
    log_var_bias_init: float | None = None,
) -> object:
    tf = _require_tf()
    layers = tf.keras.layers

    window_in = layers.Input(shape=(lookback, n_return_channels), name="window")
    eco_in = layers.Input(shape=(n_econometric_features,), name="econometric")

    h = layers.LSTM(lstm_units[0], return_sequences=True)(window_in)
    h = layers.Dropout(dropout)(h)
    h = layers.LSTM(lstm_units[1])(h)

    e = layers.Dense(8, activation="relu")(eco_in)
    merged = layers.Concatenate()([h, e])
    d = layers.Dense(8, activation="relu")(merged)
    # Linear LOG-variance output; exp() at inference guarantees positivity.
    bias_initializer = (
        tf.keras.initializers.Constant(log_var_bias_init)
        if log_var_bias_init is not None
        else "zeros"
    )
    out = layers.Dense(
        horizon,
        kernel_initializer=tf.keras.initializers.Zeros(),
        bias_initializer=bias_initializer,
    )(d)

    model = tf.keras.Model([window_in, eco_in], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4, clipnorm=1.0),
        loss=qlike_loss_logparam,
    )
    return model


class GarchLstmCandidate:
    """Volatility-only candidate: predicts the CUMULATIVE variance path."""

    name = "garch_lstm"

    def __init__(
        self,
        *,
        horizon: int,
        train_end: int,
        lookback: int = DEFAULT_LOOKBACK,
        epochs: int = 8,
        batch_size: int = 64,
        gjr: bool = True,
    ):
        self.horizon = horizon
        self.train_end = train_end
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.gjr = gjr
        self._model: Any | None = None
        self.history: Any | None = None
        self.econometric: dict | None = None

    def fit_returns(self, returns: np.ndarray) -> GarchLstmCandidate:
        r = np.asarray(returns, dtype=float)
        # Parameters are FROZEN at train time and reused at predict time;
        # refitting on the evaluation series would shift the feature
        # distribution between training and serving.
        self.econometric = fit_econometric(r[: self.train_end], gjr=self.gjr)
        dataset = build_dataset(
            r,
            horizon=self.horizon,
            econometric=self.econometric,
            lookback=self.lookback,
        )
        rv_daily = r**2
        origins = np.asarray(dataset["origins"], dtype=int)
        # Daily-variance matrix [n, horizon]: rv at t+1..t+h per origin.
        labels = np.column_stack([rv_daily[origins + 1 + step] for step in range(self.horizon)])
        finite_rows = np.isfinite(labels).all(axis=1) & (labels > 0).all(axis=1)
        windows = dataset["windows"][finite_rows]
        features = dataset["features"][finite_rows]
        # Labels stay on the RAW variance scale — qlike_loss_logparam
        # interprets y_true as raw realized variance and applies exp() to the
        # model's log-space output itself. Taking np.log here too would feed
        # floored garbage into the ratio term.
        labels = labels[finite_rows]

        mean_daily_log_var = float(np.mean(np.log(labels)))
        model: Any = build_garch_lstm(
            self.lookback,
            windows.shape[-1],
            features.shape[-1],
            self.horizon,
            log_var_bias_init=mean_daily_log_var,
        )
        self.history = model.fit(
            {"window": windows, "econometric": features},
            labels,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            verbose=0,
        )
        self._model = model
        return self

    def predict(self, returns: np.ndarray) -> np.ndarray:
        if self._model is None or self.econometric is None:
            raise RuntimeError("GarchLstmCandidate used before fit().")
        dataset = build_dataset(
            returns,
            horizon=self.horizon,
            econometric=self.econometric,  # frozen parameters
            lookback=self.lookback,
        )
        log_var = self._model.predict(
            {"window": dataset["windows"], "econometric": dataset["features"]},
            verbose=0,
        )
        return np.exp(np.clip(log_var, -30.0, 30.0))


def _require_tf():
    try:
        import tensorflow as tf  # type: ignore[import-untyped]
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"{exc.__class__.__name__}: TensorFlow is required for "
            "GarchLstmCandidate but is not installed."
        ) from exc
    return tf
