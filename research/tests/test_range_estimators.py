import math

import numpy as np
import pandas as pd
import pytest

from research.volatility_structure.range_estimators import range_variances


def _ohlc(rows):
    frame = pd.DataFrame(rows, columns=["o", "h", "l", "c"])
    return frame["o"], frame["h"], frame["l"], frame["c"]


def test_closed_forms_match_hand_calculation():
    o, h, lo, c = _ohlc([(100.0, 102.0, 99.0, 101.0)])
    out = range_variances(o, h, lo, c)
    assert out["parkinson_var"].iloc[0] == pytest.approx(
        (math.log(102.0 / 99.0) ** 2) / (4.0 * math.log(2.0))
    )
    assert out["garman_klass_var"].iloc[0] == pytest.approx(
        0.5 * math.log(102.0 / 99.0) ** 2
        - (2.0 * math.log(2.0) - 1.0) * math.log(101.0 / 100.0) ** 2
    )
    assert out["rogers_satchell_var"].iloc[0] == pytest.approx(
        math.log(102.0 / 101.0) * math.log(102.0 / 100.0)
        + math.log(99.0 / 101.0) * math.log(99.0 / 100.0)
    )
    # Single row can never fill a trailing window.
    assert np.isnan(out["yang_zhang_var"].iloc[0])


def test_flat_day_parkinson_is_zero_and_invalid_rows_are_nan():
    o, h, lo, c = _ohlc(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 99.0, 101.0, 100.0),  # high < low: invalid print
            (100.0, 103.0, 99.0, 101.0),
        ]
    )
    out = range_variances(o, h, lo, c)
    assert out["parkinson_var"].iloc[0] == 0.0
    assert out[["parkinson_var", "garman_klass_var", "rogers_satchell_var"]].iloc[1].isna().all()
    assert np.isfinite(out["parkinson_var"].iloc[2])


def test_estimators_nonnegative_on_gbm_and_yang_zhang_warms_up():
    rng = np.random.default_rng(7)
    n = 300
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    index = pd.bdate_range("2022-01-03", periods=n)
    frame = pd.DataFrame({"o": open_, "h": high, "l": low, "c": close}, index=index)
    out = range_variances(frame["o"], frame["h"], frame["l"], frame["c"], yang_zhang_window=22)
    # Parkinson / Rogers-Satchell / Yang-Zhang are term-wise non-negative.
    # Garman-Klass is unbiased but can print negative single-day values on
    # strong trend days; it must stay finite (the study harness floors it).
    assert (out[["parkinson_var", "rogers_satchell_var"]].dropna() >= 0).all().all()
    assert np.isfinite(out["garman_klass_var"].dropna()).all()
    assert out["yang_zhang_var"].iloc[:22].isna().all()
    assert np.isfinite(out["yang_zhang_var"].iloc[22:]).all()
    assert (out["yang_zhang_var"].iloc[22:] >= 0).all()


def test_empty_and_misaligned_inputs_raise():
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="empty"):
        range_variances(empty, empty, empty, empty)
    o, h, lo, c = _ohlc([(100.0, 102.0, 99.0, 101.0)])
    shifted = c.copy()
    shifted.index = shifted.index + 1
    with pytest.raises(ValueError, match="misaligned"):
        range_variances(o, h, lo, shifted)
    with pytest.raises(ValueError, match="window"):
        range_variances(o, h, lo, c, yang_zhang_window=1)
