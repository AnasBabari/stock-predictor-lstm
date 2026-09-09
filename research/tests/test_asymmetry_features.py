import math

import numpy as np
import pandas as pd
import pytest

from research.volatility_structure.asymmetry_features import asymmetry_features


def test_hand_computed_sign_conditioning():
    closes = pd.Series(
        [100.0, 110.0, 99.0, 99.0],
        index=pd.bdate_range("2024-01-02", periods=4),
    )
    out = asymmetry_features(closes)
    r_up, r_down = math.log(1.1), math.log(0.9)
    # First row has no prior close.
    assert out.iloc[0].isna().all()
    assert out["downside_sq"].iloc[1] == 0.0
    assert out["upside_sq"].iloc[1] == pytest.approx(r_up**2)
    assert out["neg_indicator"].iloc[1] == 0.0
    assert out["neg_return"].iloc[1] == 0.0
    assert out["downside_sq"].iloc[2] == pytest.approx(r_down**2)
    assert out["upside_sq"].iloc[2] == 0.0
    assert out["neg_indicator"].iloc[2] == 1.0
    assert out["neg_return"].iloc[2] == pytest.approx(r_down)
    # Exact zero return: no side claims it.
    assert out["downside_sq"].iloc[3] == 0.0
    assert out["upside_sq"].iloc[3] == 0.0
    assert out["neg_indicator"].iloc[3] == 0.0
    assert out["neg_return"].iloc[3] == 0.0


def test_nan_propagates_and_first_valid_index_is_one():
    closes = pd.Series(
        [100.0, np.nan, 105.0, 103.0],
        index=pd.bdate_range("2024-01-02", periods=4),
    )
    out = asymmetry_features(closes)
    assert out.iloc[0].isna().all()
    assert out.iloc[1].isna().all()
    assert out.iloc[2].isna().all()
    assert np.isfinite(out.iloc[3].to_numpy()).all()
    assert out["neg_indicator"].iloc[3] == 1.0


def test_columns_partition_squared_returns():
    rng = np.random.default_rng(11)
    closes = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
    out = asymmetry_features(closes)
    assert out.iloc[0].isna().all()
    assert np.isfinite(out.iloc[1:].to_numpy()).all()
    logret_sq = np.log(closes / closes.shift(1)).to_numpy()[1:] ** 2
    np.testing.assert_allclose(
        (out["downside_sq"].iloc[1:] + out["upside_sq"].iloc[1:]).to_numpy(),
        logret_sq,
        rtol=1e-12,
    )


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        asymmetry_features(pd.Series([], dtype=float))
