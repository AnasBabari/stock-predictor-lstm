"""Slice-5 tests: causal v5 feature groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panel.features import (
    FEATURE_COLUMNS_V5,
    add_cross_sectional_ranks,
    build_features_v5,
)

ROWS = 320


def make_ohlcv(rows: int = ROWS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=rows)
    rets = rng.normal(0.0005, 0.012, rows)
    close = 100 * np.exp(np.cumsum(rets))
    openp = close * np.exp(rng.normal(0, 0.004, rows))
    high = np.maximum(openp, close) * np.exp(np.abs(rng.normal(0, 0.003, rows)))
    low = np.minimum(openp, close) * np.exp(-np.abs(rng.normal(0, 0.003, rows)))
    volume = rng.integers(500_000, 5_000_000, rows).astype(float)
    return pd.DataFrame(
        {"Open": openp, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return build_features_v5(make_ohlcv())


def test_all_schema_columns_present_and_ordered_groups(features: pd.DataFrame) -> None:
    for column in FEATURE_COLUMNS_V5:
        assert column in features.columns


def test_causality_perturbing_future_rows_cannot_change_the_past() -> None:
    clean = build_features_v5(make_ohlcv(seed=11))
    perturbed_input = make_ohlcv(seed=11)
    k = len(perturbed_input) // 2
    perturbed_input.iloc[k:, :] *= 1.7
    dirty = build_features_v5(perturbed_input)
    numeric = [c for c in FEATURE_COLUMNS_V5 if pd.api.types.is_numeric_dtype(clean[c])]
    for column in numeric:
        before = clean[column].iloc[: k - 1].to_numpy()
        after = dirty[column].iloc[: k - 1].to_numpy()
        np.testing.assert_allclose(before, after, equal_nan=True)


def test_return_structure_formulas() -> None:
    df = make_ohlcv(30)
    out = build_features_v5(df)
    expected_ret1 = np.log(out["Close"] / out["Close"].shift(1))
    np.testing.assert_allclose(out["Return_1D"], expected_ret1)
    expected_on = np.log(out["Open"] / out["Close"].shift(1))
    np.testing.assert_allclose(out["Overnight_Return"], expected_on)
    expected_o2c = np.log(out["Close"] / out["Open"])
    np.testing.assert_allclose(out["OpenToClose_Return"], expected_o2c)
    expected_range = np.log(out["High"] / out["Low"])
    np.testing.assert_allclose(out["HL_Range_Log"], expected_range)


def test_streaks_reset_to_zero_on_opposite_moves() -> None:
    out = build_features_v5(make_ohlcv(60, seed=3))
    up, down = out["Up_Streak"], out["Down_Streak"]
    assert (
        up[up > 0].index == out["Return_1D"][out["Return_1D"] > 0].reindex(up[up > 0].index).index
    ).all()
    assert (down.loc[up.shift(1) > down.shift(1)].dropna().astype(int) >= 0).all()
    # Where the return is exactly zero both streaks are zero.
    zero_rows = out.index[out["Return_1D"] == 0]
    if len(zero_rows):
        assert (out.loc[zero_rows, "Up_Streak"] == 0).all()


def test_volatility_columns_finite_after_warmup_and_percentile_bounded() -> None:
    out = build_features_v5(make_ohlcv())
    warm = out.iloc[260:]
    for column in ["Vol_C2C_20", "EWMA_Var", "Vol_Of_Vol_20", "Vol_Percentile_252"]:
        values = warm[column].to_numpy()
        assert np.isfinite(values).all(), column
    pct = warm["Vol_Percentile_252"].to_numpy()
    assert ((pct > 0) & (pct <= 1)).all()


def test_regime_labels_are_from_the_allowed_vocabularies() -> None:
    out = build_features_v5(make_ohlcv())
    assert set(out["Regime_Trend"].dropna()) <= {"bull", "bear", "range"}
    assert set(out["Regime_Volatility"].dropna()) <= {"calm", "normal", "stressed"}
    assert set(out["Regime_Liquidity"].dropna()) <= {"liquid", "normal", "illiquid"}


def test_cross_sectional_ranks_are_same_date_across_tickers() -> None:
    a = make_ohlcv(120, seed=1)
    b = make_ohlcv(120, seed=2)
    base_cols = ["Return_20D", "Vol_C2C_20", "Log_Dollar_Volume"]
    panels = {t: build_features_v5(f) for t, f in (("AAA", a), ("BBB", b))}
    ranked = add_cross_sectional_ranks(panels, base_cols)
    assert set(ranked) == {"AAA", "BBB"}
    date = ranked["AAA"].index[-40]
    pair = [
        ranked["AAA"].loc[date, "Return_20D"],
        ranked["BBB"].loc[date, "Return_20D"],
    ]
    ranks = [
        ranked["AAA"].loc[date, "Return_20D_XSRank"],
        ranked["BBB"].loc[date, "Return_20D_XSRank"],
    ]
    if pair[0] < pair[1]:
        assert ranks[0] < ranks[1]
    elif pair[0] > pair[1]:
        assert ranks[0] > ranks[1]


def test_warmup_rows_remain_nan_not_zero() -> None:
    out = build_features_v5(make_ohlcv())
    assert out["Vol_C2C_20"].iloc[:19].isna().all()


def test_deployable_feature_contract_validates_exact_ordered_subset() -> None:
    from panel.features import (
        DEPLOYABLE_FEATURE_COLUMNS_V5,
        DEPLOYABLE_SCHEMA_VERSION,
        DeployableFeatureContract,
    )

    contract = DeployableFeatureContract()
    assert len(contract.feature_names) == 26
    # Valid validation passes
    contract.validate(
        list(DEPLOYABLE_FEATURE_COLUMNS_V5),
        DEPLOYABLE_SCHEMA_VERSION,
        "v5_robust",
    )

    # Feature order mismatch fails
    reordered = list(DEPLOYABLE_FEATURE_COLUMNS_V5)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="Feature contract mismatch"):
        contract.validate(reordered, DEPLOYABLE_SCHEMA_VERSION, "v5_robust")

    # Missing feature fails
    with pytest.raises(ValueError, match="Feature contract mismatch"):
        contract.validate(
            list(DEPLOYABLE_FEATURE_COLUMNS_V5[:-1]), DEPLOYABLE_SCHEMA_VERSION, "v5_robust"
        )

    # Stale schema fails
    with pytest.raises(ValueError, match="Schema version mismatch"):
        contract.validate(list(DEPLOYABLE_FEATURE_COLUMNS_V5), "deployable_v4", "v5_robust")


def test_deployable_robust_scaler_roundtrip_and_transform_parity() -> None:
    from panel.features import DEPLOYABLE_FEATURE_COLUMNS_V5, DeployableRobustScaler

    rng = np.random.default_rng(42)
    data = rng.normal(10.0, 2.0, (100, len(DEPLOYABLE_FEATURE_COLUMNS_V5)))

    scaler = DeployableRobustScaler.fit(data)
    d = scaler.to_dict()
    restored = DeployableRobustScaler.from_dict(d)

    assert scaler.median == restored.median
    assert scaler.iqr == restored.iqr

    scaled = scaler.transform(data)
    assert scaled.shape == data.shape
    # Median should be approximately zero
    np.testing.assert_allclose(
        np.median(scaled, axis=0), np.zeros(len(DEPLOYABLE_FEATURE_COLUMNS_V5)), atol=1e-5
    )
